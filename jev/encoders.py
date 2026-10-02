"""Encoder classifier heads returning the same typed answers as the LLM.

A decision over a closed option list does not need a decoder. Encode the state,
compare it against the options, and read a softmax off the scores -- one forward
pass, no token generation.

Encoders turn text into vectors:

* `FoundryEncoder`    - text-embedding-3-small served from Azure AI Foundry.
* `LocalBertEncoder`  - a distilled BERT (MiniLM) run locally via ONNX Runtime.
* `LocalCrossEncoder` - a BERT reranker scoring (state, option) pairs.

Heads turn those into typed decisions, trading cost against robustness:

* `EncoderChoiceClassifier`        - mean-pooled prototypes. Cheapest, but
  class-neutral filler dilutes the vector because pooling ignores the question.
* `LateInteractionChoiceClassifier`- per-option attention over cached token
  vectors. Same cost, roughly half the dilution damage.
* `CrossEncoderChoiceClassifier`   - full attention over state and option
  together. Most accurate; one forward pass per option, so cost is linear in
  the number of options.

All produce `ChoiceAnswer`, so they drop in wherever `JevClient` is used. Unlike
the LLM, the probabilities here are a real softmax over model outputs rather
than a number the model wrote as text.
"""

from __future__ import annotations

import time
from typing import Mapping, Protocol, Sequence

import numpy as np

from .primitives import ChoiceAnswer


class Encoder(Protocol):
    """Anything that turns text into L2-normalized vectors."""

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


class FoundryEncoder:
    """Encoder-only embedding model served from Azure AI Foundry."""

    def __init__(self, client, model: str = "text-embedding-3-small") -> None:
        self._client = client
        self._model = model
        self.input_tokens = 0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        response = self._client.embeddings.create(model=self._model, input=list(texts))
        self.input_tokens += response.usage.prompt_tokens
        vectors = np.array([item.embedding for item in response.data], dtype=np.float32)
        return _normalize_rows(vectors)


class LocalBertEncoder:
    """A distilled BERT encoder run locally through ONNX Runtime."""

    def __init__(
        self,
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_length: int = 256,
    ) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        onnx_path = hf_hub_download(model_id, filename="onnx/model.onnx")
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self._input_names = {node.name for node in self._session.get_inputs()}
        self._max_length = max_length

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        hidden, mask = self.encode_tokens(texts)
        # Mean-pool over real tokens only; padding must not dilute the vector.
        weights = mask[..., None]
        pooled = (hidden * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
        return _normalize_rows(pooled.astype(np.float32))

    def encode_tokens(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Token vectors before pooling, plus the attention mask.

        Keeping the token matrix is what makes query-conditioned pooling
        possible without re-encoding the state per option.
        """
        encoded = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="np",
        )
        feed = {
            name: encoded[name].astype(np.int64)
            for name in self._input_names
            if name in encoded
        }
        hidden = self._session.run(None, feed)[0].astype(np.float32)
        return hidden, encoded["attention_mask"].astype(np.float32)


class LocalCrossEncoder:
    """A BERT cross-encoder scoring (state, option) pairs in one pass.

    Unlike a bi-encoder, the option text is visible to attention while the state
    is read, so the model can weight the tokens that matter for *this* option and
    ignore class-neutral boilerplate. That costs one forward pass per option
    instead of one per state.
    """

    def __init__(
        self,
        model_id: str = "Xenova/ms-marco-MiniLM-L-6-v2",
        max_length: int = 256,
    ) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        onnx_path = hf_hub_download(model_id, filename="onnx/model.onnx")
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self._input_names = {node.name for node in self._session.get_inputs()}
        self._max_length = max_length

    def score(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        """Relevance logit for each (state, option) pair."""
        states = [state for state, _ in pairs]
        options = [option for _, option in pairs]
        encoded = self._tokenizer(
            states,
            options,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="np",
        )
        feed = {
            name: encoded[name].astype(np.int64)
            for name in self._input_names
            if name in encoded
        }
        logits = self._session.run(None, feed)[0]
        return logits.reshape(-1).astype(np.float32)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


class CrossEncoderChoiceClassifier:
    """Query-conditioned classifier head: score every option against the state.

    `probability` is a softmax over the per-option relevance logits.
    `confidence` is the sigmoid of the winning logit -- an absolute relevance
    score, so it still answers "does this match anything at all?".
    """

    def __init__(
        self,
        cross_encoder: LocalCrossEncoder,
        options: Mapping[str, str],
        temperature: float = 1.0,
        batch_size: int = 64,
    ) -> None:
        if len(options) < 2:
            raise ValueError("Need at least two options")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        self._cross_encoder = cross_encoder
        self._labels = tuple(options)
        self._descriptions = [options[label] for label in self._labels]
        self._temperature = temperature
        self._batch_size = batch_size
        self.fit_seconds = 0.0

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    def classify_batch(self, states: Sequence[str]) -> list[ChoiceAnswer]:
        states = list(states)
        pairs = [
            (state, description) for state in states for description in self._descriptions
        ]

        scores: list[np.ndarray] = []
        for start in range(0, len(pairs), self._batch_size):
            scores.append(self._cross_encoder.score(pairs[start : start + self._batch_size]))
        logit_matrix = np.concatenate(scores).reshape(len(states), len(self._labels))

        shifted = logit_matrix / self._temperature
        shifted -= shifted.max(axis=1, keepdims=True)
        exponentiated = np.exp(shifted)
        probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)

        answers: list[ChoiceAnswer] = []
        for row, logits in zip(probabilities, logit_matrix):
            winner = int(row.argmax())
            answers.append(
                ChoiceAnswer(
                    choice=self._labels[winner],
                    probabilities={
                        label: float(value) for label, value in zip(self._labels, row)
                    },
                    confidence=float(_sigmoid(logits[winner])),
                )
            )
        return answers

    def classify(self, state: str) -> ChoiceAnswer:
        return self.classify_batch([state])[0]


class LateInteractionChoiceClassifier:
    """Query-conditioned pooling at bi-encoder cost.

    The state is encoded once and kept as a token matrix instead of being pooled
    immediately. Each option then attends over those cached tokens, so pooling
    becomes a function of the question: tokens that are irrelevant to an option
    receive little weight rather than being averaged in regardless.

    This is the cheap half of what a cross-encoder does. Conditioning moves out
    of the transformer and into a matrix multiply, so cost stays at one forward
    pass per state no matter how many options there are.
    """

    def __init__(
        self,
        encoder: LocalBertEncoder,
        options: Mapping[str, str],
        temperature: float = 0.05,
        attention_temperature: float = 0.05,
        batch_size: int = 64,
    ) -> None:
        if len(options) < 2:
            raise ValueError("Need at least two options")
        if temperature <= 0 or attention_temperature <= 0:
            raise ValueError("temperatures must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        self._encoder = encoder
        self._labels = tuple(options)
        self._temperature = temperature
        self._attention_temperature = attention_temperature
        self._batch_size = batch_size

        start = time.perf_counter()
        self._prototypes = encoder.encode([options[label] for label in self._labels])
        self.fit_seconds = time.perf_counter() - start

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    def _answers_for(self, states: Sequence[str]) -> list[ChoiceAnswer]:
        hidden, mask = self._encoder.encode_tokens(list(states))
        tokens = hidden / np.clip(
            np.linalg.norm(hidden, axis=-1, keepdims=True), 1e-12, None
        )

        # token_scores[b, t, k] = affinity of token t for option k
        token_scores = np.einsum("btd,kd->btk", tokens, self._prototypes)

        # Attend per option over real tokens only.
        masked = np.where(mask[..., None] > 0, token_scores, -1e9)
        weights = masked / self._attention_temperature
        weights -= weights.max(axis=1, keepdims=True)
        weights = np.exp(weights)
        weights /= np.clip(weights.sum(axis=1, keepdims=True), 1e-12, None)

        # Query-dependent pooling, then score each option against its own view.
        pooled = np.einsum("btk,btd->bkd", weights, tokens)
        pooled /= np.clip(np.linalg.norm(pooled, axis=-1, keepdims=True), 1e-12, None)
        similarities = np.einsum("bkd,kd->bk", pooled, self._prototypes)

        logits = similarities / self._temperature
        logits -= logits.max(axis=1, keepdims=True)
        exponentiated = np.exp(logits)
        probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)

        answers: list[ChoiceAnswer] = []
        for row, similarity_row in zip(probabilities, similarities):
            winner = int(row.argmax())
            answers.append(
                ChoiceAnswer(
                    choice=self._labels[winner],
                    probabilities={
                        label: float(value) for label, value in zip(self._labels, row)
                    },
                    confidence=float(np.clip(similarity_row[winner], 0.0, 1.0)),
                )
            )
        return answers

    def classify_batch(self, states: Sequence[str]) -> list[ChoiceAnswer]:
        states = list(states)
        answers: list[ChoiceAnswer] = []
        for start in range(0, len(states), self._batch_size):
            answers.extend(self._answers_for(states[start : start + self._batch_size]))
        return answers

    def classify(self, state: str) -> ChoiceAnswer:
        return self._answers_for([state])[0]


class EncoderChoiceClassifier:
    """Nearest-class-mean classifier head over an encoder.

    `probability` is a softmax over cosine similarities -- relative, and always
    sums to 1. `confidence` is the raw cosine to the winning prototype --
    absolute, and the signal that detects "none of these". Both are needed: a
    state matching nothing still produces a confident-looking softmax.
    """

    def __init__(
        self,
        encoder: Encoder,
        options: Mapping[str, str],
        temperature: float = 0.05,
        batch_size: int = 64,
    ) -> None:
        if len(options) < 2:
            raise ValueError("Need at least two options")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        self._encoder = encoder
        self._labels = tuple(options)
        self._temperature = temperature
        self._batch_size = batch_size
        self.fit_seconds = 0.0

        start = time.perf_counter()
        # Prototypes are built once; classifying a clause is then one encode.
        self._prototypes = encoder.encode([options[label] for label in self._labels])
        self.fit_seconds = time.perf_counter() - start

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    def _answers_for(self, states: Sequence[str]) -> list[ChoiceAnswer]:
        vectors = self._encoder.encode(list(states))
        similarities = vectors @ self._prototypes.T

        logits = similarities / self._temperature
        logits -= logits.max(axis=1, keepdims=True)
        exponentiated = np.exp(logits)
        probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)

        answers: list[ChoiceAnswer] = []
        for row, similarity_row in zip(probabilities, similarities):
            winner = int(row.argmax())
            answers.append(
                ChoiceAnswer(
                    choice=self._labels[winner],
                    probabilities={
                        label: float(value) for label, value in zip(self._labels, row)
                    },
                    confidence=float(np.clip(similarity_row[winner], 0.0, 1.0)),
                )
            )
        return answers

    def classify_batch(self, states: Sequence[str]) -> list[ChoiceAnswer]:
        """Classify many states, chunked so one request cannot grow unbounded."""
        states = list(states)
        answers: list[ChoiceAnswer] = []
        for start in range(0, len(states), self._batch_size):
            answers.extend(self._answers_for(states[start : start + self._batch_size]))
        return answers

    def classify(self, state: str) -> ChoiceAnswer:
        return self._answers_for([state])[0]
