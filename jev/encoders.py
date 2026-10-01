"""Encoder-only classifier heads returning the same typed answers as the LLM.

A System One decision over a closed option list does not need a decoder. Encode
the state once, compare it against class prototypes, and read a softmax off the
similarities. That is a nearest-class-mean classifier head on top of a BERT-style
encoder -- one forward pass, no token generation.

Two backends, one interface:

* `FoundryEncoder`   - text-embedding-3-small served from Azure AI Foundry.
* `LocalBertEncoder` - a distilled BERT (MiniLM) run locally via ONNX Runtime.

Both produce `ChoiceAnswer`, so they drop into the pipeline wherever `JevClient`
is used. Unlike the LLM, the probabilities here are a real softmax over model
outputs rather than a number the model wrote as text.
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
        hidden = self._session.run(None, feed)[0]

        # Mean-pool over real tokens only; padding must not dilute the vector.
        mask = encoded["attention_mask"][..., None].astype(np.float32)
        pooled = (hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        return _normalize_rows(pooled.astype(np.float32))


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
    ) -> None:
        if len(options) < 2:
            raise ValueError("Need at least two options")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self._encoder = encoder
        self._labels = tuple(options)
        self._temperature = temperature
        self.fit_seconds = 0.0

        start = time.perf_counter()
        # Prototypes are built once; classifying a clause is then one encode.
        self._prototypes = encoder.encode([options[label] for label in self._labels])
        self.fit_seconds = time.perf_counter() - start

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    def classify_batch(self, states: Sequence[str]) -> list[ChoiceAnswer]:
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

    def classify(self, state: str) -> ChoiceAnswer:
        return self.classify_batch([state])[0]
