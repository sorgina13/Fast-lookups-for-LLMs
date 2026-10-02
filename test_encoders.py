"""Offline checks for the encoder classifier head. No network, no model download."""

from __future__ import annotations

import numpy as np
import pytest

from jev import (
    CrossEncoderChoiceClassifier,
    EncoderChoiceClassifier,
    LateInteractionChoiceClassifier,
)

OPTIONS = {
    "INV-1001": "cloud hosting and compute",
    "INV-1002": "legal counsel and contract review",
    "INV-1003": "office cleaning",
}


class StubEncoder:
    """Maps known text to fixed unit vectors so the head can be tested alone."""

    def __init__(self, vectors: dict[str, list[float]], dim: int = 3) -> None:
        self._vectors = vectors
        self._dim = dim

    def encode(self, texts):
        rows = [self._vectors.get(text, [1.0] + [0.0] * (self._dim - 1)) for text in texts]
        matrix = np.array(rows, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.clip(norms, 1e-12, None)


def build(extra: dict[str, list[float]] | None = None) -> EncoderChoiceClassifier:
    vectors = {
        "cloud hosting and compute": [1.0, 0.0, 0.0],
        "legal counsel and contract review": [0.0, 1.0, 0.0],
        "office cleaning": [0.0, 0.0, 1.0],
    }
    vectors.update(extra or {})
    return EncoderChoiceClassifier(StubEncoder(vectors), OPTIONS, temperature=0.05)


def test_picks_the_nearest_prototype():
    classifier = build({"hosting clause": [0.9, 0.1, 0.0]})
    answer = classifier.classify("hosting clause")

    assert answer.choice == "INV-1001"
    assert answer.probability > 0.9


def test_probabilities_form_a_distribution():
    classifier = build({"hosting clause": [0.9, 0.1, 0.0]})
    answer = classifier.classify("hosting clause")

    assert set(answer.probabilities) == set(OPTIONS)
    assert sum(answer.probabilities.values()) == pytest.approx(1.0)


def test_confidence_is_raw_cosine_not_softmax():
    # Equidistant from every prototype: softmax still has to pick a winner, but
    # the cosine stays low, which is how "matches nothing" is detected.
    classifier = build({"unrelated": [1.0, 1.0, 1.0]})
    answer = classifier.classify("unrelated")

    assert answer.confidence == pytest.approx(1 / np.sqrt(3), abs=1e-3)
    assert answer.confidence < 0.6
    assert answer.probability == pytest.approx(1 / 3, abs=1e-3)


def test_choice_is_always_a_declared_option():
    classifier = build({"anything": [0.3, 0.3, 0.4]})
    assert classifier.classify("anything").choice in OPTIONS


def test_batch_matches_single():
    classifier = build({"a": [1.0, 0.0, 0.0], "b": [0.0, 0.0, 1.0]})
    batch = classifier.classify_batch(["a", "b"])

    assert [answer.choice for answer in batch] == ["INV-1001", "INV-1003"]
    assert batch[0].choice == classifier.classify("a").choice


def test_lower_temperature_sharpens_the_distribution():
    encoder = StubEncoder(
        {
            "cloud hosting and compute": [1.0, 0.0, 0.0],
            "legal counsel and contract review": [0.0, 1.0, 0.0],
            "office cleaning": [0.0, 0.0, 1.0],
            "hosting clause": [0.8, 0.6, 0.0],
        }
    )

    sharp = EncoderChoiceClassifier(encoder, OPTIONS, temperature=0.01)
    soft = EncoderChoiceClassifier(encoder, OPTIONS, temperature=0.5)

    assert sharp.classify("hosting clause").probability > soft.classify("hosting clause").probability


def test_rejects_invalid_configuration():
    encoder = StubEncoder({})
    with pytest.raises(ValueError):
        EncoderChoiceClassifier(encoder, {"only": "one"})
    with pytest.raises(ValueError):
        EncoderChoiceClassifier(encoder, OPTIONS, temperature=0)
    with pytest.raises(ValueError):
        EncoderChoiceClassifier(encoder, OPTIONS, batch_size=0)


class CountingStubEncoder(StubEncoder):
    """Records how many encode calls and how large each one was."""

    def __init__(self, vectors, dim=3):
        super().__init__(vectors, dim)
        self.call_sizes: list[int] = []

    def encode(self, texts):
        texts = list(texts)
        self.call_sizes.append(len(texts))
        return super().encode(texts)


def test_batches_are_chunked_to_batch_size():
    encoder = CountingStubEncoder(
        {
            "cloud hosting and compute": [1.0, 0.0, 0.0],
            "legal counsel and contract review": [0.0, 1.0, 0.0],
            "office cleaning": [0.0, 0.0, 1.0],
        }
    )
    classifier = EncoderChoiceClassifier(encoder, OPTIONS, batch_size=4)
    encoder.call_sizes.clear()

    answers = classifier.classify_batch([f"clause {i}" for i in range(10)])

    assert len(answers) == 10
    # 10 items at batch_size=4 must arrive as 4 + 4 + 2, never one big request.
    assert encoder.call_sizes == [4, 4, 2]


def test_batched_and_single_agree():
    vectors = {
        "cloud hosting and compute": [1.0, 0.0, 0.0],
        "legal counsel and contract review": [0.0, 1.0, 0.0],
        "office cleaning": [0.0, 0.0, 1.0],
        "a": [0.9, 0.1, 0.0],
        "b": [0.0, 0.2, 0.8],
    }
    classifier = EncoderChoiceClassifier(StubEncoder(vectors), OPTIONS, batch_size=1)

    batched = classifier.classify_batch(["a", "b"])
    singles = [classifier.classify("a"), classifier.classify("b")]

    assert [x.choice for x in batched] == [y.choice for y in singles]
    assert batched[0].probability == pytest.approx(singles[0].probability)


class StubCrossEncoder:
    """Scores a pair by a lookup table, so the head can be tested alone."""

    def __init__(self, scores: dict[tuple[str, str], float]) -> None:
        self._scores = scores
        self.pair_counts: list[int] = []

    def score(self, pairs):
        pairs = list(pairs)
        self.pair_counts.append(len(pairs))
        return np.array([self._scores.get(p, -5.0) for p in pairs], dtype=np.float32)


def test_cross_encoder_picks_the_highest_scoring_option():
    scores = {
        ("hosting clause", "cloud hosting and compute"): 5.0,
        ("hosting clause", "legal counsel and contract review"): -2.0,
        ("hosting clause", "office cleaning"): -3.0,
    }
    classifier = CrossEncoderChoiceClassifier(StubCrossEncoder(scores), OPTIONS)
    answer = classifier.classify("hosting clause")

    assert answer.choice == "INV-1001"
    assert answer.probability > 0.9
    assert sum(answer.probabilities.values()) == pytest.approx(1.0)


def test_cross_encoder_scores_every_option():
    scores = {("x", description): 1.0 for description in OPTIONS.values()}
    encoder = StubCrossEncoder(scores)
    classifier = CrossEncoderChoiceClassifier(encoder, OPTIONS, batch_size=64)

    classifier.classify_batch(["x", "y"])

    # Two states by three options must be scored as six pairs.
    assert sum(encoder.pair_counts) == 6


def test_cross_encoder_chunks_pairs():
    encoder = StubCrossEncoder({})
    classifier = CrossEncoderChoiceClassifier(encoder, OPTIONS, batch_size=4)

    classifier.classify_batch(["a", "b", "c"])

    # 3 states x 3 options = 9 pairs, chunked at 4.
    assert encoder.pair_counts == [4, 4, 1]


def test_cross_encoder_confidence_is_absolute():
    high = {("match", "cloud hosting and compute"): 8.0}
    low = {("match", "cloud hosting and compute"): -8.0}

    confident = CrossEncoderChoiceClassifier(StubCrossEncoder(high), OPTIONS)
    unconfident = CrossEncoderChoiceClassifier(StubCrossEncoder(low), OPTIONS)

    # Both win their softmax, but only one is actually relevant.
    assert confident.classify("match").confidence > 0.99
    assert unconfident.classify("match").confidence < 0.5


def test_cross_encoder_rejects_invalid_configuration():
    encoder = StubCrossEncoder({})
    with pytest.raises(ValueError):
        CrossEncoderChoiceClassifier(encoder, {"only": "one"})
    with pytest.raises(ValueError):
        CrossEncoderChoiceClassifier(encoder, OPTIONS, temperature=0)
    with pytest.raises(ValueError):
        CrossEncoderChoiceClassifier(encoder, OPTIONS, batch_size=0)


class StubTokenEncoder(StubEncoder):
    """Returns per-token vectors so late interaction can be tested offline."""

    def __init__(self, vectors, token_vectors, dim=3):
        super().__init__(vectors, dim)
        self._token_vectors = token_vectors
        self.encode_token_calls = 0

    def encode_tokens(self, texts):
        self.encode_token_calls += 1
        rows = [self._token_vectors[text] for text in texts]
        width = max(len(r) for r in rows)
        hidden = np.zeros((len(rows), width, self._dim), dtype=np.float32)
        mask = np.zeros((len(rows), width), dtype=np.float32)
        for i, tokens in enumerate(rows):
            hidden[i, : len(tokens)] = np.array(tokens, dtype=np.float32)
            mask[i, : len(tokens)] = 1.0
        return hidden, mask


def test_late_interaction_ignores_irrelevant_tokens():
    # One signal token for hosting, three filler tokens pointing elsewhere.
    # Mean pooling would be dominated by the filler; attention should not be.
    token_vectors = {
        "clause": [[1.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.0, 0.5, 0.5], [0.0, 0.5, 0.5]],
    }
    encoder = StubTokenEncoder(
        {
            "cloud hosting and compute": [1.0, 0.0, 0.0],
            "legal counsel and contract review": [0.0, 1.0, 0.0],
            "office cleaning": [0.0, 0.0, 1.0],
        },
        token_vectors,
    )
    classifier = LateInteractionChoiceClassifier(encoder, OPTIONS)

    assert classifier.classify("clause").choice == "INV-1001"


def test_late_interaction_encodes_state_once_per_batch():
    token_vectors = {
        text: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] for text in ("a", "b", "c")
    }
    encoder = StubTokenEncoder(
        {
            "cloud hosting and compute": [1.0, 0.0, 0.0],
            "legal counsel and contract review": [0.0, 1.0, 0.0],
            "office cleaning": [0.0, 0.0, 1.0],
        },
        token_vectors,
    )
    classifier = LateInteractionChoiceClassifier(encoder, OPTIONS, batch_size=64)
    encoder.encode_token_calls = 0

    classifier.classify_batch(["a", "b", "c"])

    # Cost must not scale with the number of options: one pass for the batch.
    assert encoder.encode_token_calls == 1


def test_late_interaction_rejects_invalid_configuration():
    encoder = StubTokenEncoder({}, {})
    with pytest.raises(ValueError):
        LateInteractionChoiceClassifier(encoder, {"only": "one"})
    with pytest.raises(ValueError):
        LateInteractionChoiceClassifier(encoder, OPTIONS, temperature=0)
    with pytest.raises(ValueError):
        LateInteractionChoiceClassifier(encoder, OPTIONS, attention_temperature=0)
