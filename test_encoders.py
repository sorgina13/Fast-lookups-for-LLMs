"""Offline checks for the encoder classifier head. No network, no model download."""

from __future__ import annotations

import numpy as np
import pytest

from jev import EncoderChoiceClassifier

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
