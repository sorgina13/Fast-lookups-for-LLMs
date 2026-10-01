"""Offline checks for the type-safety guarantees. No API calls."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from jev import Choice, ChoiceAnswer, Noul, Score, build_schema

OPTIONS = ("INV-1001", "INV-1002", "INV-1003")


def test_schema_closes_the_option_list():
    schema = build_schema({"invoice": Choice(instructions="which?", options=OPTIONS)})
    invoice = schema["properties"]["invoice"]

    assert invoice["properties"]["choice"]["enum"] == list(OPTIONS)
    assert invoice["additionalProperties"] is False
    # Every option is a required probability key, so a partial distribution
    # cannot be represented.
    assert invoice["properties"]["probabilities"]["required"] == list(OPTIONS)
    assert schema["required"] == ["invoice"]


def test_answer_rejects_option_outside_the_list():
    with pytest.raises(ValidationError):
        ChoiceAnswer(
            choice="INV-9999",
            probabilities={option: 1 / 3 for option in OPTIONS},
            confidence=0.9,
        )


def test_probabilities_are_normalized():
    answer = ChoiceAnswer(
        choice="INV-1001",
        probabilities={"INV-1001": 6.0, "INV-1002": 2.0, "INV-1003": 2.0},
        confidence=0.8,
    )

    assert sum(answer.probabilities.values()) == pytest.approx(1.0)
    assert answer.probability == pytest.approx(0.6)
    assert answer.runner_up[1] == pytest.approx(0.2)
    assert answer.margin == pytest.approx(0.4)


def test_confidence_is_clamped():
    answer = ChoiceAnswer(
        choice="INV-1001",
        probabilities={option: 1 / 3 for option in OPTIONS},
        confidence=4.2,
    )
    assert answer.confidence == 1.0


def test_degenerate_distribution_falls_back_to_uniform():
    answer = ChoiceAnswer(
        choice="INV-1001",
        probabilities={option: 0.0 for option in OPTIONS},
        confidence=0.0,
    )
    assert answer.probability == pytest.approx(1 / 3)


def test_choice_requires_unique_options():
    with pytest.raises(ValidationError):
        Choice(instructions="x", options=("A", "A"))


def test_choice_requires_two_options():
    with pytest.raises(ValidationError):
        Choice(instructions="x", options=("A",))


def test_mixed_question_types_compile_together():
    schema = build_schema(
        {
            "invoice": Choice(instructions="which?", options=OPTIONS),
            "risk": Score(instructions="how risky?", rubric=("low", "high")),
            "billable": Noul(instructions="is it billable?"),
        }
    )

    assert set(schema["required"]) == {"invoice", "risk", "billable"}
    assert schema["properties"]["risk"]["properties"]["score"]["enum"] == ["low", "high"]
    assert schema["properties"]["billable"]["properties"]["noul"]["type"] == "number"
    json.dumps(schema)
