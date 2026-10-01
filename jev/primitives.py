"""Jev-style System One primitives: typed questions in, typed answers out.

A System One model evaluates typed *questions* against a *state* and returns
structured values your code can branch on directly -- no string parsing. The
option list is closed, so an answer outside it cannot be represented.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class Choice(BaseModel):
    """Choose exactly one option from a closed list."""

    kind: Literal["choice"] = "choice"
    instructions: str
    options: tuple[str, ...]

    @field_validator("options")
    @classmethod
    def _validate_options(cls, options: tuple[str, ...]) -> tuple[str, ...]:
        if len(options) < 2:
            raise ValueError("Choice needs at least two options")
        if len(set(options)) != len(options):
            raise ValueError("Choice options must be unique")
        if len(options) > 255:
            raise ValueError("Choice supports at most 255 options")
        return options


class Score(BaseModel):
    """Score the state against an ordered rubric."""

    kind: Literal["score"] = "score"
    instructions: str
    rubric: tuple[str, ...]

    @field_validator("rubric")
    @classmethod
    def _validate_rubric(cls, rubric: tuple[str, ...]) -> tuple[str, ...]:
        if len(rubric) < 2:
            raise ValueError("Score needs at least two rubric levels")
        if len(set(rubric)) != len(rubric):
            raise ValueError("Score rubric levels must be unique")
        return rubric


class Noul(BaseModel):
    """Is this statement true? Returns a probability between 0 and 1."""

    kind: Literal["noul"] = "noul"
    instructions: str


Question = Annotated[Union[Choice, Score, Noul], Field(discriminator="kind")]


def _normalize(probabilities: dict[str, float]) -> dict[str, float]:
    clamped = {key: max(0.0, float(value)) for key, value in probabilities.items()}
    total = sum(clamped.values())
    if total <= 0:
        share = 1.0 / len(clamped)
        return {key: share for key in clamped}
    return {key: value / total for key, value in clamped.items()}


class ChoiceAnswer(BaseModel):
    """A typed choice with a distribution over every option."""

    kind: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float

    @model_validator(mode="after")
    def _validate(self) -> "ChoiceAnswer":
        if self.choice not in self.probabilities:
            raise ValueError(f"choice {self.choice!r} is missing from probabilities")
        self.probabilities = _normalize(self.probabilities)
        self.confidence = min(1.0, max(0.0, self.confidence))
        return self

    @property
    def probability(self) -> float:
        """Probability assigned to the selected option."""
        return self.probabilities[self.choice]

    @property
    def runner_up(self) -> tuple[str, float] | None:
        others = sorted(
            ((key, value) for key, value in self.probabilities.items() if key != self.choice),
            key=lambda item: item[1],
            reverse=True,
        )
        return others[0] if others else None

    @property
    def margin(self) -> float:
        """Gap between the winner and the next best option."""
        runner_up = self.runner_up
        return self.probability - runner_up[1] if runner_up else self.probability


class ScoreAnswer(BaseModel):
    kind: Literal["score"] = "score"
    score: str
    probabilities: dict[str, float]
    confidence: float

    @model_validator(mode="after")
    def _validate(self) -> "ScoreAnswer":
        if self.score not in self.probabilities:
            raise ValueError(f"score {self.score!r} is missing from probabilities")
        self.probabilities = _normalize(self.probabilities)
        self.confidence = min(1.0, max(0.0, self.confidence))
        return self


class NoulAnswer(BaseModel):
    kind: Literal["noul"] = "noul"
    noul: float

    @model_validator(mode="after")
    def _validate(self) -> "NoulAnswer":
        self.noul = min(1.0, max(0.0, self.noul))
        return self


Answer = Union[ChoiceAnswer, ScoreAnswer, NoulAnswer]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    latency: float = 0.0
    cached: bool = False


class SystemOneResponse(BaseModel):
    """Every question answered in one round trip."""

    answers: dict[str, Answer]
    usage: Usage

    def choice(self, name: str) -> ChoiceAnswer:
        answer = self.answers[name]
        if not isinstance(answer, ChoiceAnswer):
            raise TypeError(f"question {name!r} is not a Choice")
        return answer

    def score(self, name: str) -> ScoreAnswer:
        answer = self.answers[name]
        if not isinstance(answer, ScoreAnswer):
            raise TypeError(f"question {name!r} is not a Score")
        return answer

    def noul(self, name: str) -> NoulAnswer:
        answer = self.answers[name]
        if not isinstance(answer, NoulAnswer):
            raise TypeError(f"question {name!r} is not a Noul")
        return answer
