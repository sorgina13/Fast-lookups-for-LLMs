"""A Jev-shaped System One client backed by an LLM deployment.

Jev takes `state` plus typed `questions` and returns typed answers with
probabilities. This reproduces that interface on top of an LLM by compiling the
questions into a strict JSON Schema, so the model is structurally unable to emit
an option that was not declared up front.

Note: the probabilities an LLM returns here are numbers it *wrote*, not a softmax
over model outputs. Treat them as a heuristic; see `jev.encoders` for scores that
come from the model's actual output distribution.

The deployed gpt-6 snapshots reject json_schema on the Responses API, so this
uses Chat Completions.
"""

from __future__ import annotations

import json
import time
from typing import Any, Mapping

from pydantic import TypeAdapter

from .primitives import (
    Answer,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    Usage,
)

SYSTEM_PROMPT = (
    "You are a System One decision model. Evaluate every question "
    "independently against the same state. For each question return a "
    "probability for every available option; probabilities for one question "
    "must sum to 1. Report calibrated confidence between 0 and 1, where high "
    "confidence means you would rarely be wrong. Answer only with the "
    "structured payload."
)

_answer_adapter: TypeAdapter[Answer] = TypeAdapter(Answer)


def _distribution_schema(labels: tuple[str, ...]) -> dict[str, Any]:
    # One required key per label, so a full distribution is structurally forced.
    return {
        "type": "object",
        "properties": {label: {"type": "number"} for label in labels},
        "required": list(labels),
        "additionalProperties": False,
    }


def _question_schema(question: Question, include_probabilities: bool = True) -> dict[str, Any]:
    if isinstance(question, Choice):
        properties: dict[str, Any] = {
            "kind": {"type": "string", "enum": ["choice"]},
            "choice": {"type": "string", "enum": list(question.options)},
        }
        if include_probabilities:
            properties["probabilities"] = _distribution_schema(question.options)
        properties["confidence"] = {"type": "number"}
        return {
            "type": "object",
            "description": question.instructions,
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }

    if isinstance(question, Score):
        properties = {
            "kind": {"type": "string", "enum": ["score"]},
            "score": {"type": "string", "enum": list(question.rubric)},
        }
        if include_probabilities:
            properties["probabilities"] = _distribution_schema(question.rubric)
        properties["confidence"] = {"type": "number"}
        return {
            "type": "object",
            "description": question.instructions,
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }

    return {
        "type": "object",
        "description": question.instructions,
        "properties": {
            "kind": {"type": "string", "enum": ["noul"]},
            "noul": {"type": "number"},
        },
        "required": ["kind", "noul"],
        "additionalProperties": False,
    }


def build_schema(
    questions: Mapping[str, Question], include_probabilities: bool = True
) -> dict[str, Any]:
    """Compile typed questions into one strict JSON Schema.

    With `include_probabilities=False` the per-option distribution is dropped.
    A full distribution requires one required key per option, which at high
    cardinality makes the schema large and the model's output long; Jev itself
    switches strategy past 255 options for the same reason.
    """
    return {
        "type": "object",
        "properties": {
            name: _question_schema(q, include_probabilities)
            for name, q in questions.items()
        },
        "required": list(questions),
        "additionalProperties": False,
    }


def _describe(questions: Mapping[str, Question]) -> str:
    lines: list[str] = []
    for name, question in questions.items():
        if isinstance(question, Choice):
            options = ", ".join(question.options)
            lines.append(f"- {name} (choice): {question.instructions}\n  options: {options}")
        elif isinstance(question, Score):
            rubric = ", ".join(question.rubric)
            lines.append(f"- {name} (score): {question.instructions}\n  rubric: {rubric}")
        else:
            lines.append(f"- {name} (noul, 0-1 truth): {question.instructions}")
    return "\n".join(lines)


class JevClient:
    """System One evaluation backed by an LLM deployment."""

    def __init__(
        self,
        client: Any,
        model: str = "gpt-6-luna",
        cache: bool = True,
        include_probabilities: bool = True,
    ) -> None:
        self._client = client
        self._model = model
        self._cache: dict[str, SystemOneResponse] | None = {} if cache else None
        self._include_probabilities = include_probabilities
        self.calls = 0
        self.cache_hits = 0

    def _fill_distribution(
        self, payload: dict[str, Any], questions: Mapping[str, Question]
    ) -> dict[str, Any]:
        """Rebuild a distribution the model was not asked to emit.

        Keeps ChoiceAnswer/ScoreAnswer total, spreading the remaining mass
        evenly. The winner's probability is the model's own confidence, so
        downstream routing still works -- it is just coarser.
        """
        for name, question in questions.items():
            answer = payload.get(name)
            if not isinstance(answer, dict) or "probabilities" in answer:
                continue

            if isinstance(question, Choice):
                labels, picked = question.options, answer.get("choice")
            elif isinstance(question, Score):
                labels, picked = question.rubric, answer.get("score")
            else:
                continue

            confidence = float(answer.get("confidence", 1.0))
            confidence = min(1.0, max(0.0, confidence))
            remainder = (1.0 - confidence) / max(1, len(labels) - 1)
            answer["probabilities"] = {
                label: (confidence if label == picked else remainder) for label in labels
            }
        return payload

    def system_one(
        self,
        state: str,
        questions: Mapping[str, Question],
    ) -> SystemOneResponse:
        if not questions:
            raise ValueError("At least one question is required")

        schema = build_schema(questions, self._include_probabilities)
        cache_key = json.dumps(
            {"state": state, "model": self._model, "schema": schema}, sort_keys=True
        )

        if self._cache is not None and cache_key in self._cache:
            self.cache_hits += 1
            cached = self._cache[cache_key]
            return cached.model_copy(
                update={"usage": cached.usage.model_copy(update={"cached": True, "latency": 0.0})}
            )

        user_prompt = f"State:\n{state}\n\nQuestions:\n{_describe(questions)}"

        start = time.perf_counter()
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "system_one_response",
                    "strict": True,
                    "schema": schema,
                },
            },
        )
        latency = time.perf_counter() - start
        self.calls += 1

        payload = json.loads(completion.choices[0].message.content)
        if not self._include_probabilities:
            payload = self._fill_distribution(payload, questions)
        answers = {name: _answer_adapter.validate_python(value) for name, value in payload.items()}

        usage = completion.usage
        response = SystemOneResponse(
            answers=answers,
            usage=Usage(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                latency=latency,
            ),
        )

        if self._cache is not None:
            self._cache[cache_key] = response
        return response
