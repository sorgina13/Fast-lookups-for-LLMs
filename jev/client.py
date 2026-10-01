"""A Jev-shaped System One client backed by an LLM deployment.

Jev takes `state` plus typed `questions` and returns typed answers with
probabilities. This reproduces that interface on top of an LLM by compiling the
questions into a strict JSON Schema, so the model is structurally unable to emit
an option that was not declared up front.

Note: the probabilities an LLM returns here are numbers it *wrote*, not a
softmax over model outputs. See `jev.encoders` for the real thing.

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
    Question,
    Score,
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


def _question_schema(question: Question) -> dict[str, Any]:
    if isinstance(question, Choice):
        return {
            "type": "object",
            "description": question.instructions,
            "properties": {
                "kind": {"type": "string", "enum": ["choice"]},
                "choice": {"type": "string", "enum": list(question.options)},
                "probabilities": _distribution_schema(question.options),
                "confidence": {"type": "number"},
            },
            "required": ["kind", "choice", "probabilities", "confidence"],
            "additionalProperties": False,
        }

    if isinstance(question, Score):
        return {
            "type": "object",
            "description": question.instructions,
            "properties": {
                "kind": {"type": "string", "enum": ["score"]},
                "score": {"type": "string", "enum": list(question.rubric)},
                "probabilities": _distribution_schema(question.rubric),
                "confidence": {"type": "number"},
            },
            "required": ["kind", "score", "probabilities", "confidence"],
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


def build_schema(questions: Mapping[str, Question]) -> dict[str, Any]:
    """Compile typed questions into one strict JSON Schema."""
    return {
        "type": "object",
        "properties": {name: _question_schema(q) for name, q in questions.items()},
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

    def __init__(self, client: Any, model: str = "gpt-6-luna", cache: bool = True) -> None:
        self._client = client
        self._model = model
        self._cache: dict[str, SystemOneResponse] | None = {} if cache else None
        self.calls = 0
        self.cache_hits = 0

    def system_one(
        self,
        state: str,
        questions: Mapping[str, Question],
    ) -> SystemOneResponse:
        if not questions:
            raise ValueError("At least one question is required")

        schema = build_schema(questions)
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
