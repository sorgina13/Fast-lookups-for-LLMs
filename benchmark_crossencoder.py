"""Does query conditioning fix dilution, and what does it cost?

A bi-encoder embeds the clause without knowing which option is being considered,
so class-neutral filler dilutes the vector. A cross-encoder reads the clause and
the option together, so attention can weight the tokens that matter for *that*
option. The trade is one forward pass per option instead of one per clause.

Cardinality (8) and volume are fixed. Only the amount of class-neutral filler
varies, across three text conditions:

  core    filler stripped, predicate only
  full    vendor + cadence + region (the scaling corpus)
  padded  filler doubled

Run:  uv run benchmark_crossencoder.py --n 200
      uv run benchmark_crossencoder.py --n 100 --with-llm
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

from corpus import CADENCE, REGIONS, TEMPLATES, VENDORS
from jev import (
    Choice,
    CrossEncoderChoiceClassifier,
    EncoderChoiceClassifier,
    JevClient,
    LateInteractionChoiceClassifier,
    LocalBertEncoder,
    LocalCrossEncoder,
)

ROOT = Path(__file__).parent
DATA = ROOT / "data"
LLM_MODEL = "gpt-6-luna"

VARIANTS = ("core", "full", "padded")


def strip_filler(text: str) -> str:
    text = re.sub(r"^the [A-Z][a-z]+( [A-Z][a-z]+)? ", "", text)
    text = re.sub(
        r",? (billed|invoiced|charged|payable|settled|chargeable|falling due|"
        r"with charges raised)\b.*$",
        "",
        text,
    )
    text = re.sub(
        r" (at|for|across|to|serving) (the [a-z ]+|all UK premises|each listed location)\b",
        "",
        text,
    )
    return text.strip().rstrip(".") + "."


def build(variant: str, count: int, seed: int = 11) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    labels = list(TEMPLATES)
    texts, truths = [], []

    for index in range(count):
        label = labels[index % len(labels)]
        filled = rng.choice(TEMPLATES[label]).format(
            vendor=rng.choice(VENDORS),
            cadence=rng.choice(CADENCE),
            region=rng.choice(REGIONS),
        )
        if variant == "core":
            filled = strip_filler(filled)
        elif variant == "padded":
            filled = (
                f"{filled} This obligation is recorded by {rng.choice(VENDORS)} "
                f"and reviewed {rng.choice(CADENCE)} at {rng.choice(REGIONS)}."
            )
        texts.append(filled)
        truths.append(label)

    return texts, truths


def measure(classifier, texts: list[str], truths: list[str]) -> dict:
    start = time.perf_counter()
    answers = classifier.classify_batch(texts)
    elapsed = time.perf_counter() - start

    hits = sum(1 for a, t in zip(answers, truths) if a.choice == t)
    return {
        "accuracy": hits / len(texts),
        "ms_per_item": elapsed * 1000 / len(texts),
        "mean_confidence": sum(a.confidence for a in answers) / len(answers),
    }


def measure_llm(texts: list[str], truths: list[str], catalogue: dict[str, str]) -> dict:
    described = "\n".join(f"{code}: {label}" for code, label in catalogue.items())
    questions = {
        "invoice": Choice(
            instructions="Which invoice does this contract clause bill against?\n" + described,
            options=tuple(catalogue),
        )
    }

    with AIProjectClient(
        endpoint=os.getenv("PROJECT_ENDPOINT"),
        credential=DefaultAzureCredential(),
    ) as project:
        with project.get_openai_client() as client:
            jev = JevClient(client, model=LLM_MODEL, cache=False)
            start = time.perf_counter()
            predictions = [
                jev.system_one(text, questions).choice("invoice").choice for text in texts
            ]
            elapsed = time.perf_counter() - start

    hits = sum(1 for p, t in zip(predictions, truths) if p == t)
    return {
        "accuracy": hits / len(texts),
        "ms_per_item": elapsed * 1000 / len(texts),
        "mean_confidence": float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--seed", type=int, default=11, help="corpus seed; use a held-out seed for reporting"
    )
    parser.add_argument(
        "--with-llm", action="store_true", help="also measure the LLM (slow, billable)"
    )
    args = parser.parse_args()

    load_dotenv()
    catalogue = json.loads((DATA / "invoices.json").read_text(encoding="utf-8"))

    bi_encoder = LocalBertEncoder()
    bi = EncoderChoiceClassifier(bi_encoder, catalogue, batch_size=args.batch_size)
    late = LateInteractionChoiceClassifier(
        bi_encoder, catalogue, batch_size=args.batch_size
    )
    cross = CrossEncoderChoiceClassifier(
        LocalCrossEncoder(), catalogue, batch_size=args.batch_size
    )

    print(f"cardinality={len(catalogue)}  volume={args.n}  seed={args.seed}  (all fixed)")
    print("only the amount of class-neutral filler changes\n")

    results: dict[str, dict[str, dict]] = {}

    for variant in VARIANTS:
        texts, truths = build(variant, args.n, seed=args.seed)
        words = sum(len(t.split()) for t in texts) / len(texts)
        print(f"--- {variant} ({words:.1f} words/clause) ---")

        results.setdefault("bi-encoder", {})[variant] = measure(bi, texts, truths)
        results.setdefault("late-interaction", {})[variant] = measure(late, texts, truths)
        results.setdefault("cross-encoder", {})[variant] = measure(cross, texts, truths)

        for name in ("bi-encoder", "late-interaction", "cross-encoder"):
            row = results[name][variant]
            print(
                f"  {name:<17} acc={row['accuracy']:.3f}  "
                f"{row['ms_per_item']:7.2f} ms/item  conf={row['mean_confidence']:.3f}"
            )

        if args.with_llm:
            row = measure_llm(texts, truths, catalogue)
            results.setdefault("llm", {})[variant] = row
            print(f"  {'llm':<17} acc={row['accuracy']:.3f}  {row['ms_per_item']:7.2f} ms/item")
        print()

    print("=" * 68)
    print("accuracy under increasing dilution")
    print("=" * 68)
    header = f"{'architecture':<18}" + "".join(v.rjust(10) for v in VARIANTS) + "      drop"
    print(header)
    print("-" * len(header))
    for name, by_variant in results.items():
        scores = [by_variant[v]["accuracy"] for v in VARIANTS]
        line = f"{name:<18}" + "".join(f"{s:.3f}".rjust(10) for s in scores)
        print(line + f"{scores[0] - scores[-1]:+10.3f}")

    print("\nlatency (ms/item)")
    for name, by_variant in results.items():
        costs = [by_variant[v]["ms_per_item"] for v in VARIANTS]
        print(f"  {name:<18}" + "".join(f"{c:9.2f}" for c in costs))

    print("\ndilution cost (core -> padded)")
    for name, by_variant in results.items():
        drop = by_variant[VARIANTS[0]]["accuracy"] - by_variant[VARIANTS[-1]]["accuracy"]
        print(f"  {name:<18}{drop:+.3f}")

    slowdown = (
        results["cross-encoder"]["full"]["ms_per_item"]
        / results["bi-encoder"]["full"]["ms_per_item"]
    )
    late_slowdown = (
        results["late-interaction"]["full"]["ms_per_item"]
        / results["bi-encoder"]["full"]["ms_per_item"]
    )
    print(
        f"\nvs bi-encoder at {len(catalogue)} options: "
        f"late-interaction {late_slowdown:.1f}x, cross-encoder {slowdown:.1f}x"
    )


if __name__ == "__main__":
    main()
