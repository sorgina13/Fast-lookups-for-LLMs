"""Sweep attention_temperature for the late-interaction head.

Attention temperature controls how peaked the per-option pooling is:

  low   -> near-argmax; only the single best-matching token contributes
  high  -> near-uniform; collapses back to plain mean pooling

Tuning and reporting on the same data would overfit, so this sweeps on a
tuning seed only. Re-run benchmark_crossencoder.py with a different --seed to
get a clean held-out number.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_crossencoder import VARIANTS, build
from jev import LateInteractionChoiceClassifier, LocalBertEncoder

DATA = Path(__file__).parent / "data"

GRID = [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.35, 0.6, 1.0, 2.0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=11, help="tuning seed")
    args = parser.parse_args()

    catalogue = json.loads((DATA / "invoices.json").read_text(encoding="utf-8"))
    encoder = LocalBertEncoder()

    corpora = {
        variant: build(variant, args.n, seed=args.seed) for variant in VARIANTS
    }

    print(f"tuning on seed={args.seed}  n={args.n}  cardinality={len(catalogue)}\n")
    header = f"{'attn_temp':>10}" + "".join(v.rjust(9) for v in VARIANTS) + f"{'mean':>9}"
    print(header)
    print("-" * len(header))

    scored: list[tuple[float, float, list[float]]] = []

    for attention_temperature in GRID:
        classifier = LateInteractionChoiceClassifier(
            encoder, catalogue, attention_temperature=attention_temperature
        )
        accuracies = []
        for variant in VARIANTS:
            texts, truths = corpora[variant]
            answers = classifier.classify_batch(texts)
            hits = sum(1 for a, t in zip(answers, truths) if a.choice == t)
            accuracies.append(hits / len(texts))

        mean_accuracy = sum(accuracies) / len(accuracies)
        scored.append((attention_temperature, mean_accuracy, accuracies))
        print(
            f"{attention_temperature:10.3f}"
            + "".join(f"{a:9.3f}" for a in accuracies)
            + f"{mean_accuracy:9.3f}"
        )

    best_temp, best_mean, best_scores = max(scored, key=lambda row: row[1])
    print(f"\nbest attention_temperature = {best_temp} (mean {best_mean:.3f})")
    print("per-variant:", {v: round(s, 3) for v, s in zip(VARIANTS, best_scores)})
    print(
        f"\nNow validate on held-out data:\n"
        f"  uv run benchmark_crossencoder.py --n {args.n} --seed {args.seed + 100}"
    )


if __name__ == "__main__":
    main()
