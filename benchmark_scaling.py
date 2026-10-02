"""Scaling test: does the encoder advantage hold as lookup volume grows?

Cardinality is held fixed at 8 invoices. Only the number of clauses varies, so
this isolates throughput scaling from discrimination difficulty.

Four paths are measured:

  local-batched   BERT encoder, chunked batches  -- the map-reduce shape
  local-single    BERT encoder, one call per item -- isolates the batching gain
  foundry-batched embedding API, chunked batches
  llm             one request per item; no batch interface exists

The LLM is capped (`--llm-max`) because it is linear, slow and billable. Beyond
the cap its cost is projected from measured per-item latency and labelled as
such -- never silently extrapolated into the headline table.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AzureOpenAI

from corpus import Clause, generate, max_unique
from jev import Choice, EncoderChoiceClassifier, FoundryEncoder, JevClient, LocalBertEncoder

ROOT = Path(__file__).parent
DATA = ROOT / "data"

LLM_MODEL = "gpt-6-luna"
EMBED_MODEL = "text-embedding-3-small"
AI_SCOPE = "https://ai.azure.com/.default"

ABSTAIN_BELOW = 0.35

# USD per 1M tokens, from the published model pages.
PRICING = {
    "llm_input": 0.10,
    "llm_output": 0.50,
    "embed_input": 0.02,
}


def load_catalogue() -> dict[str, str]:
    return json.loads((DATA / "invoices.json").read_text(encoding="utf-8"))


def score(predictions: list[str], clauses: list[Clause], cosines: list[float]) -> dict:
    """Separate model capability from threshold tuning.

    `argmax_accuracy` is what the architecture can discriminate. `routed_accuracy`
    also counts abstentions as misses, so it folds in how well ABSTAIN_BELOW is
    tuned. Reporting only the second one makes a mis-set threshold look like a
    weak model.
    """
    argmax_hits = sum(1 for p, c in zip(predictions, clauses) if p == c.truth)
    abstained = sum(1 for cos in cosines if cos < ABSTAIN_BELOW)
    routed_hits = sum(
        1
        for p, c, cos in zip(predictions, clauses, cosines)
        if cos >= ABSTAIN_BELOW and p == c.truth
    )
    return {
        "argmax_accuracy": argmax_hits / len(clauses),
        "routed_accuracy": routed_hits / len(clauses),
        "abstain_rate": abstained / len(clauses),
    }


def measure_encoder(
    classifier: EncoderChoiceClassifier,
    clauses: list[Clause],
    batched: bool,
) -> dict:
    texts = [clause.text for clause in clauses]

    start = time.perf_counter()
    if batched:
        answers = classifier.classify_batch(texts)
    else:
        answers = [classifier.classify(text) for text in texts]
    elapsed = time.perf_counter() - start

    predictions = [answer.choice for answer in answers]
    cosines = [answer.confidence for answer in answers]

    return {
        "seconds": elapsed,
        "ms_per_item": elapsed * 1000 / len(clauses),
        "items_per_s": len(clauses) / elapsed,
        **score(predictions, clauses, cosines),
    }


def measure_llm(clauses: list[Clause], catalogue: dict[str, str]) -> dict:
    described = "\n".join(f"{code}: {label}" for code, label in catalogue.items())
    questions = {
        "invoice": Choice(
            instructions="Which invoice does this contract clause bill against?\n" + described,
            options=tuple(catalogue),
        )
    }

    predictions: list[str] = []
    input_tokens = 0
    output_tokens = 0

    with AIProjectClient(
        endpoint=os.getenv("PROJECT_ENDPOINT"),
        credential=DefaultAzureCredential(),
    ) as project:
        with project.get_openai_client() as client:
            # Cache off: every clause is unique, so a cache would only mask cost.
            jev = JevClient(client, model=LLM_MODEL, cache=False)

            start = time.perf_counter()
            for clause in clauses:
                response = jev.system_one(clause.text, questions)
                answer = response.choice("invoice")
                predictions.append(answer.choice)
                input_tokens += response.usage.input_tokens
                output_tokens += response.usage.output_tokens
            elapsed = time.perf_counter() - start

    hits = sum(1 for p, c in zip(predictions, clauses) if p == c.truth)
    cost = (
        input_tokens / 1e6 * PRICING["llm_input"]
        + output_tokens / 1e6 * PRICING["llm_output"]
    )
    return {
        "seconds": elapsed,
        "ms_per_item": elapsed * 1000 / len(clauses),
        "items_per_s": len(clauses) / elapsed,
        "argmax_accuracy": hits / len(clauses),
        "routed_accuracy": hits / len(clauses),
        "abstain_rate": 0.0,
        "cost_per_1k": cost / len(clauses) * 1000,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[10, 50, 100, 250, 500, 1000]
    )
    parser.add_argument("--llm-max", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--skip-foundry", action="store_true")
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    resource_endpoint = os.getenv("EMBEDDING_ENDPOINT", "")
    if not resource_endpoint and not args.skip_foundry:
        raise SystemExit("Set EMBEDDING_ENDPOINT or pass --skip-foundry")

    largest = max(args.sizes)
    if largest > max_unique():
        raise SystemExit(f"Corpus can produce at most {max_unique()} unique clauses")

    catalogue = load_catalogue()
    print(f"Cardinality fixed at {len(catalogue)} invoices; scaling clause volume")
    print(f"Sizes: {args.sizes}   batch={args.batch_size}   llm cap={args.llm_max}\n")

    local = EncoderChoiceClassifier(
        LocalBertEncoder(), catalogue, batch_size=args.batch_size
    )
    local.classify("warmup")
    print(f"Local prototype fit: {local.fit_seconds * 1000:.1f}ms (one-off)")

    foundry = None
    foundry_encoder = None
    if not args.skip_foundry:
        token_provider = get_bearer_token_provider(DefaultAzureCredential(), AI_SCOPE)
        foundry_encoder = FoundryEncoder(
            AzureOpenAI(
                azure_endpoint=resource_endpoint,
                api_version="2024-10-21",
                azure_ad_token_provider=token_provider,
            ),
            model=EMBED_MODEL,
        )
        foundry = EncoderChoiceClassifier(
            foundry_encoder, catalogue, batch_size=args.batch_size
        )
        foundry.classify("warmup")
        print(f"Foundry prototype fit: {foundry.fit_seconds * 1000:.1f}ms (one-off)")

    rows: list[tuple[int, str, dict]] = []

    def report(label: str, result: dict) -> None:
        print(
            f"  {label:<16}{result['ms_per_item']:9.3f} ms/item  "
            f"{result['items_per_s']:9.1f} items/s  "
            f"argmax={result['argmax_accuracy']:.3f}  "
            f"routed={result['routed_accuracy']:.3f}  "
            f"abstain={result['abstain_rate']:.2f}"
        )

    for size in args.sizes:
        clauses = generate(size)
        print(f"\n--- N = {size} ---")

        result = measure_encoder(local, clauses, batched=True)
        rows.append((size, "local-batched", result))
        report("local-batched", result)

        result = measure_encoder(local, clauses, batched=False)
        rows.append((size, "local-single", result))
        report("local-single", result)

        if foundry is not None:
            result = measure_encoder(foundry, clauses, batched=True)
            rows.append((size, "foundry-batched", result))
            report("foundry-batched", result)

        if not args.skip_llm and size <= args.llm_max:
            result = measure_llm(clauses, catalogue)
            rows.append((size, "llm", result))
            report("llm", result)
        elif not args.skip_llm:
            print(f"  llm             skipped (N > --llm-max={args.llm_max})")

    print("\n" + "=" * 72)
    print("ms per item as volume grows (lower is better)")
    print("=" * 72)
    architectures = ["local-batched", "local-single", "foundry-batched", "llm"]
    header = "N".rjust(6) + "".join(a.rjust(17) for a in architectures)
    print(header)
    print("-" * len(header))
    for size in args.sizes:
        line = str(size).rjust(6)
        for architecture in architectures:
            match = next(
                (r for n, a, r in rows if n == size and a == architecture), None
            )
            line += (f"{match['ms_per_item']:.3f}" if match else "-").rjust(17)
        print(line)

    llm_rows = [(n, r) for n, a, r in rows if a == "llm"]
    local_rows = [(n, r) for n, a, r in rows if a == "local-batched"]

    print("\n" + "=" * 72)
    print("accuracy at fixed cardinality (argmax = model capability)")
    print("=" * 72)
    for architecture in architectures:
        scores = [r["argmax_accuracy"] for n, a, r in rows if a == architecture]
        if scores:
            print(
                f"  {architecture:<16} argmax {min(scores):.3f}-{max(scores):.3f} "
                f"over N={[n for n, a, _ in rows if a == architecture]}"
            )

    if llm_rows and local_rows:
        llm_ms = llm_rows[-1][1]["ms_per_item"]
        best_local = min(r["ms_per_item"] for _, r in local_rows)
        print(f"\nLLM measured at N={llm_rows[-1][0]}: {llm_ms:.1f} ms/item")
        print(f"Local batched best: {best_local:.3f} ms/item  ({llm_ms / best_local:.0f}x)")
        print(
            f"Projected LLM wall time for {largest} clauses: "
            f"{llm_ms * largest / 1000 / 60:.1f} min (extrapolated, not measured)"
        )
        print(f"Measured LLM cost: ${llm_rows[-1][1]['cost_per_1k']:.4f} per 1k clauses")

    if foundry_encoder is not None:
        embed_cost = foundry_encoder.input_tokens / 1e6 * PRICING["embed_input"]
        print(
            f"Foundry embedding tokens: {foundry_encoder.input_tokens} "
            f"(${embed_cost:.4f} for this entire run)"
        )
    print("Local BERT cost: $0.00 (runs on CPU)")


if __name__ == "__main__":
    main()
