"""Cardinality test: what happens as the candidate pool grows?

Volume and phrasing are held fixed. Only the number of invoices the lookup must
choose between changes -- 8 real categories plus distractors drawn from ordinary
corporate spend. Ground truth never changes; the haystack does.

Three architectures:

  late-interaction   local BERT, query-conditioned pooling, one pass per clause
  foundry-embedding  text-embedding-3-small bi-encoder
  foundry-gpt        gpt-6-luna with a strict enum over every option

The LLM is capped (`--llm-n`) because it is linear and billable, and because its
schema grows with cardinality. Above a few dozen options the full probability
distribution becomes impractical, so the LLM runs with `include_probabilities=
False` at every K -- consistent across the sweep, and noted in the output.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AzureOpenAI

from corpus import Clause, build_catalogue, generate, max_catalogue
from jev import (
    Choice,
    EncoderChoiceClassifier,
    FoundryEncoder,
    JevClient,
    LateInteractionChoiceClassifier,
    LocalBertEncoder,
)

LLM_MODEL = "gpt-6-luna"
EMBED_MODEL = "text-embedding-3-small"
AI_SCOPE = "https://ai.azure.com/.default"

PRICING = {"llm_input": 0.10, "llm_output": 0.50, "embed_input": 0.02}


def measure_classifier(classifier, clauses: list[Clause]) -> dict:
    texts = [clause.text for clause in clauses]
    start = time.perf_counter()
    answers = classifier.classify_batch(texts)
    elapsed = time.perf_counter() - start

    hits = sum(1 for a, c in zip(answers, clauses) if a.choice == c.truth)
    return {
        "accuracy": hits / len(clauses),
        "ms_per_item": elapsed * 1000 / len(clauses),
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
    input_tokens = output_tokens = 0
    failures = 0

    with AIProjectClient(
        endpoint=os.getenv("PROJECT_ENDPOINT"),
        credential=DefaultAzureCredential(),
    ) as project:
        with project.get_openai_client() as client:
            jev = JevClient(
                client, model=LLM_MODEL, cache=False, include_probabilities=False
            )
            start = time.perf_counter()
            for clause in clauses:
                try:
                    response = jev.system_one(clause.text, questions)
                    predictions.append(response.choice("invoice").choice)
                    input_tokens += response.usage.input_tokens
                    output_tokens += response.usage.output_tokens
                except Exception:
                    failures += 1
                    predictions.append("")
            elapsed = time.perf_counter() - start

    hits = sum(1 for p, c in zip(predictions, clauses) if p == c.truth)
    cost = (
        input_tokens / 1e6 * PRICING["llm_input"]
        + output_tokens / 1e6 * PRICING["llm_output"]
    )
    return {
        "accuracy": hits / len(clauses),
        "ms_per_item": elapsed * 1000 / len(clauses),
        "cost_per_1k": cost / len(clauses) * 1000,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cardinalities", type=int, nargs="+", default=[8, 16, 32, 64, 128, 255]
    )
    parser.add_argument("--n", type=int, default=1000, help="clauses for the encoders")
    parser.add_argument("--llm-n", type=int, default=50, help="clauses for the LLM")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()

    load_dotenv()
    resource_endpoint = os.getenv("EMBEDDING_ENDPOINT", "")
    if not resource_endpoint:
        raise SystemExit("Set EMBEDDING_ENDPOINT to your resource endpoint")

    biggest = max(args.cardinalities)
    if biggest > max_catalogue():
        raise SystemExit(f"Catalogue supports at most {max_catalogue()} invoices")

    clauses = generate(args.n, seed=args.seed)
    llm_clauses = clauses[: args.llm_n]

    print(f"volume={args.n} clauses (LLM: {args.llm_n}), phrasing fixed, seed={args.seed}")
    print(f"cardinality sweep: {args.cardinalities}")
    print("LLM runs without a full probability distribution so its schema scales\n")

    bi_encoder = LocalBertEncoder()
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), AI_SCOPE)
    embedding_client = AzureOpenAI(
        azure_endpoint=resource_endpoint,
        api_version="2024-10-21",
        azure_ad_token_provider=token_provider,
    )

    rows: list[tuple[int, str, dict]] = []
    embed_tokens_before = 0

    for size in args.cardinalities:
        catalogue = build_catalogue(size)
        print(f"--- K = {size} options ---")

        late = LateInteractionChoiceClassifier(
            bi_encoder, catalogue, batch_size=args.batch_size
        )
        result = measure_classifier(late, clauses)
        rows.append((size, "late-interaction", result))
        print(f"  late-interaction   acc={result['accuracy']:.3f}  {result['ms_per_item']:8.2f} ms/item")

        foundry_encoder = FoundryEncoder(embedding_client, model=EMBED_MODEL)
        foundry = EncoderChoiceClassifier(
            foundry_encoder, catalogue, batch_size=args.batch_size
        )
        result = measure_classifier(foundry, clauses)
        embed_tokens_before += foundry_encoder.input_tokens
        rows.append((size, "foundry-embedding", result))
        print(f"  foundry-embedding  acc={result['accuracy']:.3f}  {result['ms_per_item']:8.2f} ms/item")

        if not args.skip_llm:
            result = measure_llm(llm_clauses, catalogue)
            rows.append((size, "foundry-gpt", result))
            note = f"  failures={result['failures']}" if result["failures"] else ""
            print(
                f"  foundry-gpt        acc={result['accuracy']:.3f}  "
                f"{result['ms_per_item']:8.2f} ms/item"
                f"  ${result['cost_per_1k']:.4f}/1k{note}"
            )
        print()

    architectures = ["late-interaction", "foundry-embedding", "foundry-gpt"]

    print("=" * 74)
    print("accuracy as the candidate pool grows (volume and phrasing fixed)")
    print("=" * 74)
    header = "K".rjust(5) + "".join(a.rjust(21) for a in architectures)
    print(header)
    print("-" * len(header))
    for size in args.cardinalities:
        line = str(size).rjust(5)
        for architecture in architectures:
            match = next((r for k, a, r in rows if k == size and a == architecture), None)
            line += (f"{match['accuracy']:.3f}" if match else "-").rjust(21)
        print(line)

    print("\nms per item")
    print(header)
    print("-" * len(header))
    for size in args.cardinalities:
        line = str(size).rjust(5)
        for architecture in architectures:
            match = next((r for k, a, r in rows if k == size and a == architecture), None)
            line += (f"{match['ms_per_item']:.2f}" if match else "-").rjust(21)
        print(line)

    print("\ndegradation from smallest to largest catalogue")
    for architecture in architectures:
        scores = [r["accuracy"] for k, a, r in rows if a == architecture]
        if len(scores) >= 2:
            print(f"  {architecture:<20}{scores[0]:.3f} -> {scores[-1]:.3f}  ({scores[-1] - scores[0]:+.3f})")

    llm_costs = [r["cost_per_1k"] for _, a, r in rows if a == "foundry-gpt"]
    if llm_costs:
        print(
            f"\nLLM cost per 1k clauses: ${llm_costs[0]:.4f} at K={args.cardinalities[0]}"
            f" -> ${llm_costs[-1]:.4f} at K={args.cardinalities[-1]}"
        )
    print(f"Foundry embedding tokens for this run: {embed_tokens_before}")


if __name__ == "__main__":
    main()
