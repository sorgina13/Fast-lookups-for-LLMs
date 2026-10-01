"""Three-way comparison: LLM decoder vs Foundry encoder vs local BERT encoder.

Same 10 clauses, same 8 invoices, same typed ChoiceAnswer out of each path.
The only thing that changes is the architecture doing the lookup.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import time
from pathlib import Path

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AzureOpenAI

from jev import Choice, EncoderChoiceClassifier, FoundryEncoder, JevClient, LocalBertEncoder

ROOT = Path(__file__).parent
DATA = ROOT / "data"

LLM_MODEL = "gpt-6-luna"
EMBED_MODEL = "text-embedding-3-small"

# Embeddings are served from the resource endpoint; the project endpoint 404s.
AI_SCOPE = "https://ai.azure.com/.default"

# Ground truth: CL-010 is general terms and matches no invoice.
TRUTH = {
    "CL-001": "INV-1001",
    "CL-002": "INV-1002",
    "CL-003": "INV-1003",
    "CL-004": "INV-1004",
    "CL-005": "INV-1005",
    "CL-006": "INV-1006",
    "CL-007": "INV-1007",
    "CL-008": "INV-1008",
    "CL-009": "INV-1001",
    "CL-010": None,
}

# Cosine below this means "matches nothing in the catalogue".
ABSTAIN_BELOW = 0.35


def load_inputs() -> tuple[dict[str, str], list[dict[str, str]]]:
    catalogue = json.loads((DATA / "invoices.json").read_text(encoding="utf-8"))
    with (DATA / "contract_clauses.csv").open(encoding="utf-8", newline="") as handle:
        clauses = list(csv.DictReader(handle))
    return catalogue, clauses


def summarize(name: str, latencies_ms: list[float], correct: int, total: int) -> dict:
    return {
        "architecture": name,
        "accuracy": f"{correct}/{total}",
        "median_ms": statistics.median(latencies_ms),
        "mean_ms": statistics.mean(latencies_ms),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
    }


def run_llm(catalogue, clauses) -> dict:
    described = "\n".join(f"{code}: {label}" for code, label in catalogue.items())
    questions = {
        "invoice": Choice(
            instructions="Which invoice does this contract clause bill against?\n" + described,
            options=tuple(catalogue),
        )
    }

    latencies: list[float] = []
    correct = 0

    with AIProjectClient(
        endpoint=os.getenv("PROJECT_ENDPOINT"),
        credential=DefaultAzureCredential(),
    ) as project:
        with project.get_openai_client() as client:
            jev = JevClient(client, model=LLM_MODEL, cache=False)

            for clause in clauses:
                start = time.perf_counter()
                answer = jev.system_one(clause["clause_text"], questions).choice("invoice")
                latencies.append((time.perf_counter() - start) * 1000)

                predicted = answer.choice if answer.probability >= 0.5 else None
                if predicted == TRUTH[clause["clause_id"]]:
                    correct += 1

    return summarize(f"LLM decoder ({LLM_MODEL})", latencies, correct, len(clauses))


def run_encoder(name: str, classifier: EncoderChoiceClassifier, clauses) -> dict:
    latencies: list[float] = []
    correct = 0
    rows: list[tuple[str, str, float, float]] = []

    for clause in clauses:
        start = time.perf_counter()
        answer = classifier.classify(clause["clause_text"])
        latencies.append((time.perf_counter() - start) * 1000)

        predicted = answer.choice if answer.confidence >= ABSTAIN_BELOW else None
        if predicted == TRUTH[clause["clause_id"]]:
            correct += 1
        rows.append(
            (clause["clause_id"], predicted or "no_match", answer.probability, answer.confidence)
        )

    print(f"\n  {name}")
    for clause_id, predicted, probability, confidence in rows:
        flag = " " if predicted == (TRUTH[clause_id] or "no_match") else "X"
        print(
            f"  {flag} {clause_id} -> {predicted:<9} "
            f"p={probability:.3f} cos={confidence:.3f}"
        )

    return summarize(name, latencies, correct, len(clauses))


def main() -> None:
    load_dotenv()
    resource_endpoint = os.getenv("EMBEDDING_ENDPOINT", "")
    if not resource_endpoint:
        raise SystemExit(
            "Set EMBEDDING_ENDPOINT to your resource endpoint, e.g.\n"
            "  EMBEDDING_ENDPOINT=https://<resource>.services.ai.azure.com"
        )

    catalogue, clauses = load_inputs()
    results = []

    print(f"{len(clauses)} clauses against {len(catalogue)} invoices")

    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(credential, AI_SCOPE)
    embedding_client = AzureOpenAI(
        azure_endpoint=resource_endpoint,
        api_version="2024-10-21",
        azure_ad_token_provider=token_provider,
    )

    foundry_encoder = FoundryEncoder(embedding_client, model=EMBED_MODEL)
    foundry_classifier = EncoderChoiceClassifier(foundry_encoder, catalogue)
    # Warm the connection so the first clause is not paying TLS setup.
    foundry_classifier.classify("warmup")
    results.append(run_encoder(f"Foundry encoder ({EMBED_MODEL})", foundry_classifier, clauses))

    local_encoder = LocalBertEncoder()
    local_classifier = EncoderChoiceClassifier(local_encoder, catalogue)
    local_classifier.classify("warmup")
    results.append(run_encoder("Local BERT encoder (MiniLM-L6 ONNX)", local_classifier, clauses))

    results.append(run_llm(catalogue, clauses))

    print("\n" + "=" * 78)
    print(f"{'architecture':<38} {'acc':<7} {'median':>9} {'mean':>9} {'min':>9}")
    print("-" * 78)
    for row in results:
        print(
            f"{row['architecture']:<38} {row['accuracy']:<7} "
            f"{row['median_ms']:>8.1f}ms {row['mean_ms']:>8.1f}ms {row['min_ms']:>8.1f}ms"
        )

    baseline = next(r for r in results if r["architecture"].startswith("LLM"))
    print()
    for row in results:
        if row is baseline:
            continue
        speedup = baseline["median_ms"] / row["median_ms"]
        print(f"{row['architecture']} is {speedup:.1f}x faster than the LLM (median)")

    print(f"\nFoundry embedding tokens used: {foundry_encoder.input_tokens}")


if __name__ == "__main__":
    main()
