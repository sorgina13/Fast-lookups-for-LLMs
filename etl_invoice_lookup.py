"""ETL: map contract clauses to invoice IDs using a Jev-style lookup.

The classifier replaces a brittle keyword/regex mapping table. Each clause is
state; the invoice catalogue is a closed Choice. Because the option list is
compiled into the schema, the model cannot invent an invoice ID -- the worst
case is the wrong ID with a low probability, which the routing rules catch.
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

from jev import Choice, JevClient, Noul

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "out"

MODEL = "gpt-6-luna"

# Route on probability, not just the argmax: a confident wrong answer is the
# only failure mode worth paging a human for.
AUTO_ACCEPT = 0.85
REVIEW_FLOOR = 0.50


def load_catalogue() -> dict[str, str]:
    return json.loads((DATA / "invoices.json").read_text(encoding="utf-8"))


def load_clauses() -> list[dict[str, str]]:
    with (DATA / "contract_clauses.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_questions(catalogue: dict[str, str]) -> dict[str, Choice | Noul]:
    described = "\n".join(f"{code}: {label}" for code, label in catalogue.items())
    return {
        "invoice": Choice(
            instructions=(
                "Which invoice does this contract clause bill against?\n" + described
            ),
            options=tuple(catalogue),
        ),
        "billable": Noul(
            instructions="This clause describes billable work rather than general terms.",
        ),
    }


def route(probability: float, billable: float) -> str:
    if billable < 0.5:
        return "no_match"
    if probability >= AUTO_ACCEPT:
        return "auto"
    if probability >= REVIEW_FLOOR:
        return "review"
    return "no_match"


def main() -> None:
    load_dotenv()
    catalogue = load_catalogue()
    clauses = load_clauses()
    questions = build_questions(catalogue)

    OUT.mkdir(exist_ok=True)
    rows: list[dict[str, object]] = []

    with AIProjectClient(
        endpoint=os.getenv("PROJECT_ENDPOINT"),
        credential=DefaultAzureCredential(),
    ) as project:
        with project.get_openai_client() as openai_client:
            jev = JevClient(openai_client, model=MODEL)

            print(f"Mapping {len(clauses)} clauses against {len(catalogue)} invoices\n")
            started = time.perf_counter()

            for clause in clauses:
                response = jev.system_one(clause["clause_text"], questions)
                invoice = response.choice("invoice")
                billable = response.noul("billable").noul
                decision = route(invoice.probability, billable)

                rows.append(
                    {
                        "clause_id": clause["clause_id"],
                        "invoice_id": invoice.choice if decision != "no_match" else "",
                        "probability": round(invoice.probability, 4),
                        "margin": round(invoice.margin, 4),
                        "confidence": round(invoice.confidence, 4),
                        "billable": round(billable, 4),
                        "decision": decision,
                        "cached": response.usage.cached,
                        "latency_s": round(response.usage.latency, 3),
                    }
                )

                marker = "cache" if response.usage.cached else f"{response.usage.latency:5.2f}s"
                print(
                    f"{clause['clause_id']} -> {invoice.choice:<9} "
                    f"p={invoice.probability:.2f} margin={invoice.margin:.2f} "
                    f"conf={invoice.confidence:.2f} billable={billable:.2f} "
                    f"{decision:<9} {marker}"
                )

            elapsed = time.perf_counter() - started

    out_file = OUT / "clause_invoice_map.csv"
    with out_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = {}
    for row in rows:
        decision = str(row["decision"])
        counts[decision] = counts.get(decision, 0) + 1

    print("\nRouting: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"Model calls={jev.calls} cache hits={jev.cache_hits}")
    print(f"Total wall time {elapsed:.2f}s")
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()
