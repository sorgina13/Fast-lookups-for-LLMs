# Fast lookups for LLMs

**Encoder-only classifier heads vs LLM decoding for closed-set lookups.**

When a decision comes from a *closed* list — route this to one of N queues, map
this clause to one of N invoices — generating tokens is the wrong tool. This
repo implements the same typed decision interface three ways and measures the
difference.

The interface is modelled on [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
TypeSafe AI's "System One" model: unstructured `state` plus typed `questions` in,
typed values with probabilities out — decisions your code consumes directly,
instead of strings it has to parse and trust.

## Headline result

Same 10 contract clauses, same 8 invoices, same `ChoiceAnswer` type out of every path:

| Architecture | Accuracy | Median | Mean | Min |
| --- | --- | ---: | ---: | ---: |
| LLM decoder (`gpt-6-luna`) | 10/10 | 1721.3ms | 1995.7ms | 1443.0ms |
| Foundry encoder (`text-embedding-3-small`) | 10/10 | 218.9ms | 274.0ms | 182.0ms |
| **Local BERT encoder (MiniLM-L6 ONNX)** | **10/10** | **2.1ms** | **2.2ms** | **1.9ms** |

**The local encoder is 830x faster than the LLM at identical accuracy** — on CPU,
with no network call and no per-token cost. The Foundry encoder is 7.9x faster;
most of its 219ms is the network round trip, not compute.

Reproduce with `uv run benchmark_architectures.py`.

## Why the encoder wins

A decision over a closed option list does not need a decoder. The LLM path
generates tokens one at a time, then we parse them back into a type. The encoder
path runs one forward pass and reads a softmax off the similarities — which is
all a classification decision ever was.

This mirrors how Jev itself appears to be built. Their pricing is the tell:
**output tokens are free** ("too cheap to meter") while input is metered. Free
output means there is no decode loop. Combined with the three primitives mapping
exactly onto standard head types, the 255-option cap (a fixed head width), and
"adding questions barely changes response time" (one encode, many heads), the
evidence points to an encoder with classification heads rather than an
autoregressive LM.

| Primitive | Head | Returns |
| --- | --- | --- |
| `Choice` | N logits → softmax | `choice`, `probabilities`, `confidence` |
| `Score` | ordered logits → softmax | `score`, `probabilities`, `confidence` |
| `Noul` | 1 logit → sigmoid | `noul` (0–1) |

## Probability vs confidence

The encoder head returns two different numbers, and the distinction is what
makes "none of these" expressible:

- **`probability`** — softmax over cosine similarities. *Relative*; always sums
  to 1. Even a clause matching nothing produces a winner.
- **`confidence`** — raw cosine to the winning prototype. *Absolute*. This is
  the abstention signal.

CL-010 ("the parties shall meet quarterly…") matches no invoice. It still drew
`p=0.676` from the softmax — but `cos=0.336`, below the 0.35 abstention floor,
so it routed to `no_match`. Using the softmax alone would have mis-billed it.

> **On calibration:** the encoder probabilities are a real softmax over model
> outputs. The LLM path's are *not* — there, the model writes a number as text
> that looks like a probability. Treat LLM `probability` as a heuristic, and
> temperature-tune the encoder head against labelled data before trusting its
> absolute values.

## The type-safety guarantee

For the LLM path, the option list is compiled into a strict JSON Schema *before*
the call — the options become an `enum`, with one required probability key per
option:

```python
Choice(instructions="Which invoice does this clause bill against?",
       options=("INV-1001", "INV-1002", ...))
```

So an invoice ID that does not exist **cannot be returned** — it is not
representable in the response schema, not merely discouraged by the prompt. The
encoder path gets this for free: an `argmax` over prototypes can only ever
return an index that exists.

Either way the worst case is a *wrong* ID with a *low* score, which routing
catches — the difference between an unmapped row and a silently mis-billed one.

## Usage

The three backends are interchangeable because they all return `ChoiceAnswer`.

```python
from jev import Choice, EncoderChoiceClassifier, JevClient, LocalBertEncoder

# Encoder head: one forward pass, real softmax.
classifier = EncoderChoiceClassifier(LocalBertEncoder(), catalogue)
answer = classifier.classify("Supplier shall provide dedicated compute capacity.")

# LLM: slower, but handles questions a similarity score cannot express.
jev = JevClient(openai_client, model="gpt-6-luna")
answer = jev.system_one(clause, {"invoice": Choice(...)}).choice("invoice")

answer.choice       # 'INV-1001'  — always a declared option
answer.probability  # relative score within the catalogue
answer.confidence   # absolute match strength; drives abstention
answer.margin       # gap to the runner-up
```

## Running

```bash
cp .env.example .env     # set PROJECT_ENDPOINT and EMBEDDING_ENDPOINT
uv run etl_invoice_lookup.py        # LLM pipeline, writes out/clause_invoice_map.csv
uv run benchmark_architectures.py   # three-way comparison
```

The ETL reads [data/contract_clauses.csv](data/contract_clauses.csv) against
[data/invoices.json](data/invoices.json) and routes on confidence:

| Condition | Decision |
| --- | --- |
| `billable < 0.5` | `no_match` — general terms, not billable work |
| `p >= 0.85` | `auto` — write the mapping |
| `p >= 0.50` | `review` — queue for a human |
| otherwise | `no_match` |

CL-009 duplicates CL-001 and is served from `JevClient`'s cache at zero tokens.
Real clause corpora repeat heavily, so that is the fast path.

## Tests

```bash
uv run --with pytest --with numpy -m pytest -q   # 15 passed
```

Fully offline — no API calls, no model download. Covers schema closure,
out-of-list rejection, probability normalisation, confidence clamping, and the
encoder head's softmax/abstention behaviour via a stub encoder.

## Which to use

- **Local BERT** — closed, stable catalogue. ~2ms, free, no network. Best fit
  for this ETL.
- **Foundry embeddings** — no local model to ship; managed scaling. ~220ms,
  network-bound.
- **LLM** — when the decision needs reasoning a similarity score cannot capture
  (multi-hop conditions, negation, arithmetic over clause terms).

A practical hybrid: encoder for everything, escalating to the LLM only when
`confidence` lands in the ambiguous band. On this corpus that would route 9/10
clauses through the 2ms path.

## Caveats

- **10 clauses is a small benchmark**, and the invoice categories are well
  separated. Near-duplicate invoice descriptions would test the encoders much
  harder than this corpus does.
- **The heads are zero-shot** off pretrained encoders. `temperature=0.05` and
  the 0.35 abstention floor were chosen by inspection, not fitted. With labelled
  data, tune both and measure calibration properly.
- **Latency is environment-specific.** The local figure excludes a one-off model
  download and session warm-up; the Foundry figure is dominated by network
  distance to the region.

## Notes

- Embeddings are served from the **resource** endpoint
  (`https://<resource>.services.ai.azure.com`). The `/api/projects/...` endpoint
  returns 404 for `embeddings.create`.
- The LLM path uses **Chat Completions**. Deployed `gpt-6` snapshots reject
  `json_schema` on the Responses API (`400`) and currently 500 there.
- `Choice` caps at 255 options, matching Jev's documented cardinality limit.
- This approximates Jev's *interface and architecture*, not its training. Real
  Jev is calibrated via RLCD; these heads are zero-shot off pretrained encoders.
