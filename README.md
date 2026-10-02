# Fast lookups for LLMs

**When a decision comes from a closed list, does generating tokens make sense?**

Routing a ticket to one of N queues, mapping a contract clause to one of N
invoices — these are classification problems that LLM pipelines usually solve by
generating text and parsing it back into a type. This repo implements the same
typed decision interface across four architectures and measures what each one
actually buys.

The interface is modelled on [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
TypeSafe AI's "System One" model: unstructured `state` plus typed `questions` in,
typed values with probabilities out.

Every number below was measured on a live Azure AI Foundry deployment. Where a
result did not replicate, that is stated.

---

## Conclusion first

**Encoders win decisively on throughput and cost. They lose on discrimination,
and the gap widens as the candidate pool grows.**

| Axis | Effect on encoders | Effect on the LLM |
| --- | --- | --- |
| **Volume** 10 → 1000 rows | none — accuracy flat, 610x faster | linear; 26 min for 1000 rows |
| **Phrasing noise** | −0.075 to −0.145 accuracy | none observed |
| **Cardinality** 8 → 255 options | **−0.16 to −0.22 accuracy** | none observed |

For a small, stable option list and high volume, an encoder is the right tool
and the saving is enormous. For a large or crowded option list, it is not — and
no amount of latency advantage compensates for picking the wrong invoice.

**The practical answer is a hybrid**: the encoder shortlists candidates in
milliseconds, the LLM chooses among the shortlist. That uses each where it is
strong — encoder throughput, LLM discrimination over a small set.

---

## Test 1 — the first benchmark, and why it was misleading

Ten contract clauses, eight invoices, three architectures, identical
`ChoiceAnswer` out of each.

```
architecture                           acc        median      mean       min
Foundry encoder (text-embedding-3-small) 10/10    218.9ms    274.0ms    182.0ms
Local BERT encoder (MiniLM-L6 ONNX)    10/10        2.1ms      2.2ms      1.9ms
LLM decoder (gpt-6-luna)               10/10     1721.3ms   1995.7ms   1443.0ms
```

`uv run benchmark_architectures.py`

Read on its own this says **830x faster at identical accuracy**. That claim does
not survive contact with a harder corpus, and the rest of this README is the
story of finding out why.

The flaw: those ten clauses closely paraphrased the invoice descriptions.

- Invoice: "Acme Cloud Hosting — monthly platform **hosting** and **compute capacity**"
- Clause: "…dedicated **compute capacity** and platform **hosting**, billed **monthly**"

High lexical overlap makes cosine similarity look infallible. It is not.

One genuine finding did survive: clause CL-010 ("the parties shall meet
quarterly…") matches no invoice, scored `p=0.676` from the softmax but
`cos=0.336`, and was correctly routed to `no_match`. **Relative probability
cannot express "none of these"; absolute similarity can.** That distinction
holds throughout.

---

## Test 2 — scaling volume

Cardinality held at 8. Only the number of clauses grows, 10 → 1000, with every
clause unique so caching cannot flatter the result.

```
     N    local-batched     local-single  foundry-batched              llm
    10            7.377            7.005           90.999         1758.434
    50            4.792            6.440           43.032         1591.816
   100            3.078            3.224           33.196                -
   250            2.608           12.083           19.733                -
   500            2.767            7.042           15.440                -
  1000            2.717            6.863           16.240                -
```

`uv run benchmark_scaling.py`

**The encoder advantage grows with volume.** Local BERT amortises fixed overhead
and plateaus at ~2.7 ms/item (368 items/s). Foundry gains 5.6x purely from
batching 64 texts per request. The LLM is flat at ~1600 ms/item because no batch
interface exists — it cannot benefit from volume at all. At N=250 the gap is
**610x**; projected wall time for 1000 clauses is **26.5 minutes** versus 2.7
seconds.

Cost per 1k clauses: LLM **$0.1285**, Foundry **~$0.0005**, local **$0.00**.

But accuracy told a different story:

```
  local-batched    argmax 0.700-0.870
  foundry-batched  argmax 0.900-0.950
  llm              argmax 1.000-1.000
```

Crucially, **accuracy was flat across volume** (0.87 at N=100, 0.81 at N=1000).
Volume does not degrade accuracy. Something else did — the corpus had changed.

---

## Exploration 1 — separating capability from threshold

The first runs reported local accuracy at 0.60–0.78. Diagnosing before reporting:

```
argmax-only accuracy: 85/100 = 0.850
threshold=0.35  acc=0.760  abstained=18
cosine: min=0.220  median=0.441  max=0.659
```

Most of the "drop" was **over-abstention**. The 0.35 floor had been tuned on
Test 1's corpus, where cosines ran 0.41–0.70. On the new corpus it discarded 18
mostly-correct answers.

The benchmarks now report `argmax` (model capability) and `routed`/`abstain`
(threshold tuning) separately. Conflating them makes a mis-set constant look
like a weak model.

---

## Exploration 2 — what actually hurt: dilution

Cardinality (8) and volume (200) both fixed. Only the amount of class-neutral
boilerplate varies:

```
core    argmax=0.920  cos median=0.468  mean words=11.0
full    argmax=0.835  cos median=0.424  mean words=19.2
padded  argmax=0.815  cos median=0.435  mean words=37.3
```

Accuracy falls monotonically as filler grows, with the number of classes never
changing. The mechanism is mean pooling:

```python
pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
```

Every token contributes equally. Vendor names, billing cadences and locations
appear identically across all eight classes, so they pull every clause toward a
common centroid. Classes do not become more numerous — they become **closer
together**. An LLM is unaffected because attention can ignore filler; fixed
pooling cannot.

---

## Exploration 3 — conditioning fixes most of it

If dilution comes from question-blind pooling, then conditioning should repair
it. Two conditioned architectures were added and compared on a **held-out seed**
(tuning and reporting never share data):

```
architecture            core      full    padded      drop
bi-encoder             0.890     0.825     0.745    +0.145
late-interaction       0.890     0.865     0.815    +0.075
cross-encoder          0.965     0.925     0.855    +0.110
llm (n=50)             1.000     1.000     1.000    +0.000

latency (ms/item)
  bi-encoder             2.43     2.87     4.18
  late-interaction       2.72     4.45     8.91
  cross-encoder         39.07    38.10    65.67
```

`uv run benchmark_crossencoder.py --n 200 --seed 111`

**Late interaction halves the dilution penalty at bi-encoder cost.** Instead of
pooling once, question-blind:

```python
weights_k = softmax(tokens · q_k)   # per-option attention over cached tokens
pooled_k  = Σ weights_k · tokens    # pooling becomes a function of the question
```

The state is encoded **once** and kept as a token matrix. Conditioning moves out
of the transformer and into a matrix multiply, so cost does not scale with the
number of options. The cross-encoder is more accurate still, but re-encodes the
state per option — 13x the latency at 8 options, and linear in K thereafter.

Tuning `attention_temperature` on a separate seed found the default already
optimal, with a flat plateau from 0.03–0.12:

```
 attn_temp     core     full   padded     mean
     0.010    0.920    0.855    0.845    0.873
     0.050    0.940    0.895    0.870    0.902   <- best
     0.120    0.930    0.905    0.870    0.902   <- tie
     1.000    0.930    0.845    0.795    0.857
     2.000    0.920    0.850    0.790    0.853
```

`uv run tune_attention.py`

The curve is better evidence than the tuning result: at high temperature
attention flattens to uniform and the head **degrades smoothly into plain mean
pooling** (0.850/0.790 ≈ the bi-encoder's 0.860/0.780), exactly as the
dilution hypothesis predicts.

---

## Test 3 — scaling cardinality

Volume (1000 clauses) and phrasing held fixed. Only the candidate pool grows:
the eight real invoices plus distractors drawn from ordinary corporate spend.

```
    K     late-interaction    foundry-embedding          foundry-gpt
    8                0.867                0.943                1.000
   16                0.852                0.921                1.000
   32                0.817                0.869                1.000
   64                0.801                0.810                1.000
  128                0.744                0.774                1.000
  255                0.706                0.720                1.000

ms per item
    8                 3.15                16.93              2415.87
  255                 5.04                15.90              2558.37
```

`uv run benchmark_cardinality.py`

**This is where encoders break.** Both lose 16–22 points from 8 to 255 options
while the LLM shows no errors at any K.

Latency is **flat for all three** — the cost is paid entirely in accuracy, which
is the harder currency. The mechanism is not dilution but **crowding**: 255
prototypes in one embedding space shrink nearest-neighbour margins. Adding "data
centre colocation" puts a near-neighbour beside "cloud hosting", and cosine
ranking cannot reliably separate them. Conditioning does not help, because the
problem is not which tokens to read — two prototypes genuinely look alike.

Two further observations:

- **Foundry's lead evaporates.** It leads at K=8 (0.943 vs 0.867), is level by
  K=64 (0.810 vs 0.801), and degrades faster overall (−0.223 vs −0.161). At
  K=255 the free local model is statistically level with the paid API.
- **The LLM's cost scales with K**: $0.1559 → $1.1527 per 1k clauses, because
  every option sits in the prompt. Still ~2000x slower than late interaction.

---

## What this suggests about Jev

Jev's pricing is the tell: **output tokens are free** ("too cheap to meter")
while input is metered. Free output means there is no decode loop — one forward
pass. Combined with three primitives that map exactly onto standard head types,
and "adding questions barely changes response time" (one encode, many heads),
the evidence points to an encoder with classification heads rather than an
autoregressive LM.

| Primitive | Head | Returns |
| --- | --- | --- |
| `Choice` | N logits → softmax | `choice`, `probabilities`, `confidence` |
| `Score` | ordered logits → softmax | `score`, `probabilities`, `confidence` |
| `Noul` | 1 logit → sigmoid | `noul` (0–1) |

Two findings here corroborate that reading:

1. **A plain cross-encoder cannot be the answer.** Its cost is linear in options,
   so 255 options could not be served at Jev's quoted 70–500 ms. Late
   interaction can, because K only widens a matrix multiply. Jev is more likely
   in that family — which is also what "a new model architecture" would need to
   mean to be interesting.
2. **The 255 cap looks like a crowding boundary, not an arbitrary limit.** An
   independent implementation degrades badly approaching exactly that number,
   and Jev's documented workaround above 255 — *"score independently, then make
   an explicit choice"* — is precisely how you escape crowding: pairwise scoring
   instead of one shared space.

This is inference from published material plus measurements of comparable
architectures. It is **not** a test of Jev, which was never run here. A trained
model would also do better than these zero-shot heads: RLCD optimises the
projections for decision accuracy, where MiniLM was only ever trained for
sentence similarity.

---

## Type safety

For the LLM path the option list is compiled into a strict JSON Schema before
the call — options become an `enum`, with one required probability key each:

```python
Choice(instructions="Which invoice does this clause bill against?",
       options=("INV-1001", "INV-1002", ...))
```

An invoice ID that does not exist **cannot be returned**; it is not
representable in the response schema, not merely discouraged by the prompt. The
encoder paths get this for free — an `argmax` over prototypes can only return an
index that exists.

> **On probabilities:** encoder probabilities are a real softmax over model
> outputs. The LLM's are **not** — there the model writes a number that looks
> like a probability. Treat LLM `probability` as a heuristic. Encoder values are
> genuine but still need temperature tuning before their absolute scale means
> anything.

At high cardinality a full distribution becomes impractical — one required key
per option makes the schema large and the output long. `JevClient` accepts
`include_probabilities=False`, which drops the distribution and reconstructs a
coarse one from the model's confidence. Jev itself switches strategy past 255
options for the same reason.

---

## Usage

All four backends return `ChoiceAnswer`, so they are interchangeable.

```python
from jev import (Choice, EncoderChoiceClassifier, JevClient,
                 LateInteractionChoiceClassifier, LocalBertEncoder)

encoder = LocalBertEncoder()

# Fastest; vulnerable to class-neutral filler.
flat = EncoderChoiceClassifier(encoder, catalogue)

# Query-conditioned pooling at the same cost. Recommended default.
late = LateInteractionChoiceClassifier(encoder, catalogue)
answer = late.classify("Supplier shall provide dedicated compute capacity.")

# Most accurate; use when the decision needs real reasoning.
jev = JevClient(openai_client, model="gpt-6-luna")
answer = jev.system_one(clause, {"invoice": Choice(...)}).choice("invoice")

answer.choice       # 'INV-1001' — always a declared option
answer.probability  # relative score within the catalogue
answer.confidence   # absolute match strength; drives abstention
answer.margin       # gap to the runner-up
```

### Choosing a backend

| Situation | Use |
| --- | --- |
| Few options, high volume, clean text | `EncoderChoiceClassifier` |
| Few options, noisy or verbose text | `LateInteractionChoiceClassifier` |
| Few options, accuracy critical | `CrossEncoderChoiceClassifier` |
| Many options, or real reasoning needed | `JevClient` (LLM) |
| Many options **and** high volume | encoder shortlist → LLM decides |

---

## Running

```bash
cp .env.example .env     # set PROJECT_ENDPOINT and EMBEDDING_ENDPOINT

uv run etl_invoice_lookup.py         # the worked ETL example
uv run benchmark_architectures.py    # Test 1
uv run benchmark_scaling.py          # Test 2 — volume
uv run benchmark_crossencoder.py     # Exploration 3 — dilution
uv run benchmark_cardinality.py      # Test 3 — cardinality
uv run tune_attention.py             # attention temperature sweep
```

| File | Purpose |
| --- | --- |
| [jev/primitives.py](jev/primitives.py) | `Choice` / `Score` / `Noul` and validated answers |
| [jev/client.py](jev/client.py) | Schema compiler and LLM-backed `JevClient` |
| [jev/encoders.py](jev/encoders.py) | Bi-encoder, late-interaction and cross-encoder heads |
| [corpus.py](corpus.py) | Synthetic clause and catalogue generators |

### Tests

```bash
uv run --with pytest --with numpy -m pytest -q   # 25 passed
```

Fully offline — no API calls, no model downloads. Covers schema closure,
out-of-list rejection, probability normalisation, batch chunking, and that late
interaction encodes the state **once** regardless of option count.

---

## Caveats

- **Synthetic corpora.** Clauses and distractors are template-generated. Real
  catalogues with genuinely near-duplicate entries would likely be harder than
  the cardinality test suggests.
- **"1.000" means no observed errors.** The LLM ran on 50 clauses per
  cardinality, where ±0.02 is noise. It is not proven perfection.
- **One result did not replicate.** On the tuning seed the cross-encoder appeared
  completely immune to filler (0.920 → 0.920); on held-out data it lost 0.110.
  The earlier reading was partly luck.
- **Zero-shot heads.** No head here is trained on this task. `temperature`,
  `attention_temperature` and the abstention floor were chosen by sweep or
  inspection, not fitted on labelled data.
- **Environment-specific latency.** Local figures exclude a one-off model
  download; Foundry figures are dominated by network distance to the region.
- **Cross-encoder confidence is not calibrated** for this task — ms-marco logits
  are tuned for search relevance. Ranking is reliable; the absolute scale is not.

## Notes

- Embeddings are served from the **resource** endpoint
  (`https://<resource>.services.ai.azure.com`). The `/api/projects/...` endpoint
  returns 404 for `embeddings.create`.
- The LLM path uses **Chat Completions**. Deployed `gpt-6` snapshots reject
  `json_schema` on the Responses API (`400`) and currently 500 there.
- `Choice` caps at 255 options, matching Jev's documented cardinality limit.
