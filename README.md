# Hybrid-Retrieval RAG with Evaluation

Full local RAG pipeline — load → chunk → embed → vector store → **hybrid retrieval** →
**evaluation** → generation. Everything runs on Ollama; nothing leaves the machine.

The point of the project is the evaluation. Most RAG demos measure the generator and assume the
retriever works. This one measures the retriever against human relevance judgments, on three corpora,
across three arms — BM25, dense, and RRF-fused hybrid — and anchors two of them to published
numbers so the results are checkable rather than self-graded.

## Why these datasets

Evaluation needs two label types, and only one of them can be synthesized honestly:

| Layer | Metric | Label needed | Can an LLM fake it? |
|---|---|---|---|
| Retrieval | nDCG@10, recall@k, MRR | `query → gold doc id` | **No** — a question generated *from* a chunk is trivially retrievable by lexical overlap, which inflates BM25 and destroys the hybrid comparison |
| Generation | EM / F1 | `query → gold answer` | Yes, but no need here |

So every corpus ships human labels:

- **[MultiHopRAG](https://huggingface.co/datasets/yixuantt/MultiHopRAG)** — 609 full-text news
  articles, 2,556 multi-hop queries. Long documents, so chunking is a real stage. Gold answers
  *and* per-document supporting evidence, so it covers both layers from one download. 301 of the
  queries are deliberately unanswerable (`null_query`) with gold answer `"Insufficient
  information."` — that gives a hallucination measurement with no LLM judge involved.
- **[BEIR SciFact](https://huggingface.co/datasets/BeIR/scifact)** — 5,183 abstracts, 300 test
  queries, human qrels. Pre-chunked and retrieval-only, but it produces numbers comparable to the
  literature: BEIR reports **BM25 nDCG@10 = 65.1**. That is the sanity anchor for the whole
  indexing path.
- **[BEIR NFCorpus](https://huggingface.co/datasets/BeIR/nfcorpus)** — 3,633 nutrition/medical
  abstracts, 323 test queries. A domain corpus rather than a general one, and the only one here with
  **graded** relevance (1 or 2, not just relevant/not), which is what nDCG was designed for. Also
  densely labelled — a median of 16 gold documents per query against SciFact's 1. Published BM25
  nDCG@10 sits in a 0.29–0.33 band depending on implementation, so it is a second checkable anchor.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt --python .venv/Scripts/python.exe
ollama pull nomic-embed-text   # embeddings, 768-d
ollama pull qwen3:8b           # generation
```

### Choosing an embedder

`EMBED_MODEL` is an environment override, so the whole evaluation can be re-run on a different
embedder without touching code. Measured on a Quadro T1000 (4 GB) over 1,488-char SciFact
abstracts:

| model | ms/doc | dim | SciFact ingest | note |
|---|---|---|---|---|
| `bge-m3` | 1097 | 1024 | ~95 min | best quality, too slow on 4 GB |
| `nomic-embed-text` | 379 | 768 | ~33 min | **default** |
| `all-minilm` | 31 | 384 | ~3 min | truncates at 256 tokens — silently drops half of every abstract |

```bash
EMBED_MODEL=bge-m3 python -m src.ingest --corpus scifact   # then re-evaluate
```

## Run

```bash
python -m pytest tests -q                              # RRF math + index alignment

python -m src.ingest   --corpus scifact
python -m src.evaluate --corpus scifact --retrieval-only

python -m src.ingest   --corpus nfcorpus
python -m src.evaluate --corpus nfcorpus --retrieval-only

python -m src.ingest   --corpus multihoprag
python -m src.evaluate --corpus multihoprag --limit 300

python -m src.generate --corpus multihoprag "Who faced fraud charges over FTX?"

streamlit run app.py                                   # browser UI
```

### Browser UI

`app.py` is a Streamlit front-end over the same indexes — corpus and arm selectors,
`top_k` / `RRF_K` sliders, and the retrieved chunks shown next to the answer. The
"compare all three arms" box runs dense, bm25 and hybrid on one question and lists
the top documents side by side, which is the disagreement the results tables below
are about.

Generation is off by default: retrieval answers in ~1 s, a generated answer is
12–34 s on a 4 GB card because ollama swaps the embed and chat models per query.
The dense and hybrid arms need `ollama serve` running even in retrieval-only mode —
the query itself has to be embedded. BM25 is the only fully offline arm.

Reports land in `results/*.json`; a `--limit` run writes `*_sample<N>_report.json` so a sampled run
never overwrites a full-corpus table. Chroma persists, so re-running ingest is only slow the first
time.

Generation checkpoints to `results/*_gen.jsonl` after every query, so a killed run resumes where it
stopped. The filename carries the corpus, arm, `GEN_MODEL` and `TOP_K`, so changing any of those
starts a fresh cache rather than silently resuming another model's answers — the cache is keyed on
query id alone, and that footgun is the kind that gets wrong numbers published. Delete the file by
hand only when changing the *prompt*, which the filename cannot see.

### Where generation time actually goes

A 300-query generation run takes 2 h 54 m — **34.8 s/query** median. Retrieval is not the reason:
the search itself is 1.5 ms (bm25s 0.36, RRF 0.02, chunk→doc 0.01, Chroma text fetch 1.06) and
query embedding is ~100 ms.

It is almost entirely **prompt prefill**. Ollama's own counters, per call:

```
load_duration      0.4 s     model is resident; nothing is being reloaded
prompt_eval       33   s     ~1,900 tokens at ~17 ms/token
eval (decode)      0.3-1.3 s answers are 2-6 tokens
```

`qwen3:8b` needs 6.0 GB and the card has 4.0 GB, so ollama runs it **61% CPU / 39% GPU** and
prefill crawls. Prefill is linear in prompt size, which is `TOP_K × CHUNK_CHARS`:

| TOP_K | prompt tokens | wall |
|---|---|---|
| 10 | 2363 | 43.5 s |
| 5 | 1200 | 19.9 s |
| 3 | 689 | 11.1 s |

So the levers are `TOP_K` and a generator that fits in VRAM (`phi3:mini` at 2.2 GB would).
Lowering `TOP_K` trades against context recall, which is 0.97 on answerable types at `TOP_K=10`,
so it has to be scored on both numbers at once.

**What does not work: batching the phases.** Retrieving everything before generating anything, to
avoid an embed/chat model swap, was built and timed — 34.0 s/query against 34.8 s interleaved.
There is no swap to avoid: `load_duration` is 0.4 s on every call because qwen3 spills to CPU
instead of evicting. The idea is recorded in `evaluate.eval_generation` so it does not get
re-attempted.

## Results

Document-level retrieval, all three arms. Regenerate with the commands above — or read the
reports behind these tables directly, since `results/*_report.json` ships in the repo: scores,
p-values and per-query win/tie/loss counts, checkable without a GPU. Dependencies are pinned
exactly (`requirements.txt`, Python 3.12.6) because a reproducibility claim that cannot
reproduce its own environment is not one.

`nomic-embed-text` embeddings, `CHUNK_CHARS=1000`, `RRF_K=60`, `EVAL_OVERFETCH=6`, document-level
scoring at `k=20`. Every arm is searched at the same candidate depth — see
[Fair comparison](#fair-comparison) for why that needs saying.

**Arm-vs-arm claims are significance-tested.** A higher number is not a better retriever: with
300 queries a 0.016 gap can be noise. Each table is followed by a paired Fisher randomisation
test on nDCG@10 (1000 permutations, seeded), reported as wins/ties/losses per query and a
p-value. Differences that fail at p < 0.05 are called out as not distinguishable, not as wins.

The test covers **nDCG@10 only** — it is the headline metric and the one the anchors are quoted
against. Gaps on recall, MRR and P@1 are descriptive: read them as what happened on this query
set, not as established orderings. A 0.002 recall gap between two arms is below the noise floor
of the dense index either way (see [Reproducibility](#reproducibility)).

### SciFact — 300 queries, 5,183 abstracts (unchunked)

| arm | nDCG@10 | recall@10 | recall@20 | MRR@10 | P@1 |
|---|---|---|---|---|---|
| dense | 0.702 | 0.846 | 0.897 | 0.662 | 0.577 |
| bm25 | 0.686 | 0.819 | 0.857 | 0.649 | 0.550 |
| **hybrid** | **0.731** | **0.857** | **0.909** | **0.698** | **0.610** |

| comparison | W/T/L | p | verdict |
|---|---|---|---|
| hybrid vs bm25 | 72/203/25 | 0.000 | hybrid wins |
| hybrid vs dense | 69/192/39 | 0.020 | hybrid wins |
| dense vs bm25 | 73/170/57 | 0.341 | **not distinguishable** |

**Sanity anchor holds**: BM25 nDCG@10 = 0.686 against BEIR's published 0.651 — within ~3.5 points
of a number produced by a different implementation on the same qrels, which is what validates the
indexing and metric path. (This BM25 indexes title and abstract as one field and uses bm25s'
default parameters, so exact agreement was never expected.)

Hybrid leads on every metric, and the test backs it against both single arms. The dense-over-bm25
gap does not survive: 73 wins against 57 losses over 300 queries is what chance looks like
(p = 0.34). On this corpus dense and bm25 are two different ways of being equally good — which is
the ideal setup for fusion, and exactly where hybrid posts its largest win of the three corpora.

### NFCorpus — 323 queries, 3,633 nutrition/medical abstracts (unchunked)

| arm | nDCG@10 | recall@10 | recall@20 | MRR@10 | P@1 |
|---|---|---|---|---|---|
| dense | 0.341 | **0.169** | 0.203 | 0.536 | 0.440 |
| bm25 | 0.322 | 0.147 | 0.177 | 0.525 | 0.440 |
| **hybrid** | **0.351** | 0.167 | 0.203 | **0.565** | **0.483** |

| comparison | W/T/L | p | verdict |
|---|---|---|---|
| hybrid vs bm25 | 124/133/66 | 0.000 | hybrid wins |
| hybrid vs dense | 110/124/89 | 0.090 | **not distinguishable** |
| dense vs bm25 | 121/107/95 | 0.058 | **not distinguishable** |

**Second anchor holds**: BM25 nDCG@10 = 0.322 against a published band of 0.29–0.33. Two
independent corpora now agree that the indexing and metric path is sound.

Read the recall column against the dataset, not against SciFact. NFCorpus labels a median of 16
gold documents per query (mean 38.2, one query has 475), so ten slots cannot hold them all — the
arithmetic ceiling on recall@10 is 0.615. The arms reach 0.17, about 27% of it. Low, but genuinely
low, not an artifact.

**This corpus splits by metric type.** Hybrid takes the precision metrics — P@1 by 4.3 points
over both single arms, MRR@10 by 2.9 — while dense holds recall@10 (0.169 vs 0.167) and the two
are level on recall@20. Fusion is promoting the right document into the top slot without widening
the candidate set. On MultiHopRAG it does exactly the opposite. Same fusion, same `RRF_K`,
opposite effect, which is the argument for measuring per corpus rather than adopting hybrid on
reputation.

**But only the hybrid-over-bm25 result is statistically supported here.** Hybrid over dense is
p = 0.090 and dense over bm25 is p = 0.058 — both suggestive, neither conclusive at 323 queries.
The honest reading is that hybrid clearly beats the lexical arm, and the three-way ordering above
it is unresolved. The plain-language health questions ("Do Cholesterol Statin Drugs Cause Breast
Cancer?") against clinical abstract vocabulary are the kind of mismatch where embeddings should
earn their cost, and dense does lead on points — but this corpus is too small to prove it. More
queries, not more tuning, is what would settle it.

(dense and bm25 tie exactly on P@1 at 0.4396 — 142 of 323 queries each. Coincident counts on
different query sets, not a duplicated run; the two arms differ on every other metric.)

### MultiHopRAG — 2,255 labelled queries, 609 articles → 9,345 chunks

| arm | nDCG@10 | recall@10 | recall@20 | MRR@10 | P@1 |
|---|---|---|---|---|---|
| dense | 0.642 | 0.771 | 0.925 | 0.714 | 0.576 |
| bm25 | **0.719** | 0.824 | 0.931 | **0.799** | **0.692** |
| **hybrid** | **0.719** | **0.842** | **0.956** | 0.786 | 0.671 |

| comparison | W/T/L | p | verdict |
|---|---|---|---|
| bm25 vs dense | 1197/371/687 | 0.000 | bm25 wins |
| hybrid vs dense | 1260/530/465 | 0.000 | hybrid wins |
| hybrid vs bm25 | 848/580/827 | 0.877 | **statistically tied** |

A more honest split than SciFact, and the only corpus with enough queries to resolve every
comparison. Hybrid wins recall at both depths (+1.8 pts @10, +2.5 @20 over BM25) but comes in
slightly behind on MRR@10 and P@1. Fusion pulls more gold documents into the candidate set without
ordering the top few better — the recall-strong / precision-flat signature that a cross-encoder
reranker exists to fix.

The nDCG@10 tie is a real tie, not a rounding coincidence: 848 wins against 827 losses across
2,255 queries, p = 0.877. At this sample size the test would have caught a genuine gap easily —
it caught both of the others at p = 0.000. BM25 beating the dense arm outright is the strongest
single-arm result in the project: these queries name entities, dates and outlets verbatim, which
is lexical matching's home ground.

The 301 `null_query` rows carry no gold documents and are excluded from every retrieval number
above; they are scored only as refusals below.

### Generation — `qwen3:8b` over the hybrid arm, 300-query seeded sample

| question type | n | EM | F1 | context recall |
|---|---|---|---|---|
| overall | 300 | 0.583 | 0.595 | 0.867 |
| inference_query | 99 | 0.919 | 0.941 | 0.980 |
| comparison_query | 104 | 0.327 | 0.340 | 0.971 |
| temporal_query | 64 | 0.359 | 0.359 | 0.969 |
| null_query | 33 | 0.818 | 0.818 | — |

`null_query` EM **is** the refusal rate: on 33 deliberately unanswerable questions the model
correctly replied `Insufficient information.` 82% of the time. Hallucination on unanswerable
input is the remaining 18%, measured without an LLM judge.

**The generator is the bottleneck, not the retriever.** Context recall is 0.97+ on every
answerable type — the gold document is in the prompt almost every time — yet EM collapses from
0.919 on single-fact `inference_query` to ~0.33 on `comparison_query` and `temporal_query`.

The failure is over-refusal, not wrong answers. Counting refusals on questions that *do* have an
answer:

| question type | answered | refused | over-refusal rate |
|---|---|---|---|
| inference_query | 94 | 5 | 0.051 |
| temporal_query | 37 | 27 | 0.422 |
| comparison_query | 50 | 54 | **0.519** |

Only 33 of 267 answerable questions got a confidently wrong answer. Everything else that scored
zero was the model declining while holding the evidence. Comparison and temporal questions need
two articles combined ("did outlet A publish before outlet B?"), and an 8B model handed the
strict context-only instruction treats "I must synthesise this myself" as "the context does not
contain it".

That makes the next experiment a prompt change, not a retrieval change — the refusal instruction
is currently tuned to keep `null_query` honest, and it is costing roughly half of the multi-hop
answers. Any fix has to be scored on both numbers at once, since loosening refusal will trade
`null_query` EM away.

The three tables together are the argument for measuring rather than assuming. Hybrid wins
outright on SciFact, wins on precision but not recall on NFCorpus, and wins on recall but ties on
nDCG@10 on MultiHopRAG — three corpora, three different shapes of result, one unchanged fusion
rule. Which arm to ship, and whether a reranker would pay for itself, is a per-corpus question
with a per-corpus answer. No amount of generator-side evaluation would have surfaced any of it.

The one claim that holds everywhere, at p < 0.05 on all three corpora, is **hybrid beats BM25
alone**. Everything else is corpus-specific, and two of the side comparisons are not resolvable
at these sample sizes at all. That is a weaker headline than "hybrid wins" and a more defensible
one — and the difference between the two only shows up if you run the test.

### Fair comparison

Two things had to be true before any of the above could be trusted, and neither was free:

**Every arm searches the same candidate depth.** `hybrid()` over-fetches internally before fusing,
so a caller that has already scaled its depth makes hybrid search `depth × OVERFETCH` while dense
and bm25 see `depth` — three times the candidate pool for one arm in a three-arm comparison.
Everything routes through `retrieve.ranked_docs()` for this reason, and `tests/test_rag.py` spies
on the depth each arm requests. It is not a harmless bug in either direction: fixing it *raised*
hybrid recall@20 by 0.007 (NFCorpus) and 0.005 (SciFact), because a deeper pool lets
both-arms-agree documents accumulate above a gold document that only one arm found.

**Retrieval depth must outlast the chunk→document collapse.** Metrics are document-level, but the
arms return chunks, and MultiHopRAG averages ~15 chunks per article. At `k × 3`, 28% of its
queries collapsed to fewer than 20 distinct documents — so recall@20 was scored against a list
with no 20th slot. `EVAL_OVERFETCH=6` fixes it and is measured to saturate; 12 changes nothing.
This understated every arm, by 0.014 (bm25) to 0.023 (dense).

### Reproducibility

BM25 is exact and reproduces bit-for-bit. Ollama embeddings are bit-identical across calls and
invariant to batch composition (checked). The dense and hybrid arms are stable to **three
decimals, not four**: Chroma's HNSW is an approximate index, and repeat full runs at identical
config land within ~0.0005 of each other. That is why the tables above are quoted to three.

The approximation itself costs nothing measurable. Checked against exact brute-force cosine over
every vector: ≤0.003 nDCG@10 on NFCorpus and 0.000 on SciFact. `hnsw:search_ef` is pinned at
ingest so the neighbours no longer depend on how many results were requested.

## How it works

**Chunking** (`src/data.py`) — paragraph-aware greedy packer, `CHUNK_CHARS` cap with
`CHUNK_OVERLAP` carry-over across the seam. SciFact abstracts are already passage-sized and are
indexed whole.

**Two indexes, one chunk list** (`src/ingest.py`) — Chroma (dense, cosine) and bm25s (lexical) are
built from the same `chunks` list in the same order, so a bm25s positional result and a Chroma
`chunk_id` name the same text. Drift between the two is the classic way a hybrid retriever produces
confident nonsense, so ingest asserts the sizes match and `tests/test_rag.py` re-checks the
`chunk_id → doc_id` mapping in both stores.

**Fusion** (`src/retrieve.py`) — Reciprocal Rank Fusion:

```
score(d) = Σ_arms  1 / (RRF_K + rank_arm(d))        rank is 1-based
```

Rank-based, so BM25's unbounded term-weight sums and cosine's `[-1, 1]` never have to be
normalised onto a common scale. Each arm over-fetches `TOP_K × OVERFETCH` before fusing.
`RRF_K` controls how much a lone rank-1 hit is worth versus agreement between arms — it is a
knob worth sweeping, not a constant. Fusion depth is a second such knob, and it does not point
the same way on every corpus: see [Fair comparison](#fair-comparison).

**Scoring granularity** — both datasets label whole documents, not chunks, so `to_docs()` collapses
chunk hits to deduped documents (best rank wins) and metrics are computed at document level: any
chunk of a gold document counts as a hit.

**Generation** (`src/generate.py`) — context-only prompt that must reply with the exact string
`Insufficient information.` when the context does not support an answer. MultiHopRAG uses that
same string as the gold answer for its unanswerable queries, so refusal is scored by the ordinary
exact-match metric.

**Metrics** (`src/evaluate.py`) — `ranx` for nDCG/recall/MRR/P@1. SQuAD-style normalised exact
match and token F1 for answers (MultiHopRAG answers are 2–25 characters, so EM is meaningful).
Deterministic and judge-free by design: a local 8B model is too noisy to be a trustworthy judge,
so no headline number depends on one.

**Significance** (`src/evaluate.py`) — every pair of arms gets a paired Fisher randomisation test
on nDCG@10 (`ranx.compare`, 1000 permutations, seeded), written into `results/*_report.json`
alongside the scores. Point estimates rank arms; only the test says whether the ranking would
survive a different query sample. Two comparisons in this project do not.

## Layout

```
src/config.py    tunable constants -- models, chunk size, TOP_K, RRF_K, depths
src/data.py      dataset loading + chunking (one loader covers any BEIR dataset)
src/embed.py     batched Ollama embeddings
src/ingest.py    build Chroma + bm25s from one shared chunk list
src/retrieve.py  dense / bm25 / hybrid, RRF, chunk->doc collapsing, equal-depth entry point
src/generate.py  context-only answering
src/evaluate.py  3-arm retrieval metrics + significance tests + EM/F1 + refusal rate
tests/test_rag.py
```

## Not built (and when to add it)

- **Cross-encoder reranking** — add if hybrid `recall@20` is strong but `nDCG@10` is weak. That
  exact gap is what a reranker closes.
- **Query rewriting / HyDE** — only if multi-hop recall stays poor after sweeping `RRF_K`.
- **Synthetic QA generation** (RAGAS `TestsetGenerator`) — unnecessary here; both corpora ship
  human labels. Worth adding only when pointing the pipeline at a private unlabeled corpus.
- **RAGAS faithfulness** — would need an LLM judge; the deterministic metrics above cover the
  same ground more cheaply.
