# Hybrid-Retrieval RAG Benchmark

A local RAG pipeline whose point is the **evaluation**: load → chunk → embed → index → retrieve →
score → generate. Everything runs on Ollama, nothing leaves the machine.

Most RAG demos measure the generator and assume the retriever works. This one measures the
retriever against human relevance judgments — three corpora, three arms (BM25, dense, RRF hybrid) —
anchors two of them to published BEIR numbers, and significance-tests every comparison.

## What it found

**1. "Use hybrid retrieval" is not a portable answer.** Same fusion rule, `RRF_K=60`, unchanged
across all three corpora — and it wins three different shapes:

| corpus | what hybrid wins | what it doesn't |
|---|---|---|
| SciFact | everything | — |
| NFCorpus | precision (P@1 +4.3) | dense holds recall@10 |
| MultiHopRAG | recall (+2.5 @20) | ties BM25 on nDCG@10 |

**2. Only one claim survives everywhere: hybrid beats BM25 alone** (p < 0.05 on all three).
Everything else is corpus-specific, and two comparisons aren't resolvable at these sample sizes at
all. That's a weaker headline than "hybrid wins" and a defensible one — the difference only shows
up if you run the test.

**3. The generator is the bottleneck, not the retriever.** Context recall is 0.97+ on every
answerable question type — the gold document is in the prompt almost every time — yet exact match
collapses from 0.92 to ~0.33 on multi-hop questions. The cause is **over-refusal**: on comparison
questions the model declines 52% of the time while holding the evidence.

**4. Two bugs in the evaluation itself, found and fixed** — unequal search depth between arms, and
`recall@20` scored against lists shorter than 20. Both understated results; details in
[Why the numbers are trustworthy](#why-the-numbers-are-trustworthy).

## Results

`nomic-embed-text`, `CHUNK_CHARS=1000`, `RRF_K=60`, `EVAL_OVERFETCH=6`, document-level scoring at
`k=20`. The reports behind every table ship in `results/*_report.json` — scores, p-values and
per-query win/tie/loss counts, checkable without a GPU. Dependencies are pinned exactly, because a
reproducibility claim that can't reproduce its own environment isn't one.

Each table carries a paired **Fisher randomisation test** on nDCG@10 (1000 permutations, seeded).
A higher number is not a better retriever: with 300 queries a 0.016 gap can be noise. Gaps on
recall/MRR/P@1 are descriptive only — the test covers nDCG@10.

### SciFact — 300 queries, 5,183 abstracts

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

**Anchor holds**: BM25 nDCG@10 = 0.686 vs BEIR's published **0.651** — a different implementation
on the same qrels, which is what validates the indexing and metric path.

Dense *looks* better than BM25 here, but 73 wins against 57 losses is what chance looks like. Two
different ways of being equally good is the ideal setup for fusion, and this is hybrid's largest
win of the three corpora.

### NFCorpus — 323 queries, 3,633 nutrition/medical abstracts

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

**Second anchor holds**: BM25 nDCG@10 = 0.322 against a published 0.29–0.33 band. Two independent
corpora now agree the metric path is sound.

Hybrid takes the precision metrics, dense holds recall@10 — the mirror image of MultiHopRAG below.
But only hybrid-over-bm25 is statistically supported; the ordering above it is unresolved at 323
queries. More queries, not more tuning, is what would settle it.

Read recall against the dataset, not against SciFact: NFCorpus labels a median of 16 gold documents
per query (mean 38.2), so ten slots can't hold them — the arithmetic ceiling on recall@10 is 0.615.
The arms reach 27% of it. Genuinely low, not an artifact.

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

The only corpus large enough to resolve every comparison. Hybrid pulls more gold documents into the
candidate set without ordering the top few better — the recall-strong / precision-flat signature a
cross-encoder reranker exists to fix.

The nDCG@10 tie is real, not rounding: 848 wins against 827 losses over 2,255 queries. BM25 beating
dense outright is the strongest single-arm result here — these queries name entities, dates and
outlets verbatim, which is lexical matching's home ground.

The 301 `null_query` rows have no gold documents and are excluded from every retrieval number above.

## Generation — `qwen3:8b` over the hybrid arm, 300-query sample

| question type | n | EM | F1 | context recall |
|---|---|---|---|---|
| overall | 300 | 0.583 | 0.595 | 0.867 |
| inference_query | 99 | 0.919 | 0.941 | 0.980 |
| comparison_query | 104 | 0.327 | 0.340 | 0.971 |
| temporal_query | 64 | 0.359 | 0.359 | 0.969 |
| null_query | 33 | 0.818 | 0.818 | — |

`null_query` EM **is** the refusal rate: on 33 deliberately unanswerable questions the model
correctly replied `Insufficient information.` 82% of the time. The remaining 18% is hallucination,
measured with no LLM judge involved.

Context recall is 0.97+ on every answerable type, so retrieval is doing its job. The failure is
over-refusal, not wrong answers:

| question type | answered | refused | over-refusal rate |
|---|---|---|---|
| inference_query | 94 | 5 | 0.051 |
| temporal_query | 37 | 27 | 0.422 |
| comparison_query | 50 | 54 | **0.519** |

Only 33 of 267 answerable questions got a confidently wrong answer. Everything else scoring zero was
the model declining while holding the evidence. Multi-hop questions need two articles combined, and
an 8B model under a strict context-only instruction reads "I must synthesise this myself" as "the
context doesn't contain it".

So the next experiment is a **prompt change, not a retrieval change** — and it has to be scored on
both numbers at once, since loosening refusal trades `null_query` EM away.

## Why the numbers are trustworthy

**Two published anchors.** BM25 lands within 3.5 points of BEIR on SciFact and inside the published
band on NFCorpus. Independent confirmation that indexing and metrics are sound.

**Every arm searches the same depth.** `hybrid()` over-fetches internally, so a caller that already
scaled its depth made hybrid search 3× the candidate pool of the other arms. All comparisons route
through `retrieve.ranked_docs()` and a test asserts it. Fixing this *raised* hybrid recall@20 by
0.007 (NFCorpus) and 0.005 (SciFact) — a deeper pool lets both-arms-agree documents crowd out a gold
document only one arm found.

**Retrieval depth outlasts the chunk→document collapse.** Metrics are document-level but arms return
chunks, and MultiHopRAG averages ~15 chunks per article. At the old depth, 28% of its queries
collapsed to fewer than 20 documents — `recall@20` was scored against a list with no 20th slot.
`EVAL_OVERFETCH=6` fixes it and is measured to saturate. This understated every arm by 0.014–0.023.

**Reproducibility.** BM25 reproduces bit-for-bit; Ollama embeddings are bit-identical and
batch-invariant. Dense and hybrid are stable to **three decimals, not four** — Chroma's HNSW is
approximate, and repeat runs land within ~0.0005. The approximation itself costs nothing: checked
against exact brute-force cosine over every vector, ≤0.003 nDCG@10 on NFCorpus and 0.000 on SciFact.

**Judge-free by design.** No LLM grades any headline number — a local 8B model is too noisy to be a
trustworthy judge. Refusal is measured by ordinary exact match because MultiHopRAG ships
`"Insufficient information."` as the gold answer for its unanswerable queries.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt --python .venv/Scripts/python.exe
ollama pull nomic-embed-text   # embeddings, 768-d
ollama pull qwen3:8b           # generation
```

`EMBED_MODEL` is an env override, so the whole evaluation re-runs on a different embedder without
touching code. Measured on a Quadro T1000 (4 GB):

| model | ms/doc | dim | SciFact ingest | note |
|---|---|---|---|---|
| `bge-m3` | 1097 | 1024 | ~95 min | best quality, too slow on 4 GB |
| `nomic-embed-text` | 379 | 768 | ~33 min | **default** |
| `all-minilm` | 31 | 384 | ~3 min | truncates at 256 tokens — silently drops half of every abstract |

## Run

```bash
python -m pytest tests -q                              # RRF math + index alignment + depth invariant

python -m src.ingest   --corpus scifact
python -m src.evaluate --corpus scifact --retrieval-only     # repeat for nfcorpus
python -m src.ingest   --corpus multihoprag
python -m src.evaluate --corpus multihoprag --limit 300      # includes generation

python -m src.generate --corpus multihoprag "Who faced fraud charges over FTX?"
streamlit run app.py                                   # browser UI
```

The Streamlit app runs all three arms on one question side by side, marks the human-judged relevant
documents, and scores each arm — pick a labelled query from the sidebar to see which arm was right.
Generation is off by default. BM25 is the only fully offline arm; dense and hybrid need
`ollama serve` even in retrieval-only mode, since the query itself must be embedded.

Reports land in `results/*.json`. A `--limit` run writes `*_sample<N>_report.json` so it never
overwrites a full-corpus table. Generation checkpoints to `*_gen.jsonl` after every query so a
killed run resumes; the filename carries corpus, arm, `GEN_MODEL` and `TOP_K`, so changing any of
them starts a fresh cache instead of silently resuming another model's answers.

### Why a generated answer takes ~35 s

Not retrieval — that's 1.5 ms of search plus ~100 ms to embed the query. It's **prompt prefill**:
`qwen3:8b` needs 6.0 GB against a 4 GB card, so Ollama runs it 61% on CPU and prefill crawls at
~17 ms/token. Ollama's own counters put `load_duration` at 0.4 s, so nothing is being reloaded.

Prefill is linear in `TOP_K × CHUNK_CHARS`: `TOP_K=10` → 43.5 s, `5` → 19.9 s, `3` → 11.1 s. So the
levers are `TOP_K` (trades against context recall) or a generator that fits in VRAM.

*Tried and rejected:* batching all retrievals before all generations, to dodge a model swap. Built
and timed at 34.0 s/query against 34.8 s — there is no swap to avoid. Recorded in
`evaluate.eval_generation` so it doesn't get re-attempted.

## How it works

**Datasets** — every corpus ships human `query → gold doc id` labels, and that is not negotiable: a
question generated *from* a chunk is trivially retrievable by lexical overlap, which inflates BM25
and destroys the whole comparison. [MultiHopRAG](https://huggingface.co/datasets/yixuantt/MultiHopRAG)
(long articles, so chunking is a real stage; 301 deliberately unanswerable queries),
[BEIR SciFact](https://huggingface.co/datasets/BeIR/scifact) (the published anchor),
[BEIR NFCorpus](https://huggingface.co/datasets/BeIR/nfcorpus) (graded 1–2 relevance, densely
labelled).

**Two indexes, one chunk list** (`src/ingest.py`) — Chroma (dense, cosine) and bm25s (lexical) are
built from the same list in the same order, so a bm25s positional result and a Chroma `chunk_id`
name the same text. Drift between them is the classic way a hybrid retriever produces confident
nonsense, so ingest asserts sizes match and a test re-checks the `chunk_id → doc_id` mapping in both.

**Fusion** (`src/retrieve.py`) — Reciprocal Rank Fusion:

```
score(d) = Σ_arms  1 / (RRF_K + rank_arm(d))        rank is 1-based
```

Rank-based, so BM25's unbounded term-weight sums and cosine's `[-1, 1]` never need normalising onto
a common scale. `RRF_K` sets how much a lone rank-1 hit is worth against agreement between arms — a
knob worth sweeping, not a constant.

**Scoring granularity** — the datasets label documents, not chunks, so `to_docs()` collapses chunk
hits to deduped documents (best rank wins) and metrics are document-level.

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

- **Cross-encoder reranking** — add when `recall@20` is strong but `nDCG@10` is weak. MultiHopRAG is
  exactly that shape now.
- **Query rewriting / HyDE** — only if multi-hop recall stays poor after sweeping `RRF_K`.
- **Synthetic QA generation** (RAGAS `TestsetGenerator`) — unnecessary here; every corpus ships human
  labels. Worth adding only for a private unlabeled corpus.
- **RAGAS faithfulness** — needs an LLM judge; the deterministic metrics above cover the same ground
  more cheaply and more defensibly.
- **nomic task prefixes** — tested, not adopted. Re-embedding SciFact with `search_document:` /
  `search_query:` moved nDCG@10 by +0.0005. Not worth a re-ingest.
