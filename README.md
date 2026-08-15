# Hybrid-Retrieval RAG Benchmark

A local RAG pipeline whose point is the **evaluation**. Most RAG demos measure the generator and
assume the retriever works. This one measures the retriever against human relevance judgments —
three corpora, three arms (BM25, dense, RRF hybrid) — anchors two of them to published BEIR
numbers, and significance-tests every comparison. Runs entirely on Ollama.

## What it found

**1. "Use hybrid retrieval" is not a portable answer.** Same fusion rule, `RRF_K=60`, unchanged
across all three corpora — three different shapes of win:

| corpus | hybrid wins | hybrid doesn't |
|---|---|---|
| SciFact | everything | — |
| NFCorpus | precision (P@1 +4.3) | dense holds recall@10 |
| MultiHopRAG | recall (+2.5 @20) | ties BM25 on nDCG@10 |

**2. Only one claim survives everywhere: hybrid beats BM25 alone.** Everything else is
corpus-specific, and two comparisons aren't resolvable at these sample sizes at all. A weaker
headline than "hybrid wins" — and the difference only shows up if you run the test.

**3. The generator is the bottleneck, not the retriever.** Context recall is 0.97+ on every
answerable question type, yet exact match collapses from 0.92 to ~0.33 on multi-hop questions. The
cause is **over-refusal**: on comparison questions the model declines 52% of the time while holding
the evidence in its prompt.

**4. Two bugs in the evaluation itself**, found and fixed — arms searching unequal depth, and
`recall@20` scored against lists shorter than 20.

## Results

`nomic-embed-text`, `CHUNK_CHARS=1000`, `RRF_K=60`, `EVAL_OVERFETCH=6`, document-level scoring at
`k=20`. Full reports ship in `results/*_report.json`.

### SciFact — 300 queries, 5,183 abstracts

| arm | nDCG@10 | recall@10 | recall@20 | MRR@10 | P@1 |
|---|---|---|---|---|---|
| dense | 0.702 | 0.846 | 0.897 | 0.662 | 0.577 |
| bm25 | 0.686 | 0.819 | 0.857 | 0.649 | 0.550 |
| **hybrid** | **0.731** | **0.857** | **0.909** | **0.698** | **0.610** |

**Anchor holds**: BM25 = 0.686 vs BEIR's published **0.651** — a different implementation on the
same qrels, which validates the indexing and metric path.

### NFCorpus — 323 queries, 3,633 nutrition/medical abstracts

| arm | nDCG@10 | recall@10 | recall@20 | MRR@10 | P@1 |
|---|---|---|---|---|---|
| dense | 0.341 | **0.169** | 0.203 | 0.536 | 0.440 |
| bm25 | 0.322 | 0.147 | 0.177 | 0.525 | 0.440 |
| **hybrid** | **0.351** | 0.167 | 0.203 | **0.565** | **0.483** |

**Second anchor holds**: BM25 = 0.322 against a published 0.29–0.33 band. Recall looks low because
NFCorpus labels a median of 16 gold documents per query — the ceiling on recall@10 is 0.615.

### MultiHopRAG — 2,255 labelled queries, 609 articles → 9,345 chunks

| arm | nDCG@10 | recall@10 | recall@20 | MRR@10 | P@1 |
|---|---|---|---|---|---|
| dense | 0.642 | 0.771 | 0.925 | 0.714 | 0.576 |
| bm25 | **0.719** | 0.824 | 0.931 | **0.799** | **0.692** |
| **hybrid** | **0.719** | **0.842** | **0.956** | 0.786 | 0.671 |

BM25 beats dense outright — these queries name entities, dates and outlets verbatim, which is
lexical matching's home ground. Hybrid pulls more gold documents into the candidate set without
ordering the top few better: the recall-strong / precision-flat signature a reranker exists to fix.

### Significance

Paired Fisher randomisation test on nDCG@10, 1000 permutations, seeded. A higher number is not a
better retriever — with 300 queries a 0.016 gap can be noise. Per-query win/tie/loss counts are in
the shipped reports.

| comparison | SciFact | NFCorpus | MultiHopRAG |
|---|---|---|---|
| hybrid vs bm25 | **0.000** ✓ | **0.000** ✓ | 0.877 — tied |
| hybrid vs dense | **0.020** ✓ | 0.090 — ns | **0.000** ✓ |
| dense vs bm25 | 0.341 — ns | 0.058 — ns | **0.000** bm25 |

`ns` = not distinguishable. Dense *looks* better than BM25 on SciFact (0.702 vs 0.686), but 73 wins
against 57 losses is what chance looks like. Two arms being equally good by different routes is the
ideal setup for fusion, and that is where hybrid posts its largest win.

## Generation — `qwen3:8b` over the hybrid arm, 300-query sample

| question type | n | EM | F1 | context recall | over-refusal |
|---|---|---|---|---|---|
| overall | 300 | 0.583 | 0.595 | 0.867 | — |
| inference_query | 99 | 0.919 | 0.941 | 0.980 | 0.051 |
| temporal_query | 64 | 0.359 | 0.359 | 0.969 | 0.422 |
| comparison_query | 104 | 0.327 | 0.340 | 0.971 | **0.519** |
| null_query | 33 | 0.818 | 0.818 | — | — |

`null_query` EM **is** the refusal rate: on 33 deliberately unanswerable questions the model
correctly declined 82% of the time. The remaining 18% is hallucination, measured with no LLM judge.

Only 33 of 267 answerable questions got a confidently wrong answer — everything else scoring zero
was the model declining while holding the evidence. Multi-hop questions need two articles combined,
and an 8B model under a strict context-only instruction reads "I must synthesise this myself" as
"the context doesn't contain it". So the next experiment is a **prompt change, not a retrieval
change**, scored on both numbers at once since loosening refusal trades `null_query` EM away.

## Why the numbers are trustworthy

- **Two published anchors.** BM25 lands within 3.5 points of BEIR on SciFact and inside the
  published band on NFCorpus.
- **Every arm searches the same depth.** `hybrid()` over-fetches internally, so a caller that
  already scaled its depth gave hybrid 3× the candidate pool. Fixing it *raised* hybrid recall@20
  by 0.005–0.007; a test now asserts the invariant.
- **Depth outlasts the chunk→document collapse.** At the old depth, 28% of MultiHopRAG queries
  collapsed to fewer than 20 documents, so `recall@20` was scored against a list with no 20th slot.
  Understated every arm by 0.014–0.023.
- **Reproducible.** BM25 is bit-exact; embeddings are bit-identical and batch-invariant. Dense and
  hybrid are stable to three decimals — HNSW is approximate, but checked against brute-force cosine
  over every vector it costs ≤0.003 nDCG@10. Dependencies pinned exactly.
- **Judge-free.** No LLM grades any headline number. Refusal is scored by ordinary exact match,
  because MultiHopRAG ships `"Insufficient information."` as its gold answer for unanswerable
  queries.

## Setup & run

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt --python .venv/Scripts/python.exe
ollama pull nomic-embed-text && ollama pull qwen3:8b

python -m pytest tests -q                                # RRF math, index alignment, depth invariant
python -m src.ingest   --corpus scifact                  # also: nfcorpus, multihoprag
python -m src.evaluate --corpus scifact --retrieval-only
python -m src.evaluate --corpus multihoprag --limit 300   # includes generation
streamlit run app.py                                     # browser UI
```

The Streamlit app runs all three arms on one question side by side, marks the human-judged relevant
documents and scores each arm — pick a labelled query from the sidebar to see which arm was right.
BM25 is the only fully offline arm; dense and hybrid need `ollama serve` even for retrieval, since
the query itself must be embedded.

`EMBED_MODEL` is an env override, so the whole evaluation re-runs on another embedder without
touching code. On a Quadro T1000 (4 GB): `bge-m3` 1097 ms/doc (best, too slow), `nomic-embed-text`
379 ms/doc (default), `all-minilm` 31 ms/doc — but it truncates at 256 tokens and silently drops
half of every abstract.

**Why a generated answer takes ~35 s:** not retrieval (1.5 ms of search + ~100 ms to embed), but
prompt prefill. `qwen3:8b` needs 6.0 GB against a 4 GB card, so Ollama runs it 61% on CPU at
~17 ms/token. Prefill is linear in `TOP_K`: 10 → 43.5 s, 5 → 19.9 s, 3 → 11.1 s.

## How it works

- **Datasets** — every corpus ships human `query → gold doc id` labels, and that is not negotiable:
  a question generated *from* a chunk is trivially retrievable by lexical overlap, which inflates
  BM25 and destroys the comparison. [MultiHopRAG](https://huggingface.co/datasets/yixuantt/MultiHopRAG),
  [BEIR SciFact](https://huggingface.co/datasets/BeIR/scifact),
  [BEIR NFCorpus](https://huggingface.co/datasets/BeIR/nfcorpus).
- **Two indexes, one chunk list** — Chroma (dense, cosine) and bm25s (lexical) built from the same
  list in the same order, so a bm25s positional result and a Chroma `chunk_id` name the same text.
  Drift between them is the classic way a hybrid retriever produces confident nonsense, so a test
  re-checks the `chunk_id → doc_id` mapping in both stores.
- **Fusion** — Reciprocal Rank Fusion, `score(d) = Σ 1 / (RRF_K + rank)`. Rank-based, so BM25's
  unbounded scores and cosine's `[-1, 1]` never need normalising onto a common scale.
- **Scoring granularity** — the datasets label documents, not chunks, so chunk hits collapse to
  deduped documents (best rank wins) and all metrics are document-level.

```
src/config.py  constants   src/ingest.py    build both indexes   src/generate.py  context-only answering
src/data.py    load+chunk  src/retrieve.py  3 arms, RRF, collapse src/evaluate.py  metrics + significance
```

## Not built (and when to add it)

- **Cross-encoder reranking** — when `recall@20` is strong but `nDCG@10` is weak. MultiHopRAG is
  exactly that shape now.
- **Query rewriting / HyDE** — only if multi-hop recall stays poor after sweeping `RRF_K`.
- **Synthetic QA generation / RAGAS faithfulness** — both need an LLM judge or LLM-written labels.
  Every corpus here ships human labels, and the deterministic metrics cover the same ground.
- **nomic task prefixes** — tested, not adopted. Re-embedding SciFact with `search_document:` /
  `search_query:` moved nDCG@10 by +0.0005.
- **Batching retrieval before generation** — tested, not adopted. 34.0 s/query against 34.8 s;
  there was no model swap to avoid.
