"""Dataset loading. Field names verified against the actual HF records.

multihoprag corpus : category, author, published_at, body, title, url, source
multihoprag queries: query, answer, question_type, evidence_list[{url, fact, ...}]
                     609 docs / 2556 queries; 301 are null_query with empty
                     evidence and answer == "Insufficient information."

Every BEIR dataset ships the same three tables, so they share one loader:
  <name> corpus     : _id, title, text
  <name> queries    : _id, title, text     (text is the claim/question)
  <name>-qrels test : query-id, corpus-id, score
ids are ints in scifact and strings in nfcorpus ("PLAIN-2" / "MED-2427"), so
everything is str()-ed. scifact scores are all 1; nfcorpus grades 1-2 and
includes explicit 0 rows, which are non-relevant and dropped.
"""
from collections import defaultdict

from . import config  # noqa: F401  -- sets HF_HOME before datasets imports
from datasets import load_dataset

BEIR = ("scifact", "nfcorpus")


def load_corpus(name):
    """-> list of {doc_id, text}. doc_id is what retrieval is scored against."""
    if name in BEIR:
        ds = load_dataset(f"BeIR/{name}", "corpus")["corpus"]
        return [
            {"doc_id": r["_id"], "text": f"{r['title']}\n\n{r['text']}".strip()}
            for r in ds
        ]
    if name == "multihoprag":
        ds = load_dataset("yixuantt/MultiHopRAG", "corpus")["train"]
        return [
            {"doc_id": r["url"], "text": f"{r['title']}\n\n{r['body']}".strip()}
            for r in ds
        ]
    raise ValueError(f"unknown corpus {name!r}")


def load_queries(name):
    """-> list of {qid, query, gold ({doc_id: relevance}), answer, qtype}.

    `gold` is graded, not binary: nfcorpus labels at 1 or 2 and nDCG is meant
    to use the difference. MultiHopRAG evidence is unweighted, so it grades 1.
    `gold` is empty for MultiHopRAG null_query; those are excluded from
    retrieval metrics and used only for the refusal-rate check.
    """
    if name in BEIR:
        qrels = load_dataset(f"BeIR/{name}-qrels", split="test")
        gold = defaultdict(dict)
        for r in qrels:
            # nfcorpus ships explicit score-0 rows: judged and not relevant.
            if int(r["score"]) > 0:
                gold[str(r["query-id"])][str(r["corpus-id"])] = int(r["score"])
        text = {
            r["_id"]: r["text"]
            for r in load_dataset(f"BeIR/{name}", "queries")["queries"]
        }
        return [
            {"qid": qid, "query": text[qid], "gold": g, "answer": None,
             "qtype": "query"}
            for qid, g in gold.items()
        ]

    if name == "multihoprag":
        ds = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG")["train"]
        return [
            {
                "qid": str(i),
                "query": r["query"],
                "gold": {e["url"]: 1 for e in r["evidence_list"]},
                "answer": r["answer"],
                "qtype": r["question_type"],
            }
            for i, r in enumerate(ds)
        ]
    raise ValueError(f"unknown corpus {name!r}")


def chunk(text, size, overlap):
    """Paragraph-aware greedy packer, every chunk <= `size` chars.

    Paragraphs are packed until the next one would overflow; the new chunk then
    carries `overlap` trailing chars of the previous one so a fact split across
    the seam is still retrievable. A paragraph longer than `size` on its own is
    hard-sliced with the same overlap."""
    assert 0 <= overlap < size, "overlap must be smaller than size or slicing loops"
    units = []
    for p in (p.strip() for p in text.split("\n")):
        while len(p) > size:
            units.append(p[:size])
            p = p[size - overlap:]
        if p:
            units.append(p)

    out, buf = [], ""
    for u in units:
        if not buf:
            buf = u
        elif len(buf) + 1 + len(u) <= size:
            buf = f"{buf}\n{u}"
        else:
            out.append(buf)
            tail = buf[-overlap:] if overlap else ""
            buf = f"{tail}\n{u}" if tail and len(tail) + 1 + len(u) <= size else u
    if buf:
        out.append(buf)
    return out or [text[:size]]


def to_chunks(docs, size, overlap, do_chunk=True):
    """-> list of {chunk_id, doc_id, text}. chunk_id is positional and is the
    shared key between the Chroma collection and the BM25 index."""
    rows = []
    for d in docs:
        pieces = chunk(d["text"], size, overlap) if do_chunk else [d["text"]]
        for j, t in enumerate(pieces):
            rows.append({"chunk_id": str(len(rows)), "doc_id": d["doc_id"],
                         "text": t, "pos": j})
    return rows
