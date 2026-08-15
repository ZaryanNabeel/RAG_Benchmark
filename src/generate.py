"""Answer generation over retrieved context.

The refusal instruction reuses config.NO_ANSWER verbatim: MultiHopRAG's
null_query rows carry exactly that string as their gold answer, so refusal
behaviour falls out of the same exact-match metric as everything else -- no
LLM judge needed to measure hallucination on unanswerable questions.
"""
import argparse

import ollama

from . import config
from .retrieve import Index, search

_client = ollama.Client(host=config.OLLAMA_HOST)

SYSTEM = (
    "You answer strictly from the provided context passages.\n"
    "Answer with the shortest possible span -- a name, date, number or phrase. "
    "No explanation, no full sentence, no citation markers.\n"
    f'If the context does not contain the answer, reply exactly: {config.NO_ANSWER}'
)


def build_prompt(query, hits):
    ctx = "\n\n".join(f"[{i}] {text}" for i, (_, _, text) in enumerate(hits, 1))
    return f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:"


def generate_from(query, hits):
    """Answer from context that has already been retrieved.

    Kept separate from answer() so the retrieval and generation costs can be
    timed apart -- which is how the swapping theory got disproved. On a 4 GB
    card this call is ~35 s, and ollama attributes ~33 s of it to prompt
    prefill and 0.4 s to model loading. Prompt size is the lever, not call
    ordering; see the note in evaluate.eval_generation."""
    r = _client.chat(
        model=config.GEN_MODEL,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": build_prompt(query, hits)}],
        think=False,            # qwen3 emits <think> blocks otherwise
        options={"temperature": 0.0},
    )
    return r["message"]["content"].strip()


def answer(index, query, arm="hybrid", k=None):
    """Retrieve then generate, one query. Interactive path -- the app and the
    CLI use this; batch evaluation splits the two phases instead."""
    hits = search(index, query, arm=arm, k=k)
    return generate_from(query, hits), hits


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=config.CORPORA, default="multihoprag")
    ap.add_argument("--arm", choices=["dense", "bm25", "hybrid"], default="hybrid")
    ap.add_argument("query")
    a = ap.parse_args()
    idx = Index.load(a.corpus)
    text, hits = answer(idx, a.query, arm=a.arm)
    print(f"\n{text}\n")
    for cid, doc, t in hits:
        print(f"  - {doc}  [{cid}] {t[:90]}...")
