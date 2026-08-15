"""Retrieval and generation evaluation.

Retrieval is the headline: three arms x published-baseline-comparable metrics.
SciFact BM25 nDCG@10 should land near 0.65 (BEIR reports 65.1) -- that number
is the sanity anchor for the whole indexing path. If it is far off, the bug is
in the index, not in the retriever.

Generation uses SQuAD-style normalised EM/F1. MultiHopRAG answers are 2-25
characters, so exact match is meaningful and no LLM judge is involved.
"""
import argparse
import json
import random
import re
import string
from collections import Counter, defaultdict

from ranx import Qrels, Run, compare, evaluate as ranx_evaluate
from tqdm import tqdm

from . import config
from .data import load_queries
from .retrieve import ARMS, Index, ranked_docs

METRICS = ["ndcg@10", "recall@10", "recall@20", "mrr@10", "precision@1"]


# ---------------------------------------------------------------- answer text

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(s):
    """SQuAD normalisation: lowercase, drop punctuation/articles, squash space."""
    s = s.lower().translate(_PUNCT)
    return " ".join(_ARTICLES.sub(" ", s).split())


def exact_match(pred, gold):
    return float(normalize(pred) == normalize(gold))


def token_f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if not same:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


# ------------------------------------------------------------------ retrieval

def score_one(gold, docs, metrics=METRICS):
    """Metrics for a single query -> dict, or None when the query has no gold.

    Same ranx path as eval_retrieval below, so a number shown in the UI is the
    per-query number the report averages. `docs` is to_docs() output."""
    if not gold:
        return None
    run = dict(docs) or {"__none__": 0.0}
    scores = ranx_evaluate(Qrels({"q": dict(gold)}), Run({"q": run}), metrics)
    return {m: float(scores[m]) for m in metrics}


def eval_retrieval(index, queries, k=20, arms=tuple(ARMS)):
    """Document-level metrics for each arm. Queries with no gold docs
    (MultiHopRAG null_query) are excluded -- they have nothing to retrieve."""
    labelled = [q for q in queries if q["gold"]]
    # Graded where the dataset grades: nfcorpus labels 1-2, and nDCG discounts
    # a 1 below a 2. Flattening them to binary throws that away.
    qrels = Qrels({q["qid"]: dict(q["gold"]) for q in labelled})
    # Ask every arm for the same chunk depth. Chunks collapse to fewer
    # documents than chunks, so this has to overshoot k or recall@k is scored
    # against a list with fewer than k slots -- see config.EVAL_OVERFETCH.
    depth = k * config.EVAL_OVERFETCH

    out, runs = {}, []
    for arm in arms:
        run = {}
        for q in tqdm(labelled, desc=f"retrieval:{arm}"):
            docs = dict(ranked_docs(index, q["query"], arm, depth, top=k))
            # ranx cannot score an empty result list; a sentinel doc_id that no
            # qrels entry uses scores 0 and keeps the query in the average.
            run[q["qid"]] = docs or {"__none__": 0.0}
        r = Run(run, name=arm)
        runs.append(r)
        # ranx hands back np.float64, which json.dumps refuses.
        scores = ranx_evaluate(qrels, r, METRICS)
        out[arm] = {m: float(scores[m]) for m in METRICS}
    return out, len(labelled), significance(qrels, runs)


def significance(qrels, runs, metric="ndcg@10"):
    """Paired significance test between every pair of arms.

    "hybrid beats dense by 0.009" is a ranking of point estimates, not a
    result, until you know whether 323 queries can tell those two apart.
    Fisher's randomisation test is the IR standard for this: paired, exact,
    and it assumes nothing about the score distribution. Seeded, so the
    p-values reproduce like everything else here."""
    rep = compare(qrels, runs, [metric], stat_test="fisher",
                  n_permutations=1000, max_p=0.05, random_seed=0).to_dict()
    out = {}
    for a in rep["model_names"]:
        for b, per_metric in rep[a]["comparisons"].items():
            if a >= b:                      # each unordered pair once
                continue
            wtl = rep[a]["win_tie_loss"][b][metric]
            p = float(per_metric[metric])
            out[f"{a} vs {b}"] = {
                "metric": metric, "p": p, "significant_at_0.05": p < 0.05,
                "wins": wtl["W"], "ties": wtl["T"], "losses": wtl["L"]}
    return out


# ------------------------------------------------------------------ generation

def eval_generation(index, queries, arm="hybrid", k=None, cache=None):
    """`cache` is a JSONL path written one row at a time, so an interrupted run
    resumes instead of restarting. 300 local generations is ~3 hours on a 4GB
    card -- losing that to a closed shell is the expensive failure here.

    The cache is keyed on qid ONLY, so it is valid only for the same arm, k,
    and GEN_MODEL. Those three are encoded in the caller's filename so a
    changed setting starts a new cache instead of silently resuming another
    model's answers -- see main().
    """
    from .generate import generate_from    # imported late: retrieval-only runs
    from .retrieve import search           # should not need a loaded LLM
    done = {}
    if cache and cache.exists():
        done = {r["qid"]: r for r in
                (json.loads(l) for l in cache.read_text().splitlines() if l.strip())}
        print(f"resuming: {len(done)} cached generations in {cache.name}")

    # Do NOT split this into a retrieve-everything-then-generate-everything
    # pass to dodge model swapping. Measured on a 4 GB card: ollama reports
    # load_duration ~0.4 s on every call, so nothing is being reloaded -- both
    # models stay resident because qwen3:8b spills to CPU rather than evicting.
    # The 34.8 s/query is prompt PREFILL at ~17 ms/token (61% of layers on
    # CPU), and batching does not change prompt size. A two-phase version was
    # built and timed at 34.0 s/query, identical. The knobs that do move it are
    # TOP_K (prefill is linear in it: k=10 -> 43 s, k=5 -> 20 s, k=3 -> 11 s)
    # and a generator small enough to fit in VRAM.
    rows = []
    fh = cache.open("a", encoding="utf-8") if cache else None
    try:
        for q in tqdm(queries, desc=f"generate:{arm}"):
            if q["qid"] in done:
                rows.append(done[q["qid"]])
                continue
            hits = search(index, q["query"], arm=arm, k=k)
            pred = generate_from(q["query"], hits)
            row = {
                "qid": q["qid"], "qtype": q["qtype"], "query": q["query"],
                "gold": q["answer"], "pred": pred,
                "em": exact_match(pred, q["answer"]),
                "f1": token_f1(pred, q["answer"]),
                "ctx_hit": float(bool(set(q["gold"]) & {d for _, d, _ in hits})),
            }
            rows.append(row)
            if fh:
                fh.write(json.dumps(row) + "\n")
                fh.flush()          # survive a kill, not just a clean exit
    finally:
        if fh:
            fh.close()

    by_type = defaultdict(list)
    for r in rows:
        by_type[r["qtype"]].append(r)

    def agg(rs):
        n = len(rs)
        return {"n": n,
                "em": sum(r["em"] for r in rs) / n,
                "f1": sum(r["f1"] for r in rs) / n,
                "context_recall": sum(r["ctx_hit"] for r in rs) / n}

    summary = {"overall": agg(rows)}
    summary.update({t: agg(rs) for t, rs in sorted(by_type.items())})
    # null_query EM is the refusal rate: did the model decline when it should?
    if "null_query" in by_type:
        summary["refusal_rate_on_unanswerable"] = summary["null_query"]["em"]
    return summary, rows


# ------------------------------------------------------------------------ cli

def _table(results):
    cols = METRICS
    w = max(len(a) for a in results) + 2
    print("\n" + "arm".ljust(w) + "".join(c.rjust(12) for c in cols))
    print("-" * (w + 12 * len(cols)))
    for arm, m in results.items():
        print(arm.ljust(w) + "".join(f"{m[c]:12.4f}" for c in cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=config.CORPORA, required=True)
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--limit", type=int, help="seeded random sample of N queries")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--gen-arm", choices=list(ARMS), default="hybrid")
    a = ap.parse_args()

    index = Index.load(a.corpus)
    queries = load_queries(a.corpus)
    sampled = bool(a.limit) and a.limit < len(queries)
    if sampled:
        # Random, not the first N: MultiHopRAG is grouped by question_type, so
        # a head slice would report one type and call it the overall score.
        queries = random.Random(a.seed).sample(queries, a.limit)
    print(f"{a.corpus}: {len(queries)} queries, {len(index.chunk_ids)} chunks")

    report = {"corpus": a.corpus, "n_queries": len(queries),
              "embed_model": config.EMBED_MODEL, "gen_model": config.GEN_MODEL,
              "chunk_chars": config.CHUNK_CHARS, "rrf_k": config.RRF_K}

    retrieval, n_labelled, sig = eval_retrieval(index, queries, k=a.k)
    report["retrieval"] = retrieval
    report["n_labelled"] = n_labelled
    report["significance"] = sig
    _table(retrieval)
    print("\npaired significance on ndcg@10 (Fisher randomisation, n=1000):")
    for pair, s in sig.items():
        verdict = "significant" if s["significant_at_0.05"] else "NOT significant"
        print(f"  {pair:<20} p={s['p']:.4f}  {verdict:<15} "
              f"W/T/L {s['wins']}/{s['ties']}/{s['losses']}")

    if not a.retrieval_only and any(q["answer"] for q in queries):
        config.RESULTS_DIR.mkdir(exist_ok=True)
        tag = f"_sample{a.limit}" if sampled else ""
        summary, rows = eval_generation(
            index, queries, arm=a.gen_arm,
            # Model and k are in the filename, not just the comment: the cache
            # is keyed on qid alone, so a run with a different generator or
            # context size would otherwise resume from the old one and report
            # its answers as the new model's. ':' is illegal in Windows paths.
            cache=config.RESULTS_DIR / (
                f"{a.corpus}{tag}_{a.gen_arm}"
                f"_{config.GEN_MODEL.replace(':', '-')}"
                f"_k{config.TOP_K}_gen.jsonl"))
        report["generation"] = summary
        print(f"\ngeneration ({a.gen_arm}):")
        for name, m in summary.items():
            if isinstance(m, dict):
                print(f"  {name:<20} n={m['n']:<5} em={m['em']:.3f} "
                      f"f1={m['f1']:.3f} ctx_recall={m['context_recall']:.3f}")
        (config.RESULTS_DIR / f"{a.corpus}{tag}_predictions.json").write_text(
            json.dumps(rows, indent=2))

    config.RESULTS_DIR.mkdir(exist_ok=True)
    # A --limit run samples the query set, so its numbers are not the full-corpus
    # ones -- separate file, or the sampled run silently overwrites the real table.
    suffix = f"_sample{a.limit}" if sampled else ""
    path = config.RESULTS_DIR / f"{a.corpus}{suffix}_report.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
