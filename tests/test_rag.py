"""Smallest checks that fail if the non-obvious logic breaks.

Index-alignment tests skip when nothing has been ingested yet, so `pytest`
is useful before the first (slow) embedding run.
"""
import pytest

from src import config
from src.data import chunk, to_chunks
from src.evaluate import exact_match, token_f1
from src.retrieve import rrf, to_docs


def test_rrf_is_rank_based_not_score_based():
    # 'a' and 'z' each top exactly one arm; 'b' is rank 2 in both.
    dense = ["a", "b", "c"]
    lex = ["z", "b", "y"]

    scores = dict(rrf([dense, lex], k=60))
    assert scores["b"] == pytest.approx(2 / 62)      # agreement across arms
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["c"] == pytest.approx(1 / 63)
    order = [cid for cid, _ in rrf([dense, lex], k=60)]
    assert order[0] == "b", order      # consensus at rank 2 beats a lone rank 1
    assert set(order) == {"a", "b", "c", "y", "z"}

    # k is a real knob: shrink it and a lone rank-1 hit is worth as much as
    # rank-2 agreement (a and b tie at 1.0, alphabetical tiebreak picks 'a').
    assert rrf([dense, lex], k=0)[0][0] == "a"


def test_rrf_ranks_are_one_based():
    # A single arm must reproduce its own order, with no zero-division at k=0.
    assert [c for c, _ in rrf([["x", "y"]], k=0)] == ["x", "y"]


def test_chunk_respects_size_and_covers_text():
    text = "\n".join(f"para {i} " + "word " * 40 for i in range(20))
    out = chunk(text, 500, 100)
    assert len(out) > 1
    assert all(len(c) <= 500 for c in out)
    assert "para 19" in out[-1]


def test_chunk_hard_splits_oversized_paragraph():
    out = chunk("x" * 2500, 500, 100)
    assert all(len(c) <= 500 for c in out)
    assert len(out) >= 5


def test_chunk_ids_are_unique_and_map_to_docs():
    docs = [{"doc_id": "d1", "text": "a " * 900}, {"doc_id": "d2", "text": "short"}]
    rows = to_chunks(docs, 200, 50)
    ids = [r["chunk_id"] for r in rows]
    assert len(ids) == len(set(ids))
    assert {r["doc_id"] for r in rows} == {"d1", "d2"}


def test_to_docs_dedupes_and_keeps_best_rank():
    class FakeIndex:
        doc_of = {"0": "dA", "1": "dB", "2": "dA", "3": "dC"}

    got = to_docs(FakeIndex(), ["0", "1", "2", "3"])
    assert [d for d, _ in got] == ["dA", "dB", "dC"]
    scores = [s for _, s in got]
    assert scores == sorted(scores, reverse=True)


def test_every_arm_searches_the_same_depth(monkeypatch):
    """The bug this exists for: hybrid() overfetches internally, so a caller
    that pre-scaled its depth had hybrid searching depth*OVERFETCH while dense
    and bm25 saw depth. The arms were then compared on unequal candidate pools,
    which cost hybrid 0.005-0.007 recall@20."""
    import src.retrieve as R

    asked = []

    def spy(name):
        def fn(index, query, k, *a, **kw):
            asked.append((name, k))
            return []
        return fn

    monkeypatch.setattr(R, "dense", spy("dense"))
    monkeypatch.setattr(R, "lexical", spy("lexical"))
    monkeypatch.setattr(R, "ARMS", {"dense": R.dense, "bm25": R.lexical,
                                    "hybrid": R.hybrid})

    class FakeIndex:
        doc_of = {}

    for arm in ("dense", "bm25", "hybrid"):
        R.ranked_docs(FakeIndex(), "q", arm, depth=60)
    assert {k for _, k in asked} == {60}, asked


def test_answer_metrics():
    assert exact_match("Sam Bankman-Fried", "Caroline Ellison") == 0.0
    assert exact_match(" Sam Bankman Fried ", "the sam bankman fried") == 1.0
    assert exact_match("the FTX exchange", "FTX exchange.") == 1.0
    assert token_f1("sam fried", "sam bankman fried") == pytest.approx(0.8)
    assert token_f1("nothing", "sam") == 0.0


def test_generation_cache_resumes_without_regenerating(tmp_path, monkeypatch):
    """A 300-query generation run is ~3 hours; an interrupted one must not
    start over. Second pass must answer only what the cache is missing."""
    import src.generate
    import src.retrieve
    from src.evaluate import eval_generation

    calls, retrieved = [], []

    def fake_search(index, query, arm="hybrid", k=None):
        retrieved.append(query)
        return [("0", "dA", "ctx")]

    def fake_generate(query, hits):
        calls.append(query)
        return "Sam Bankman-Fried"

    monkeypatch.setattr(src.retrieve, "search", fake_search)
    monkeypatch.setattr(src.generate, "generate_from", fake_generate)
    qs = [{"qid": str(i), "query": f"q{i}", "qtype": "t",
           "answer": "Sam Bankman-Fried", "gold": {"dA": 1}} for i in range(3)]
    cache = tmp_path / "gen.jsonl"

    s1, r1 = eval_generation(None, qs, cache=cache)
    assert len(calls) == 3 and s1["overall"]["em"] == 1.0

    # Interrupted after 2: drop the last cached row, rerun, only it regenerates.
    cache.write_text("\n".join(cache.read_text().splitlines()[:2]) + "\n")
    s2, r2 = eval_generation(None, qs, cache=cache)
    assert len(calls) == 4, calls          # 3 + exactly one redo
    assert r2 == r1 and s2 == s1           # resumed run scores identically
    # Retrieval is skipped for cached queries too -- a resumed run must not pay
    # retrieval cost for answers it already has.
    assert len(retrieved) == 4, retrieved


# ------------------------------------------------------- needs a built index

def _index(name):
    from src.retrieve import Index
    if not (config.BM25_DIR / name / "meta.json").exists():
        pytest.skip(f"{name} not ingested yet")
    return Index.load(name)


@pytest.mark.parametrize("name", config.CORPORA)
def test_indexes_are_aligned(name):
    """The bug this exists for: Chroma and bm25s drifting out of order, which
    makes hybrid silently fuse unrelated chunks."""
    idx = _index(name)
    assert idx.coll.count() == len(idx.chunk_ids)
    for cid in (idx.chunk_ids[0], idx.chunk_ids[len(idx.chunk_ids) // 2],
                idx.chunk_ids[-1]):
        meta = idx.coll.get(ids=[cid], include=["metadatas"])["metadatas"][0]
        assert meta["doc_id"] == idx.doc_of[cid], cid
