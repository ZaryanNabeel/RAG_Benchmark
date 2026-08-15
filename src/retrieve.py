"""Three retrieval arms over one shared chunk store.

Every arm returns an ordered list of chunk_ids, so fusion and scoring never
have to care which arm produced what. RRF fuses on *rank*, not score, which is
the whole reason it is the right choice here: BM25 scores are unbounded
term-weight sums and cosine similarities live in [-1, 1], so any score-level
combination needs a normalisation step that RRF simply does not have.
"""
import json
from dataclasses import dataclass

import bm25s
import chromadb
import Stemmer

from . import config
from .embed import embed_query

_stemmer = Stemmer.Stemmer("english")


@dataclass
class Index:
    name: str
    coll: object
    bm25: object
    chunk_ids: list
    doc_of: dict          # chunk_id -> doc_id

    @classmethod
    def load(cls, name):
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        coll = client.get_collection(name)
        d = config.BM25_DIR / name
        bm25 = bm25s.BM25.load(str(d), mmap=False)
        meta = json.loads((d / "meta.json").read_text())
        chunk_ids, doc_ids = meta["chunk_ids"], meta["doc_ids"]
        assert coll.count() == len(chunk_ids), "chroma/bm25 index size mismatch"
        return cls(name, coll, bm25, chunk_ids, dict(zip(chunk_ids, doc_ids)))


def dense(index, query, k):
    v = embed_query(query)
    k = min(k, len(index.chunk_ids))
    return index.coll.query(query_embeddings=[v], n_results=k)["ids"][0]


def lexical(index, query, k):
    k = min(k, len(index.chunk_ids))
    toks = bm25s.tokenize([query], stopwords="en", stemmer=_stemmer,
                          show_progress=False)
    if not any(len(t) for t in toks.ids):        # query was all stopwords
        return []
    try:
        # bm25s maps the query-local vocab back through the corpus vocab
        # (Tokenized -> string list -> vocab_dict), so ids cannot cross-wire.
        idx, _ = index.bm25.retrieve(toks, k=k, show_progress=False)
    except ValueError:
        return []                                # no query term is in the vocab
    return [index.chunk_ids[i] for i in idx[0]]


def rrf(rankings, k=None, top=None):
    """score(d) = sum over arms of 1 / (k + rank), rank 1-based.

    -> ordered list of (chunk_id, score), best first."""
    k = config.RRF_K if k is None else k
    scores = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    out = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return out[:top] if top else out


def hybrid(index, query, k, rrf_k=None, overfetch=None):
    n = k * (overfetch or config.OVERFETCH)
    fused = rrf([dense(index, query, n), lexical(index, query, n)], k=rrf_k)
    return [cid for cid, _ in fused[:k]]


ARMS = {"dense": dense, "bm25": lexical, "hybrid": hybrid}


def ranked_docs(index, query, arm, depth, top=None):
    """One arm -> deduped documents, every arm searched at the SAME depth.

    hybrid() overfetches internally, so a caller that has already scaled its
    depth would have hybrid searching depth*OVERFETCH while dense and bm25 see
    depth -- the arms would not be compared on equal footing. `overfetch=1`
    switches that second multiplication off. Measured cost of getting this
    wrong: hybrid recall@20 was 0.0070 (nfcorpus) and 0.0050 (scifact) lower,
    because a deeper pool lets more both-arms-agree documents accumulate above
    a gold document that only one arm found.

    Every retrieval comparison must go through here rather than calling ARMS
    directly, or the asymmetry comes straight back."""
    if arm == "hybrid":
        ids = hybrid(index, query, depth, overfetch=1)
    else:
        ids = ARMS[arm](index, query, depth)
    return to_docs(index, ids, top=top)


def to_docs(index, chunk_ids, top=None):
    """Collapse chunk hits to deduped documents, best-rank-wins.

    Both datasets label whole documents, not chunks, so retrieval is scored at
    document level: any chunk of a gold doc counts as a hit.
    -> ordered list of (doc_id, score) with score strictly descending."""
    seen, out = set(), []
    for rank, cid in enumerate(chunk_ids):
        d = index.doc_of[cid]
        if d in seen:
            continue
        seen.add(d)
        out.append((d, 1.0 / (rank + 1)))
    return out[:top] if top else out


def search(index, query, arm="hybrid", k=None):
    """Chunk-level results for generation: [(chunk_id, doc_id, text)]."""
    k = k or config.TOP_K
    ids = ARMS[arm](index, query, k)
    got = index.coll.get(ids=ids, include=["documents", "metadatas"])
    by_id = dict(zip(got["ids"], got["documents"]))
    return [(cid, index.doc_of[cid], by_id[cid]) for cid in ids if cid in by_id]
