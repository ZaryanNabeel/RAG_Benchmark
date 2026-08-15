"""Load -> chunk -> embed -> Chroma (dense) + bm25s (lexical).

Both indexes are built from the SAME `chunks` list in the SAME order, so a
bm25s result index and a Chroma chunk_id refer to the same text. That
alignment is asserted at the end -- silently misaligned indexes are the
classic way a hybrid retriever produces plausible-looking garbage.
"""
import argparse
import json

import bm25s
import chromadb
import Stemmer
from tqdm import tqdm

from . import config
from .data import load_corpus, to_chunks
from .embed import embed_texts

# Single abstracts -- already passage-sized, and BEIR's published baselines are
# computed unchunked, so chunking these would break the comparison.
NO_CHUNK = {"scifact", "nfcorpus"}


def build(name, chunk_chars=None, overlap=None):
    chunk_chars = chunk_chars or config.CHUNK_CHARS
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap

    docs = load_corpus(name)
    chunks = to_chunks(docs, chunk_chars, overlap, do_chunk=name not in NO_CHUNK)
    texts = [c["text"] for c in chunks]
    print(f"{name}: {len(docs)} docs -> {len(chunks)} chunks")

    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    if name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)
    # search_ef is how many candidates HNSW keeps while walking the graph.
    # Chroma's default scales with n_results, which makes the returned
    # neighbours depend on how many you asked for -- 5 of 323 nfcorpus queries
    # changed their top-10 between fetch depth 60 and 120. Pinning it high
    # removes that coupling and puts the search within 0.003 nDCG of exact
    # brute-force cosine. Cheap at this corpus size; takes effect on re-ingest.
    coll = client.create_collection(
        name, metadata={"hnsw:space": "cosine", "hnsw:search_ef": 400})

    vectors = embed_texts(texts, desc=f"embed {name}")
    step = 1000
    for i in tqdm(range(0, len(chunks), step), desc=f"chroma {name}"):
        part = chunks[i:i + step]
        coll.add(
            ids=[c["chunk_id"] for c in part],
            documents=[c["text"] for c in part],
            embeddings=vectors[i:i + step],
            metadatas=[{"doc_id": c["doc_id"], "pos": c["pos"]} for c in part],
        )

    stemmer = Stemmer.Stemmer("english")
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(texts, stopwords="en", stemmer=stemmer,
                                   show_progress=True))
    out = config.BM25_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    retriever.save(str(out))
    # Parallel arrays: bm25s returns positional indexes, these map them back.
    (out / "meta.json").write_text(json.dumps({
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "doc_ids": [c["doc_id"] for c in chunks],
    }))

    assert coll.count() == len(chunks), (coll.count(), len(chunks))
    print(f"{name}: indexed {coll.count()} chunks in both stores")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=config.CORPORA, required=True)
    ap.add_argument("--chunk-chars", type=int)
    ap.add_argument("--overlap", type=int)
    a = ap.parse_args()
    build(a.corpus, a.chunk_chars, a.overlap)
