"""Ollama embeddings. Batched because a 15k-chunk corpus one-at-a-time is
minutes of pure HTTP overhead."""
from functools import lru_cache

import ollama
from tqdm import tqdm

from . import config

_client = ollama.Client(host=config.OLLAMA_HOST)


def embed_texts(texts, desc="embed", batch=None):
    batch = batch or config.EMBED_BATCH
    out = []
    for i in tqdm(range(0, len(texts), batch), desc=desc):
        r = _client.embed(model=config.EMBED_MODEL, input=texts[i:i + batch])
        out.extend(r["embeddings"])
    return out


@lru_cache(maxsize=8192)
def embed_query(text):
    # Cached because the dense and hybrid arms embed the same query during a
    # single evaluation sweep.
    return _client.embed(model=config.EMBED_MODEL, input=[text])["embeddings"][0]
