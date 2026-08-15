"""All tunable constants. Nothing else lives here."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"
BM25_DIR = DATA_DIR / "bm25"
RESULTS_DIR = ROOT / "results"

# Keep HF downloads inside the project instead of %USERPROFILE%\.cache.
os.environ.setdefault("HF_HOME", str(DATA_DIR / "hf"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

OLLAMA_HOST = "http://localhost:11434"

# Swap with EMBED_MODEL=... to re-run the whole evaluation on another embedder.
# Measured on a Quadro T1000 (4GB) over 1488-char SciFact abstracts:
#   bge-m3            1097 ms/doc   (1024d, best quality, 95 min for SciFact)
#   nomic-embed-text   379 ms/doc   ( 768d, default -- 33 min for SciFact)
#   all-minilm          31 ms/doc   ( 384d, but truncates at 256 tokens, so it
#                                     silently drops half of every abstract)
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
GEN_MODEL = os.environ.get("GEN_MODEL", "qwen3:8b")
EMBED_BATCH = 32

# Chunking, in characters (~4 chars/token). Only long-form corpora are chunked;
# SciFact abstracts are already passage-sized and stay whole.
CHUNK_CHARS = 1000
CHUNK_OVERLAP = 150

# Retrieval knobs -- these are what the evaluation sweeps.
TOP_K = 10
RRF_K = 60          # Cormack et al. default; lower = more weight on rank-1 hits
OVERFETCH = 3       # hybrid fetches k * OVERFETCH chunks per arm before fusing

# Evaluation fetches k * EVAL_OVERFETCH chunks per arm, because chunks collapse
# to fewer documents than chunks and the metrics are document-level. At 3, 28%
# of MultiHopRAG queries yielded fewer than 20 distinct documents, so recall@20
# was scored against a list that had no 20th slot -- it understated bm25 by
# 0.014. Measured to saturate at 6; higher changes nothing.
EVAL_OVERFETCH = 6

# MultiHopRAG's gold answer for its deliberately-unanswerable queries. The
# generator prompt reuses this string so refusal is measurable with plain EM.
NO_ANSWER = "Insufficient information."

CORPORA = ("scifact", "nfcorpus", "multihoprag")
