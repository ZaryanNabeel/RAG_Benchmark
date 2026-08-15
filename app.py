"""Streamlit front-end over the built indexes.

    streamlit run app.py

Lives at the repo root, not in src/, because `streamlit run` executes the file
as a script -- package-relative imports (`from . import config`) do not work
under it.

Retrieval-only is the default: it answers in ~1 s, while generation is 12-34 s
per query on a 4 GB card because ollama swaps the embed and chat models in and
out. Tick "generate an answer" when you want the LLM in the loop.
"""
import time
from urllib.parse import urlsplit

import streamlit as st

from src import config
from src.data import BEIR, load_queries
from src.retrieve import ARMS, Index, search

st.set_page_config(page_title="RAG Benchmark", page_icon="🔍", layout="wide")


@st.cache_resource(show_spinner="loading index...")
def get_index(corpus):
    """Cached across reruns -- Streamlit replays the whole script on every
    widget change, and reloading Chroma + bm25s each keystroke is seconds."""
    return Index.load(corpus)


@st.cache_resource(show_spinner="loading labels...")
def get_gold(corpus):
    """query text -> {doc_id: relevance}, for the queries the dataset labels.

    A typed question has no gold docs, so the compare panel can only show what
    each arm returned, never who was right. Picking a labelled query is what
    turns that panel into a scoreable comparison."""
    return {q["query"]: q["gold"] for q in load_queries(corpus) if q["gold"]}


@st.cache_resource(show_spinner=False)
def first_chunk_of(corpus, _index):
    """doc_id -> its lowest chunk_id, i.e. the chunk holding the title.

    to_chunks() appends chunks in document order, so the first chunk_id seen
    for a document is chunk 0 of it. A *retrieved* chunk can be from the middle
    of an article and start mid-sentence, which makes a useless label."""
    out = {}
    for cid, did in _index.doc_of.items():
        out.setdefault(did, cid)
    return out


def get_titles(index, doc_ids):
    """doc_id -> title line, for the ~30 documents on screen.

    A raw doc_id is a bare number in BEIR and an article URL in MultiHopRAG.
    Neither says what the document is, which is the entire question when the
    three arms return different ones."""
    first = first_chunk_of(index.name, index)
    ids = [first[d] for d in doc_ids if d in first]
    if not ids:
        return {}
    got = index.coll.get(ids=ids, include=["documents"])
    out = {}
    for cid, text in zip(got["ids"], got["documents"]):
        # BEIR stores "title\n\ntext"; a missing title leaves a blank line.
        line = next((l for l in text.split("\n") if l.strip()), "")
        out[index.doc_of[cid]] = line.strip()
    return out


# Three arms x 10 documents is up to 30 distinct rows when they disagree; the
# tail of that is all rank-10 near-misses, so the table stops before it.
SHOWN_DOCS = 15

# The rank columns need almost no width, so the document column gets the rest.
# Long enough for a full SciFact abstract title, which is the longest kind here.
TITLE_CHARS = 130


def source_of(doc_id):
    """MultiHopRAG doc_ids are article URLs -- the outlet is the readable part
    and the one the queries actually name."""
    if doc_id.startswith("http"):
        return urlsplit(doc_id).netloc.removeprefix("www.")
    return doc_id


picked = None
with st.sidebar:
    st.header("retrieval")
    corpus = st.selectbox("corpus", config.CORPORA)
    with st.expander("labelled queries"):
        st.caption("only dataset queries carry human relevance judgments — "
                   "pick one to score the arms instead of just listing them")
        pick = st.selectbox("query", list(get_gold(corpus)), index=None,
                            placeholder="search or scroll...",
                            label_visibility="collapsed")
        if st.button("run it", disabled=not pick, use_container_width=True):
            picked = pick
    arm = st.radio("arm", list(ARMS), index=list(ARMS).index("hybrid"),
                   horizontal=True)
    top_k = st.slider("top_k chunks", 1, 20, config.TOP_K)
    # RRF_K belongs next to top_k, but whether it is greyed out depends on
    # `compare` further down -- so reserve the slot here and fill it below.
    rrf_slot = st.empty()

    st.divider()
    compare = st.checkbox("compare all three arms", value=True,
                          help="document-level scoreboard for all three arms; "
                               "`arm` above still picks which one is read in "
                               "full and answered from")
    do_gen = st.checkbox("generate an answer", value=False,
                         help=f"{config.GEN_MODEL} via ollama, 12-34 s/query. "
                              f"Only multihoprag ships gold answers; the BEIR "
                              f"corpora are claim-verification sets with "
                              f"nothing to extract.")

    # The compare panel always runs hybrid, so RRF_K is live whenever it is on,
    # whatever `arm` is set to. Greying it out on `arm` alone claimed otherwise.
    rrf_k = rrf_slot.slider("RRF_K", 1, 200, config.RRF_K,
                            disabled=arm != "hybrid" and not compare,
                            help="lower = a lone rank-1 hit outweighs "
                                 "agreement between arms")
    st.divider()
    if st.button("clear chat"):
        st.session_state.turns = []
    st.caption(f"embed: `{config.EMBED_MODEL}`\n\ngen: `{config.GEN_MODEL}`")

# ponytail: retrieve.hybrid reads config.RRF_K when its rrf_k arg is None, and
# search() has no rrf_k parameter -- setting the module constant is the whole
# slider. Thread a real argument through search()/answer() if anything else
# ever needs to vary it per call.
config.RRF_K = rrf_k

index = get_index(corpus)
# Deliberately arm-neutral: a fixed "Hybrid" in the heading reads as a claim
# about which arm is running. The live arm is in the caption below instead.
st.title("RAG Retrieval Benchmark")
st.caption(f"{corpus} · {len(index.chunk_ids):,} chunks · "
           f"{len(set(index.doc_of.values())):,} documents · answering from "
           f"**{arm}**" + (f" (RRF_K {rrf_k})" if arm == "hybrid" else "")
           + (" · comparing all three" if compare else ""))

st.session_state.setdefault("turns", [])


def render(turn):
    with st.chat_message("user"):
        st.write(turn["query"])
    with st.chat_message("assistant"):
        if turn.get("error"):
            st.error(turn["error"])
        elif turn.get("answer"):
            st.success(turn["answer"])
            if turn.get("gen_note"):
                st.caption(turn["gen_note"])

        if turn.get("compare"):
            gold = turn.get("gold") or {}
            scores = turn.get("scores") or {}
            titles = turn.get("titles") or {}
            panels = turn["compare"]
            arms = list(panels)
            # One row per document, its rank under each arm. Listing each
            # document once instead of three times is what removes the markers:
            # a blank cell already says "this arm missed it", so nothing has to
            # be flagged for it.
            rank = {a: {d: i for i, d in enumerate(docs, 1)}
                    for a, docs in panels.items()}
            order = sorted({d for docs in panels.values() for d in docs},
                           key=lambda d: min(rank[a].get(d, 99) for a in arms))
            # Ties go to whichever arm ARMS lists first; the numbers are on
            # screen, so the star is a pointer, not a verdict.
            best = max(scores, key=lambda a: scores[a]["ndcg@10"]) if gold else None

            head = " | ".join(a + (" ⭐" if a == best else "") for a in arms)
            rows = [f"| document | {head} |", "|---|" + ":--:|" * len(arms)]
            for d in order[:SHOWN_DOCS]:
                title = titles.get(d) or source_of(d)
                if len(title) > TITLE_CHARS:
                    title = title[:TITLE_CHARS - 1].rstrip() + "…"
                # A pipe in a title would split the cell and skew every column.
                title = title.replace("|", "\\|")
                src = source_of(d)
                # MultiHopRAG queries name outlets, so the domain earns its
                # space; a bare BEIR id does not.
                if src != title and d.startswith("http"):
                    title = f"{title} — {src}"
                cells = " | ".join(str(rank[a].get(d, "—")) for a in arms)
                rows.append(f"| {'✅ ' if d in gold else ''}{title} | {cells} |")
            for m in ("ndcg@10", "recall@10") if gold else ():
                top = max(scores[a][m] for a in arms)
                cells = " | ".join(
                    f"**{scores[a][m]:.3f}**" if scores[a][m] == top
                    else f"{scores[a][m]:.3f}" for a in arms)
                rows.append(f"| **{m}** | {cells} |")
            st.markdown("\n".join(rows))

            note = ["`—` = not in this arm's top 10"]
            if gold:
                note.insert(0, f"✅ = one of {len(gold)} documents humans "
                               f"judged relevant")
            if len(order) > SHOWN_DOCS:
                note.append(f"showing {SHOWN_DOCS} of {len(order)} documents")
            st.caption(" &nbsp;·&nbsp; ".join(note))
            if not gold:
                st.caption("No human labels for this query, so the arms can be "
                           "compared but not scored — pick a **labelled query** "
                           "in the sidebar for that.")

        hits = turn.get("hits") or []
        if hits:
            label = ("context sent to the model" if turn.get("answer")
                     else "chunk text from the selected arm")
            with st.expander(f"{label} — {len(hits)} chunks "
                             f"({turn['arm']}, {turn['ms']} ms)"):
                for i, (cid, doc_id, text) in enumerate(hits, 1):
                    st.markdown(f"**[{i}]** `{doc_id}`")
                    st.text(text)
                    st.divider()


for turn in st.session_state.turns:
    render(turn)

typed = st.chat_input("Ask the corpus...")
if query := (typed or picked):
    turn = {"query": query, "arm": arm}
    t0 = time.time()
    try:
        if compare:
            # Document-level, same collapse and the same chunk depth
            # eval_retrieval uses, so what shows here is what the metrics
            # scored -- fewer chunks would flatter recall@10 by hiding misses.
            from src.retrieve import ranked_docs
            from src.evaluate import score_one
            turn["gold"] = get_gold(corpus).get(query, {})
            turn["compare"], turn["scores"] = {}, {}
            for name in ARMS:
                docs = ranked_docs(index, query, name,
                                   top_k * config.EVAL_OVERFETCH, top=10)
                turn["compare"][name] = [d for d, _ in docs]
                turn["scores"][name] = score_one(turn["gold"], docs)
            # One Chroma get for the ~30 documents on screen, not one per row.
            turn["titles"] = get_titles(
                index, {d for docs in turn["compare"].values() for d in docs})
        if do_gen:
            from src.generate import answer          # late: needs ollama up
            turn["answer"], turn["hits"] = answer(index, query, arm=arm, k=top_k)
            if corpus in BEIR:
                # BEIR queries are claims to verify, not questions, and carry no
                # gold answer -- the prompt asks for a short extracted span, so
                # a refusal here is the correct output, not a failure.
                turn["gen_note"] = (
                    f"`{corpus}` is a claim-verification set: its queries are "
                    f"statements to check, not questions, and none of them has "
                    f"a gold answer. Expect refusals — generation is only "
                    f"meaningful on multihoprag.")
        else:
            turn["hits"] = search(index, query, arm=arm, k=top_k)
    except Exception as e:
        # ollama not running is the common one, and the traceback buries it.
        turn["error"] = f"{type(e).__name__}: {e}"
    turn["ms"] = int((time.time() - t0) * 1000)
    st.session_state.turns.append(turn)
    render(turn)
