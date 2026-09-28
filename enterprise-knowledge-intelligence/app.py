"""
Enterprise Knowledge Intelligence Platform
Claude-style chat UI with session memory + SQLite permanent history
"""

import os
import sqlite3
import datetime
import tempfile
import uuid

import streamlit as st

from src.rag_pipeline import RAGPipeline
from src.ingestion import load_document

# ─────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Knowledge Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# CLAUDE-STYLE CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* App background */
.stApp {
    background-color: #1a1a1a;
    color: #ececec;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111111;
    border-right: 1px solid #2a2a2a;
}
section[data-testid="stSidebar"] * {
    color: #c5c5c5 !important;
}

/* ── Chat messages ── */
.chat-wrapper {
    max-width: 780px;
    margin: 0 auto;
    padding-bottom: 120px;
}

.msg-row {
    display: flex;
    gap: 14px;
    margin-bottom: 28px;
    align-items: flex-start;
}

.msg-avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
    margin-top: 2px;
}

.avatar-user {
    background: #3d3d3d;
    color: #e0e0e0;
}

.avatar-ai {
    background: #c96442;
    color: #fff;
}

.msg-bubble {
    flex: 1;
    padding: 14px 18px;
    border-radius: 12px;
    line-height: 1.65;
    font-size: 0.93rem;
}

.bubble-user {
    background: #252525;
    color: #e8e8e8;
    border: 1px solid #333;
}

.bubble-ai {
    background: #1e1e1e;
    color: #ececec;
    border: 1px solid #2d2d2d;
}

.msg-meta {
    font-size: 0.72rem;
    color: #666;
    margin-top: 6px;
}

/* ── Source chips ── */
.source-header {
    font-size: 0.72rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 14px;
    margin-bottom: 6px;
}

.source-chip {
    display: inline-block;
    background: #2a2a2a;
    border: 1px solid #383838;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.73rem;
    color: #aaa;
    margin: 2px 4px 2px 0;
    cursor: default;
}

/* ── Input area ── */
.input-bar {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    z-index: 100;
    background: linear-gradient(to top, #1a1a1a 80%, transparent);
    padding: 18px 24px 24px;
}

/* ── Text input override ── */
.stTextInput input {
    background: #252525 !important;
    border: 1px solid #383838 !important;
    border-radius: 10px !important;
    color: #ececec !important;
    font-size: 0.95rem !important;
    padding: 14px 18px !important;
}
.stTextInput input:focus {
    border-color: #c96442 !important;
    box-shadow: 0 0 0 2px rgba(201,100,66,0.2) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #252525;
    border: 1px solid #383838;
    color: #c5c5c5;
    border-radius: 8px;
    font-size: 0.82rem;
    padding: 6px 14px;
    transition: all 0.15s;
}
.stButton > button:hover {
    background: #2e2e2e;
    border-color: #555;
    color: #fff;
}

/* ── Sidebar history items ── */
.hist-item {
    padding: 9px 12px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 0.83rem;
    color: #b0b0b0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 3px;
    transition: background 0.1s;
}
.hist-item:hover {
    background: #222;
}

/* ── Spinner / status ── */
.status-tag {
    display: inline-block;
    background: #1e2e1e;
    border: 1px solid #2d4d2d;
    color: #6db86d;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.78rem;
    margin-bottom: 8px;
}

/* ── Section titles ── */
.section-label {
    font-size: 0.72rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid #2a2a2a;
}

/* Hide streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# SQLITE  –  persistent chat history
# ─────────────────────────────────────────
DB_PATH = "data/chat_history.db"

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id   TEXT PRIMARY KEY,
            title        TEXT,
            created_at   TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id   TEXT,
            role         TEXT,
            content      TEXT,
            sources      TEXT,
            timestamp    TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    """)
    conn.commit()
    conn.close()

def create_session(session_id: str, title: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT OR IGNORE INTO sessions VALUES (?, ?, ?)",
        (session_id, title, datetime.datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def save_message(session_id: str, role: str, content: str, sources: str = ""):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO messages (session_id, role, content, sources, timestamp) VALUES (?,?,?,?,?)",
        (session_id, role, content, sources, datetime.datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def load_sessions():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    rows = c.execute(
        "SELECT session_id, title, created_at FROM sessions ORDER BY created_at DESC LIMIT 30"
    ).fetchall()
    conn.close()
    return rows

def load_session_messages(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    rows = c.execute(
        "SELECT role, content, sources, timestamp FROM messages WHERE session_id=? ORDER BY id",
        (session_id,)
    ).fetchall()
    conn.close()
    return rows

def delete_session(session_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
    c.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
    conn.commit()
    conn.close()

init_db()


# ─────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.session_title = "New Chat"
    create_session(st.session_state.session_id, "New Chat")

if "messages" not in st.session_state:       # temporary in-memory history
    st.session_state.messages = []

if "uploaded_files_processed" not in st.session_state:
    st.session_state.uploaded_files_processed = []

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
@st.cache_resource
def get_default_pipeline():
    return RAGPipeline()

def start_new_chat():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.session_title = "New Chat"
    st.session_state.messages = []
    create_session(st.session_state.session_id, "New Chat")
    st.session_state.pipeline = None
    st.session_state.uploaded_files_processed = []

def load_chat(session_id: str, title: str):
    st.session_state.session_id = session_id
    st.session_state.session_title = title
    rows = load_session_messages(session_id)
    st.session_state.messages = [
        {"role": r, "content": c, "sources": s, "timestamp": t}
        for r, c, s, t in rows
    ]
    st.session_state.pipeline = None  # will use default pipeline


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-label">Knowledge Intelligence</div>', unsafe_allow_html=True)

    if st.button("＋  New Chat", use_container_width=True):
        start_new_chat()
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── PDF Upload ──
    st.markdown('<div class="section-label">Upload Documents</div>', unsafe_allow_html=True)
    uploaded_pdfs = st.file_uploader(
        "Add PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_pdfs:
        new_files = [f.name for f in uploaded_pdfs
                     if f.name not in st.session_state.uploaded_files_processed]
        if new_files:
            with st.spinner("Processing PDFs…"):
                all_docs = []
                for pdf_file in uploaded_pdfs:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(pdf_file.read())
                        tmp_path = tmp.name
                    docs = load_document(tmp_path)
                    all_docs.extend(docs)
                    os.unlink(tmp_path)

                # Build a fresh pipeline from uploaded docs
                from src.chunking import split_documents
                from src.metadata import enrich_metadata
                from src.ids import generate_chunk_id
                from src.embeddings import create_embedding_model
                from src.vector_store import create_vector_store
                from src.m25_retriever import BM25Retriever
                from src.hybrid_retriever import HybridRetriever
                from src.reranking import Reranker
                from src.llm import create_llm
                from src.generation import generate_answer
                from src.context_builder import build_context

                chunks = split_documents(all_docs)
                chunks = enrich_metadata(chunks)
                for chunk in chunks:
                    chunk.metadata["chunk_id"] = generate_chunk_id(chunk)

                emb = create_embedding_model()
                vs  = create_vector_store(chunks, emb)
                bm25 = BM25Retriever(chunks)
                hybrid = HybridRetriever(vs, bm25)
                reranker = Reranker()
                llm = create_llm()

                # Attach to session state as a simple namespace
                class UploadedPipeline:
                    def ask(self, query):
                        candidates = hybrid.search(query, k=10)
                        reranked   = reranker.rerank(query, candidates, top_k=5)
                        documents  = [d for d, _ in reranked]
                        context    = build_context(documents)
                        answer     = generate_answer(llm, query, context)
                        return {"answer": answer, "documents": documents}

                st.session_state.pipeline = UploadedPipeline()
                st.session_state.uploaded_files_processed = [f.name for f in uploaded_pdfs]

            st.success(f"{len(uploaded_pdfs)} PDF(s) loaded ✓")

    if st.session_state.uploaded_files_processed:
        st.markdown('<div class="section-label" style="margin-top:10px">Active Documents</div>', unsafe_allow_html=True)
        for fn in st.session_state.uploaded_files_processed:
            st.markdown(f'<div class="hist-item">📄 {fn}</div>', unsafe_allow_html=True)

    # ── Chat History ──
    st.markdown('<div class="section-label" style="margin-top:20px">Recent Chats</div>', unsafe_allow_html=True)

    sessions = load_sessions()
    for sid, title, created_at in sessions:
        label = title if title != "New Chat" else "Chat " + created_at[11:16]
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(f"💬 {label[:28]}", key=f"sess_{sid}", use_container_width=True):
                load_chat(sid, title)
                st.rerun()
        with col2:
            if st.button("✕", key=f"del_{sid}"):
                delete_session(sid)
                if sid == st.session_state.session_id:
                    start_new_chat()
                st.rerun()


# ─────────────────────────────────────────
# MAIN CHAT AREA
# ─────────────────────────────────────────
# Header
st.markdown(f"""
<div style="text-align:center; padding: 30px 0 10px; max-width:780px; margin:0 auto;">
  <h2 style="color:#ececec; font-weight:500; font-size:1.5rem; margin:0;">
    {st.session_state.session_title}
  </h2>
  <p style="color:#666; font-size:0.82rem; margin:6px 0 0;">
    Hybrid RAG · BM25 + Dense · Cross-encoder Reranking
  </p>
</div>
""", unsafe_allow_html=True)

# ── Render conversation ──
st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; padding:60px 20px; color:#555;">
      <div style="font-size:2.5rem; margin-bottom:16px;">⚡</div>
      <p style="font-size:0.95rem; color:#666; max-width:440px; margin:0 auto; line-height:1.7;">
        Upload your PDFs in the sidebar, then ask anything.<br>
        The default knowledge base is ready if you haven't uploaded files.
      </p>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    role    = msg["role"]
    content = msg["content"]
    sources = msg.get("sources", "")
    ts      = msg.get("timestamp", "")[:16].replace("T", " ")

    if role == "user":
        st.markdown(f"""
        <div class="msg-row">
          <div class="msg-avatar avatar-user">U</div>
          <div>
            <div class="msg-bubble bubble-user">{content}</div>
            <div class="msg-meta">{ts}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        sources_html = ""
        if sources:
            chips = "".join(f'<span class="source-chip">{s.strip()}</span>' for s in sources.split("|") if s.strip())
            sources_html = f'<div class="source-header">Sources</div>{chips}'
        st.markdown(f"""
        <div class="msg-row">
          <div class="msg-avatar avatar-ai">K</div>
          <div>
            <div class="msg-bubble bubble-ai">{content}{sources_html}</div>
            <div class="msg-meta">{ts}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────
# INPUT BAR  (pinned to bottom)
# ─────────────────────────────────────────
st.markdown('<div class="input-bar">', unsafe_allow_html=True)

col_input, col_btn = st.columns([9, 1])

with col_input:
    query = st.text_input(
        "query",
        placeholder="Ask anything about your documents…",
        label_visibility="collapsed",
        key="chat_input"
    )

with col_btn:
    send = st.button("Send", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────
# HANDLE SEND
# ─────────────────────────────────────────
if send and query.strip():
    # 1. Record user message
    ts_now = datetime.datetime.now().isoformat()
    user_msg = {"role": "user", "content": query, "sources": "", "timestamp": ts_now}
    st.session_state.messages.append(user_msg)
    save_message(st.session_state.session_id, "user", query, "", )

    # 2. Update session title from first question
    if st.session_state.session_title == "New Chat":
        new_title = query[:40] + ("…" if len(query) > 40 else "")
        st.session_state.session_title = new_title
        conn = sqlite3.connect(DB_PATH)
        conn.execute("UPDATE sessions SET title=? WHERE session_id=?",
                     (new_title, st.session_state.session_id))
        conn.commit()
        conn.close()

    # 3. Choose pipeline
    pipeline = st.session_state.pipeline or get_default_pipeline()

    # 4. Query with spinner
    with st.spinner("Searching knowledge base…"):
        result = pipeline.ask(query)

    answer    = result["answer"]
    documents = result["documents"]

    # 5. Build source string
    source_parts = []
    for doc in documents:
        fn = doc.metadata.get("file_name", "doc")
        pg = doc.metadata.get("page_label", "?")
        source_parts.append(f"{fn} p.{pg}")
    sources_str = " | ".join(source_parts)

    # 6. Record AI message
    ai_ts = datetime.datetime.now().isoformat()
    ai_msg = {"role": "assistant", "content": answer, "sources": sources_str, "timestamp": ai_ts}
    st.session_state.messages.append(ai_msg)
    save_message(st.session_state.session_id, "assistant", answer, sources_str)

    st.rerun()