"""
RAG Document Q&A System - Streamlit Application.

Interactive web interface for document upload, processing, and Q&A.
"""

import os
import tempfile
from pathlib import Path

import streamlit as st

from src.pipeline import RAGPipeline


# ── Page Configuration ──────────────────────────────────────────
st.set_page_config(
    page_title="📄 RAG Document Q&A",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom Styling ──────────────────────────────────────────────
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .stChatMessage {
        border-radius: 12px;
    }
    .source-card {
        background: #1e1e2e;
        border-radius: 8px;
        padding: 12px;
        margin: 4px 0;
        border-left: 3px solid #667eea;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ── Session State ───────────────────────────────────────────────
def init_session_state():
    """Initialize session state variables."""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = 0


init_session_state()


# ── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key",
    )

    st.markdown("---")
    st.markdown("## 📄 Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
        help="Upload documents to ask questions about",
    )

    # Pipeline settings
    with st.expander("🔧 Advanced Settings"):
        chunk_size = st.slider("Chunk Size", 200, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, 50)
        top_k = st.slider("Top K Results", 1, 10, 5)
        use_hybrid = st.checkbox("Hybrid Search (BM25 + Semantic)", value=True)

    # Process button
    if st.button("🚀 Process Documents", type="primary"):
        if not api_key:
            st.error("Please enter your OpenAI API key")
        elif not uploaded_files:
            st.error("Please upload at least one document")
        else:
            os.environ["OPENAI_API_KEY"] = api_key

            with st.spinner("Processing documents..."):
                # Initialize pipeline
                pipeline = RAGPipeline(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    top_k=top_k,
                    use_hybrid_search=use_hybrid,
                    api_key=api_key,
                )

                # Save and process uploaded files
                for file in uploaded_files:
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=Path(file.name).suffix,
                    ) as tmp:
                        tmp.write(file.read())
                        stats = pipeline.ingest(tmp.name)

                st.session_state.pipeline = pipeline
                st.session_state.documents_loaded = len(uploaded_files)

            st.success(f"✅ Processed {len(uploaded_files)} document(s)!")

    # Status
    if st.session_state.documents_loaded > 0:
        st.markdown("---")
        st.success(f"📚 {st.session_state.documents_loaded} " f"document(s) loaded")


# ── Main Content ────────────────────────────────────────────────
st.markdown(
    '<p class="main-header">🔍 RAG Document Q&A</p>',
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: #888;'>"
    "Upload documents and ask questions — powered by RAG</p>",
    unsafe_allow_html=True,
)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📎 View Sources"):
                for src in message["sources"]:
                    st.markdown(
                        f'<div class="source-card">'
                        f'📄 <b>{src["file"]}</b> — '
                        f'Page {src["page"]}</div>',
                        unsafe_allow_html=True,
                    )

# Chat input
if prompt := st.chat_input("Ask a question about your documents"):
    if not st.session_state.pipeline:
        st.error("⚠️ Please upload and process documents first " "(use the sidebar)")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.pipeline.query(prompt)
                st.markdown(response.answer)

                # Show sources
                if response.sources:
                    with st.expander("📎 View Sources"):
                        for src in response.sources:
                            st.markdown(
                                f'<div class="source-card">'
                                f'📄 <b>{src["file"]}</b> — '
                                f'Page {src["page"]}</div>',
                                unsafe_allow_html=True,
                            )

        # Save to history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response.answer,
                "sources": response.sources,
            }
        )
