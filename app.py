"""
app.py — Streamlit Web App Entry Point
───────────────────────────────────────
Run with:
    uv run streamlit run app.py
"""

import streamlit as st
from core.llm import get_ollama_models, chat
from core.loader import load_document, save_uploaded_file, document_stats

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Vectorless Agent",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Vectorless RAG Agent")
st.caption("Powered by PageIndex strategy + your local Ollama models")

# ── Sidebar: Model Selector ───────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    provider = st.selectbox(
        "LLM Provider",
        options=["ollama", "openai", "groq"],
        index=0,
        help="Choose where to run your LLM"
    )

    if provider == "ollama":
        available_models = get_ollama_models()
        if not available_models:
            st.error("⚠️ Ollama not running! Start it with: `ollama serve`")
            model = "gemma3:4b"
        else:
            model = st.selectbox("Model", options=available_models)
    elif provider == "openai":
        model = st.text_input("Model", value="gpt-4o")
    elif provider == "groq":
        model = st.text_input("Model", value="llama3-8b-8192")

    st.divider()
    st.info(f"🟢 Using: **{model}** via **{provider}**")

# ── Step 1: Quick Connection Test ────────────────────────────
st.subheader("💬 Step 1 — Connection Test")
test_prompt = st.text_input("Ask anything:", placeholder="What is 2 + 2?")

if st.button("Send 🚀") and test_prompt:
    with st.spinner("Thinking..."):
        try:
            reply = chat(
                messages=[{"role": "user", "content": test_prompt}],
                model=model,
                provider=provider
            )
            st.success("✅ Connected!")
            st.write(f"**Reply:** {reply}")
        except Exception as e:
            st.error(f"❌ Error: {e}")

st.divider()

# ── Step 2: Document Upload ───────────────────────────────────
st.subheader("📄 Step 2 — Upload a Document")
st.write("Upload a PDF or TXT file — we'll load it page by page.")

uploaded_file = st.file_uploader(
    "Choose a file",
    type=["pdf", "txt", "md"],
    help="PDF or plain text files supported"
)

if uploaded_file:
    with st.spinner("Reading document..."):
        file_path = save_uploaded_file(uploaded_file)
        pages = load_document(file_path)
        st.session_state["pages"] = pages
        st.session_state["filename"] = uploaded_file.name

    stats = document_stats(pages)
    st.success(f"✅ Loaded **{uploaded_file.name}**")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Pages", stats["total_pages"])
    col2.metric("Total Characters", f"{stats['total_chars']:,}")
    col3.metric("Avg Chars/Page", f"{stats['avg_chars_per_page']:,}")

    with st.expander("👀 Preview Page 1"):
        preview_text = pages[0]["text"]
        st.text(preview_text[:1000] + "..." if len(preview_text) > 1000 else preview_text)

    with st.expander("📖 Browse all pages"):
        page_num = st.slider("Page", min_value=1, max_value=len(pages), value=1)
        selected = next(p for p in pages if p["page"] == page_num)
        st.text(selected["text"])

st.divider()
st.caption("📍 Step 2 complete — Document Loader ready! Next: PageIndex Tree Builder")