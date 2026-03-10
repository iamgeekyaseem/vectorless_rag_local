"""
app.py — Streamlit Web App Entry Point
───────────────────────────────────────
Run with:
    uv run streamlit run app.py
"""

import streamlit as st
from core.llm import get_ollama_models, chat
from core.loader import load_document, save_uploaded_file, document_stats
from core.indexer import build_index
from core.models import DocumentIndex

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
        index=0
    )

    if provider == "ollama":
        available_models = get_ollama_models()
        if not available_models:
            st.error("⚠️ Ollama not running! Run: `ollama serve`")
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
                model=model, provider=provider
            )
            st.success("✅ Connected!")
            st.write(f"**Reply:** {reply}")
        except Exception as e:
            st.error(f"❌ Error: {e}")

st.divider()

# ── Step 2: Document Upload ───────────────────────────────────
st.subheader("📄 Step 2 — Upload a Document")

uploaded_file = st.file_uploader(
    "Choose a file",
    type=["pdf", "txt", "md"]
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
        preview = pages[0]["text"]
        st.text(preview[:1000] + "..." if len(preview) > 1000 else preview)

    with st.expander("📖 Browse all pages"):
        page_num = st.slider("Page", min_value=1, max_value=len(pages), value=1)
        selected = next(p for p in pages if p["page"] == page_num)
        st.text(selected["text"])

st.divider()

# ── Step 3: Build PageIndex Tree ─────────────────────────────
st.subheader("🌲 Step 3 — Build PageIndex Tree")

if "pages" not in st.session_state:
    st.info("⬆️ Upload a document first (Step 2)")
else:
    pages = st.session_state["pages"]

    col_a, col_b = st.columns([2, 1])
    with col_b:
        batch_size = st.slider(
            "Pages per batch",
            min_value=2, max_value=10, value=5,
            help="How many pages the LLM reads at once. Lower = safer for small models."
        )

    with col_a:
        build_btn = st.button("🌲 Build PageIndex Tree", type="primary")

    if build_btn:
        progress_box = st.empty()
        progress_msgs = []

        def on_progress(msg):
            progress_msgs.append(msg)
            # Show last 4 messages so the user sees live updates
            progress_box.info("\n\n".join(progress_msgs[-4:]))

        with st.spinner("Building tree index... (this takes a minute)"):
            try:
                index = build_index(
                    pages=pages,
                    model=model,
                    provider=provider,
                    batch_size=batch_size,
                    on_progress=on_progress
                )
                st.session_state["index"] = index
                progress_box.empty()
                st.success("✅ PageIndex Tree built!")
            except Exception as e:
                st.error(f"❌ Error building index: {e}")

    # Show the tree if it's been built
    if "index" in st.session_state:
        index: DocumentIndex = st.session_state["index"]

        st.write(f"**📘 {index.title}**")
        st.caption(index.description)

        with st.expander("🌲 View Full Tree Structure", expanded=True):
            st.code(index.to_text_outline(), language=None)

        with st.expander("💾 Download Tree as JSON"):
            st.download_button(
                label="Download index.json",
                data=index.to_json(),
                file_name="pageindex.json",
                mime="application/json"
            )

st.divider()
st.caption("📍 Step 3 complete — PageIndex Tree Builder ready! Next: Tree Search Retriever")