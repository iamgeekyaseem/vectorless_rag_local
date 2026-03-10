"""
app.py — Streamlit Web App Entry Point
───────────────────────────────────────
This is the file you run to start the app:
    streamlit run app.py

Right now it's a skeleton — we'll fill it in as we build each step.
"""

import streamlit as st
from core.llm import get_ollama_models, chat

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

    # Provider selector
    provider = st.selectbox(
        "LLM Provider",
        options=["ollama", "openai", "groq"],
        index=0,        # default = ollama
        help="Choose where to run your LLM"
    )

    # Model selector — dynamically loads your local Ollama models
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

# ── Main Area: Quick Test ─────────────────────────────────────
st.subheader("💬 Quick Connection Test")
st.write("Let's make sure your LLM is connected before we build more!")

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
st.caption("📍 Step 1 complete — More features coming in next steps!")
