"""
app.py — Vectorless RAG Agent — Full Streamlit App
────────────────────────────────────────────────────
Run with:
    uv run streamlit run app.py
"""

import os
import streamlit as st

from core.llm       import get_ollama_models, chat
from core.loader    import load_document, save_uploaded_file, document_stats
from core.indexer   import build_index
from core.models    import DocumentIndex
from core.retriever import retrieve
from core.rag       import rag_answer_with_sources
from core.agent     import run_agent, AgentStep

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Vectorless RAG Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Session State Defaults ────────────────────────────────────
for key, default in {
    "pages"       : None,
    "filename"    : None,
    "index"       : None,
    "chat_history": [],
    "agent_steps" : []
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ═════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🧠 Vectorless Agent")
    st.caption("PageIndex · Ollama · Streamlit")
    st.divider()

    # ── LLM Provider ─────────────────────────────────────────
    st.subheader("⚙️ Model Settings")
    provider = st.selectbox(
        "Provider",
        options=["ollama", "openai", "groq"],
        index=0
    )

    if provider == "ollama":
        available_models = get_ollama_models()
        if not available_models:
            st.error("⚠️ Ollama not running!\nRun: `ollama serve`")
            model = "gemma3:4b"
        else:
            model = st.selectbox("Model", options=available_models)
    elif provider == "openai":
        model = st.text_input("Model", value="gpt-4o")
        if not os.getenv("OPENAI_API_KEY"):
            st.warning("Set OPENAI_API_KEY in .env")
    elif provider == "groq":
        model = st.text_input("Model", value="llama3-8b-8192")
        if not os.getenv("GROQ_API_KEY"):
            st.warning("Set GROQ_API_KEY in .env")

    st.info(f"🟢 **{model}** via **{provider}**")
    st.divider()

    # ── Document Status ───────────────────────────────────────
    st.subheader("📄 Document Status")
    if st.session_state["filename"]:
        st.success(f"✅ {st.session_state['filename']}")
        pages = st.session_state["pages"]
        st.caption(f"{len(pages)} pages loaded")
    else:
        st.info("No document loaded")

    if st.session_state["index"]:
        index: DocumentIndex = st.session_state["index"]
        st.success(f"🌲 Index ready")
        st.caption(f"{len(index.all_nodes_flat())} nodes")
        with st.expander("View tree outline"):
            st.code(index.to_text_outline(), language=None)
    else:
        st.info("No index built yet")

    st.divider()

    # ── Clear Everything ──────────────────────────────────────
    if st.button("🗑️ Clear Everything", use_container_width=True):
        for key in ["pages", "filename", "index", "chat_history", "agent_steps"]:
            st.session_state[key] = [] if key in ["chat_history", "agent_steps"] else None
        st.rerun()


# ═════════════════════════════════════════════════════════════
# MAIN AREA — TABS
# ═════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "📄 Setup",
    "💬 RAG Chat",
    "🤖 Agent",
    "🔍 Debug"
])


# ─────────────────────────────────────────────────────────────
# TAB 1 — SETUP (Upload + Build Index)
# ─────────────────────────────────────────────────────────────

with tab1:
    st.header("📄 Document Setup")

    col_upload, col_index = st.columns(2)

    # ── Upload ────────────────────────────────────────────────
    with col_upload:
        st.subheader("1️⃣ Upload Document")
        uploaded_file = st.file_uploader(
            "PDF or TXT file",
            type=["pdf", "txt", "md"],
            label_visibility="collapsed"
        )

        if uploaded_file:
            with st.spinner("Reading..."):
                path  = save_uploaded_file(uploaded_file)
                pages = load_document(path)
                st.session_state["pages"]    = pages
                st.session_state["filename"] = uploaded_file.name
                st.session_state["index"]    = None  # reset index on new upload

            stats = document_stats(pages)
            st.success(f"✅ Loaded **{uploaded_file.name}**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Pages",      stats["total_pages"])
            c2.metric("Characters", f"{stats['total_chars']:,}")
            c3.metric("Avg/Page",   f"{stats['avg_chars_per_page']:,}")

            with st.expander("👀 Preview Page 1"):
                txt = pages[0]["text"]
                st.text(txt[:1200] + "..." if len(txt) > 1200 else txt)

            with st.expander("📖 Browse pages"):
                pn = st.slider("Page", 1, len(pages), 1)
                st.text(next(p["text"] for p in pages if p["page"] == pn))

    # ── Build Index ───────────────────────────────────────────
    with col_index:
        st.subheader("2️⃣ Build PageIndex Tree")

        if not st.session_state["pages"]:
            st.info("Upload a document first →")
        else:
            batch_size = st.slider(
                "Pages per batch",
                min_value=2, max_value=10, value=5,
                help="Lower = safer for small models. Higher = faster but needs bigger context."
            )

            build_btn = st.button("🌲 Build Index", type="primary", use_container_width=True)

            if build_btn:
                progress_placeholder = st.empty()
                msgs = []

                def on_progress(msg):
                    msgs.append(f"• {msg}")
                    progress_placeholder.info("\n".join(msgs[-5:]))

                with st.spinner("Building PageIndex tree..."):
                    try:
                        index = build_index(
                            pages=st.session_state["pages"],
                            model=model,
                            provider=provider,
                            batch_size=batch_size,
                            on_progress=on_progress
                        )
                        st.session_state["index"] = index
                        st.session_state["chat_history"] = []  # reset chat on new index
                        progress_placeholder.empty()
                        st.success("✅ PageIndex built!")
                        st.balloons()
                    except Exception as e:
                        st.error(f"❌ {e}")

            if st.session_state["index"]:
                index = st.session_state["index"]
                st.info(f"**{index.title}**\n\n{index.description}")

                nodes = index.all_nodes_flat()
                st.metric("Tree nodes", len(nodes))

                # Download index
                st.download_button(
                    "💾 Download index.json",
                    data=index.to_json(),
                    file_name="pageindex.json",
                    mime="application/json",
                    use_container_width=True
                )

    # ── Load saved index ──────────────────────────────────────
    st.divider()
    st.subheader("Or load a previously saved index")
    saved_index_file = st.file_uploader(
        "Upload pageindex.json",
        type=["json"],
        key="index_upload"
    )
    if saved_index_file:
        try:
            index = DocumentIndex.from_json(saved_index_file.read().decode())
            st.session_state["index"] = index
            st.success(f"✅ Loaded index: **{index.title}** ({len(index.all_nodes_flat())} nodes)")
        except Exception as e:
            st.error(f"❌ Could not load index: {e}")


# ─────────────────────────────────────────────────────────────
# TAB 2 — RAG CHAT
# ─────────────────────────────────────────────────────────────

with tab2:
    st.header("💬 RAG Chat")
    st.caption("Ask questions about your document. Answers are grounded in the PageIndex.")

    if not st.session_state["index"]:
        st.warning("⬆️ Build the PageIndex first (Setup tab)")
    else:
        index = st.session_state["index"]
        pages = st.session_state["pages"]

        # ── Chat Settings ─────────────────────────────────────
        with st.expander("⚙️ Chat settings"):
            top_k      = st.slider("Sections to retrieve (top_k)", 1, 4, 2)
            use_verify = st.checkbox("Verify relevance", value=True)
            show_src   = st.checkbox("Show sources", value=True)

        # ── Chat History Display ──────────────────────────────
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state["chat_history"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg.get("sources") and show_src:
                        st.caption("📄 Sources: " + " · ".join(msg["sources"]))

        # ── Chat Input ────────────────────────────────────────
        question = st.chat_input("Ask a question about your document...")

        if question:
            # Show user message
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(question)

            # Add to history
            st.session_state["chat_history"].append({
                "role": "user", "content": question
            })

            # Retrieve + Answer
            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("Searching document..."):
                        try:
                            # Get retrieval
                            retrieval = retrieve(
                                question=question,
                                index=index,
                                pages=pages,
                                model=model,
                                provider=provider,
                                top_k=top_k,
                                verify=use_verify
                            )

                            # Get RAG answer
                            # Pass history (excluding last user message — it's in retrieval)
                            history_for_rag = [
                                {"role": m["role"], "content": m["content"]}
                                for m in st.session_state["chat_history"][:-1]
                                if m["role"] in ("user", "assistant")
                            ]

                            result = rag_answer_with_sources(
                                question=question,
                                retrieval_result=retrieval,
                                model=model,
                                provider=provider,
                                chat_history=history_for_rag if history_for_rag else None
                            )

                            answer  = result["answer"]
                            sources = result["sources"]

                            st.markdown(answer)
                            if sources and show_src:
                                st.caption("📄 Sources: " + " · ".join(sources))

                            # Save to history
                            st.session_state["chat_history"].append({
                                "role"   : "assistant",
                                "content": answer,
                                "sources": sources
                            })

                        except Exception as e:
                            err = f"❌ Error: {e}"
                            st.error(err)

        # ── Clear Chat ────────────────────────────────────────
        if st.session_state["chat_history"]:
            if st.button("🗑️ Clear chat history"):
                st.session_state["chat_history"] = []
                st.rerun()


# ─────────────────────────────────────────────────────────────
# TAB 3 — AGENT
# ─────────────────────────────────────────────────────────────

with tab3:
    st.header("🤖 ReAct Agent")
    st.caption("For complex questions that need multi-step reasoning across the document.")

    if not st.session_state["index"]:
        st.warning("⬆️ Build the PageIndex first (Setup tab)")
    else:
        index = st.session_state["index"]
        pages = st.session_state["pages"]

        # ── When to use Agent vs RAG ──────────────────────────
        with st.expander("💡 When to use Agent vs RAG Chat?"):
            st.markdown("""
**RAG Chat** — best for:
- Simple, direct questions: *"What is X?"*
- Looking up specific facts from the document

**Agent** — best for:
- Complex, multi-part questions: *"Compare X and Y"*
- Questions requiring multiple sections: *"Summarize all recommendations"*
- Analytical questions: *"What are the pros and cons of approach Z?"*
            """)

        col_q, col_s = st.columns([3, 1])
        with col_q:
            agent_question = st.text_area(
                "Complex question:",
                placeholder="e.g. Compare the main causes and effects described in this document...",
                height=80
            )
        with col_s:
            max_steps = st.slider("Max steps", 2, 8, 5)

        run_btn = st.button("🤖 Run Agent", type="primary", use_container_width=False)

        if run_btn and agent_question:
            steps_placeholder = st.empty()
            steps_so_far = []

            def on_step(step: AgentStep):
                steps_so_far.append(step)
                with steps_placeholder.container():
                    for s in steps_so_far:
                        with st.expander(f"Step {s.step_num}: {s.action[:60]}...", expanded=False):
                            st.markdown(f"**💭 Thought:** {s.thought}")
                            st.markdown(f"**🔧 Action:** `{s.action}`")
                            st.markdown(f"**👁 Observation:**")
                            st.text(s.observation[:500] + "..." if len(s.observation) > 500 else s.observation)

            with st.spinner("Agent reasoning..."):
                try:
                    agent_result = run_agent(
                        question  = agent_question,
                        index     = index,
                        pages     = pages,
                        model     = model,
                        provider  = provider,
                        max_steps = max_steps,
                        on_step   = on_step
                    )
                    st.session_state["agent_steps"] = agent_result.steps
                except Exception as e:
                    st.error(f"❌ Agent error: {e}")
                    agent_result = None

            if agent_result and agent_result.answer:
                st.divider()
                st.subheader("🎯 Agent Answer")
                st.markdown(agent_result.answer)
                st.caption(f"Completed in {len(agent_result.steps)} steps")


# ─────────────────────────────────────────────────────────────
# TAB 4 — DEBUG
# ─────────────────────────────────────────────────────────────

with tab4:
    st.header("🔍 Debug & Inspect")

    if not st.session_state["index"]:
        st.info("Build an index to explore here")
    else:
        index = st.session_state["index"]
        pages = st.session_state["pages"]

        debug_tab1, debug_tab2, debug_tab3 = st.tabs([
            "🌲 Tree Nodes",
            "🔍 Test Retrieval",
            "📡 Raw LLM Test"
        ])

        # All tree nodes table
        with debug_tab1:
            st.subheader("All nodes in the tree")
            all_nodes = index.all_nodes_flat()
            for node in all_nodes:
                with st.expander(f"[{node.node_id}] {node.title} ({node.page_range()})"):
                    st.write(f"**Summary:** {node.summary}")
                    st.write(f"**Children:** {len(node.nodes)}")
                    st.write(f"**Is leaf:** {node.is_leaf()}")

        # Retrieval tester
        with debug_tab2:
            st.subheader("Test tree retrieval directly")
            dbg_q = st.text_input("Test query:", key="debug_query")
            dbg_k = st.slider("top_k", 1, 4, 2, key="debug_k")
            dbg_v = st.checkbox("Verify", value=False, key="debug_verify")

            if st.button("Retrieve", key="debug_retrieve") and dbg_q:
                with st.spinner("Retrieving..."):
                    result = retrieve(
                        question=dbg_q,
                        index=index,
                        pages=pages,
                        model=model,
                        provider=provider,
                        top_k=dbg_k,
                        verify=dbg_v
                    )
                st.write(f"**Retrieved nodes:** {result.node_ids}")
                st.write(f"**Sources:** {result.source_citations()}")
                for c in result.contents:
                    with st.expander(f"{c['title']} — {c['page_range']}"):
                        st.text(c["text"][:2000])

        # Raw LLM test
        with debug_tab3:
            st.subheader("Send a raw message to the LLM")
            raw_msg = st.text_area("Message:", height=100, key="raw_msg")
            if st.button("Send", key="raw_send") and raw_msg:
                with st.spinner("Waiting for response..."):
                    reply = chat(
                        messages=[{"role": "user", "content": raw_msg}],
                        model=model,
                        provider=provider
                    )
                st.markdown("**Response:**")
                st.write(reply)