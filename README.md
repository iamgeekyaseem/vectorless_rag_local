# 🧠 Vectorless RAG Agent

A fully local, vectorless RAG agent powered by **PageIndex** strategy, **Ollama**, and **Streamlit**.

No vector database. No embeddings. Pure LLM reasoning.

---

## 🚀 Quick Start

### Prerequisites
- [Python 3.11+](https://python.org)
- [uv](https://docs.astral.sh/uv/) — `pip install uv`
- [Ollama](https://ollama.ai) — download and install

### 1. Clone & install
```bash
git clone <your-repo>
cd vectorless-agent
uv sync
```

### 2. Pull a model
```bash
ollama pull gemma3:4b
```

### 3. Start Ollama
```bash
ollama serve
```

### 4. Run the app
```bash
uv run streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## 🐳 Docker Deployment

### Option A — Docker Compose (recommended, includes Ollama)
```bash
# Start everything
docker compose up -d

# Pull your model inside the Ollama container
docker compose exec ollama ollama pull gemma3:4b

# Open http://localhost:8501
```

### Option B — Docker only (if Ollama is already running locally)
```bash
docker build -t vectorless-agent .
docker run -p 8501:8501 --env-file .env vectorless-agent
```

---

## ☁️ Deploy to the Web

### Streamlit Community Cloud (free)
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → Deploy
4. Add secrets in Settings → Secrets (paste your `.env` contents)

> **Note:** Streamlit Cloud can't run Ollama (no local GPU).
> For cloud deployment, set `OPENAI_API_KEY` or `GROQ_API_KEY` in secrets
> and select OpenAI/Groq as your provider in the app.

### Railway / Render / Fly.io
Use the `Dockerfile` — these platforms support Docker deployments directly.
Set your environment variables in the platform's dashboard.

---

## 🏗️ Architecture

```
vectorless-agent/
├── app.py              # Streamlit UI (4 tabs)
├── core/
│   ├── llm.py          # Unified LLM client (Ollama/OpenAI/Groq)
│   ├── loader.py       # PDF + TXT document loader (page-by-page)
│   ├── models.py       # Pydantic tree data models
│   ├── indexer.py      # PageIndex tree builder
│   ├── retriever.py    # Tree search retriever (3-step: navigate→fetch→verify)
│   ├── rag.py          # RAG pipeline (context injection + answer)
│   └── agent.py        # ReAct agent loop (multi-step reasoning)
├── uploads/            # Uploaded documents (temp)
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## 💡 How PageIndex Works

Traditional RAG uses vector similarity to find chunks.
**PageIndex** uses LLM reasoning to navigate a document tree:

1. **Index** — LLM reads the document and builds a hierarchical Table of Contents with page ranges + summaries
2. **Navigate** — LLM reads the tree outline and reasons which section answers the question
3. **Fetch** — Pull the actual pages for that section
4. **Answer** — LLM generates a grounded answer from those pages

Result: no embeddings, no vector DB, fully explainable retrieval.

---

## 🔑 Adding API Keys (for cloud models)

Edit `.env`:
```
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
```

Then in the app sidebar, switch Provider to `openai` or `groq`.

---

## 📦 Packages Used

| Package | Purpose |
|---|---|
| `streamlit` | Web UI |
| `ollama` | Local Ollama client |
| `pypdf2` | PDF text extraction |
| `pydantic` | Tree index data models |
| `openai` | OpenAI + Groq client |
| `python-dotenv` | .env file loading |