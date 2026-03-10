# ─────────────────────────────────────────────────────────────
# Dockerfile — Vectorless RAG Agent
# ─────────────────────────────────────────────────────────────
# Build:  docker build -t vectorless-agent .
# Run:    docker run -p 8501:8501 --env-file .env vectorless-agent
#
# NOTE: This container runs the Streamlit app only.
#       Ollama must run on your HOST machine (not inside this container).
#       The app connects to Ollama via OLLAMA_BASE_URL in .env
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Install uv (fast package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files first (for Docker layer caching)
COPY pyproject.toml uv.lock* ./

# Install dependencies using uv (faster than pip)
RUN uv sync --frozen --no-dev

# Copy the rest of the app
COPY . .

# Create uploads folder
RUN mkdir -p uploads

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the app
CMD ["uv", "run", "streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
