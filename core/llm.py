"""
core/llm.py — LLM Connection & Model Switcher
──────────────────────────────────────────────
This file is the single place where we talk to any LLM.
Currently supports:
  - Ollama  (local, no API key needed)
  - OpenAI  (cloud, needs OPENAI_API_KEY)
  - Groq    (cloud, needs GROQ_API_KEY — fast & free tier!)

HOW IT WORKS:
  Every LLM call in this project goes through `chat()`.
  You pick a provider + model, and it handles the rest.
"""

import os
from dotenv import load_dotenv

# Load variables from .env file into environment
load_dotenv()


# ─────────────────────────────────────────────────────────────
# OLLAMA — Local Models
# ─────────────────────────────────────────────────────────────

def get_ollama_models() -> list[str]:
    """
    Ask Ollama which models you have downloaded locally.
    Returns a list of model names like ['gemma3:4b', 'mistral', ...]
    
    WHY: This lets the UI show a live dropdown of YOUR models,
         so you never have to hardcode a model name.
    """
    try:
        import ollama
        models = ollama.list()
        # models.models is a list of Model objects; we extract just the name
        return [m.model for m in models.models]
    except Exception as e:
        print(f"[llm.py] Could not connect to Ollama: {e}")
        return []


def chat_ollama(model: str, messages: list[dict], stream: bool = False):
    """
    Send a chat request to a local Ollama model.

    Args:
        model    : e.g. 'gemma3:4b'
        messages : list of {"role": "user"/"assistant"/"system", "content": "..."}
        stream   : if True, streams tokens back one by one (for live typing effect)

    Returns:
        The assistant's reply as a string (or a stream object if stream=True)
    """
    import ollama

    if stream:
        # Stream mode: returns a generator of chunks
        return ollama.chat(model=model, messages=messages, stream=True)
    else:
        # Normal mode: waits for the full response
        response = ollama.chat(model=model, messages=messages)
        return response.message.content


# ─────────────────────────────────────────────────────────────
# OPENAI — Cloud Models (future use)
# ─────────────────────────────────────────────────────────────

def chat_openai(model: str, messages: list[dict]) -> str:
    """
    Send a chat request to OpenAI.
    Requires OPENAI_API_KEY in your .env file.

    Args:
        model    : e.g. 'gpt-4o', 'gpt-3.5-turbo'
        messages : same format as Ollama (OpenAI standard format)
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────────────────────
# GROQ — Fast Cloud Models (future use)
# ─────────────────────────────────────────────────────────────

def chat_groq(model: str, messages: list[dict]) -> str:
    """
    Send a chat request to Groq.
    Groq is FAST (runs Llama, Mixtral, etc. on custom hardware).
    Free tier available at console.groq.com

    Requires GROQ_API_KEY in your .env file.

    Args:
        model    : e.g. 'llama3-8b-8192', 'mixtral-8x7b-32768'
        messages : same format as Ollama/OpenAI
    """
    from openai import OpenAI  # Groq uses the OpenAI-compatible client!

    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────────────────────
# UNIFIED CHAT — Use this everywhere in the project!
# ─────────────────────────────────────────────────────────────

def chat(
    messages: list[dict],
    model: str = "gemma3:4b",
    provider: str = "ollama",
    stream: bool = False
):
    """
    THE MAIN FUNCTION — use this everywhere in the project.

    Routes your request to the right LLM based on provider.

    Args:
        messages : [{"role": "user", "content": "Hello!"}]
        model    : model name (depends on provider)
        provider : "ollama" | "openai" | "groq"
        stream   : only works with ollama for now

    Returns:
        str: The LLM's response text

    Example:
        reply = chat(
            messages=[{"role": "user", "content": "What is 2+2?"}],
            model="gemma3:4b",
            provider="ollama"
        )
        print(reply)  # "4"
    """
    if provider == "ollama":
        return chat_ollama(model=model, messages=messages, stream=stream)
    elif provider == "openai":
        return chat_openai(model=model, messages=messages)
    elif provider == "groq":
        return chat_groq(model=model, messages=messages)
    else:
        raise ValueError(f"Unknown provider: '{provider}'. Choose: ollama, openai, groq")
