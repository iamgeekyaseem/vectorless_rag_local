"""
core/rag.py — RAG Pipeline
──────────────────────────────────────────────────────────────
Takes retrieved context (from retriever.py) and a question,
crafts a grounded prompt, and returns a final answer.

RAG = Retrieval Augmented Generation
  Retrieval  → retriever.py already did this (gave us context)
  Augmented  → we inject that context into the LLM prompt
  Generation → LLM generates an answer grounded in the context

WHY THIS MATTERS:
  Without RAG: LLM answers from its training data (may hallucinate)
  With RAG:    LLM answers ONLY from your document (grounded, accurate)
"""

from core.llm import chat
from core.retriever import RetrievalResult


# ─────────────────────────────────────────────────────────────
# PROMPT TEMPLATES
# ─────────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are a helpful document assistant. 
Answer questions based ONLY on the provided document context.
Rules:
- If the answer is in the context, answer clearly and accurately
- Always mention which section/page your answer comes from
- If the context doesn't contain the answer, say so honestly
- Do not use outside knowledge — stick to the document
- Be concise but complete"""

RAG_USER_TEMPLATE = """DOCUMENT CONTEXT:
{context}

QUESTION: {question}

Answer based only on the context above. Cite the source section and page numbers."""


# ─────────────────────────────────────────────────────────────
# MAIN RAG FUNCTION
# ─────────────────────────────────────────────────────────────

def rag_answer(
    question         : str,
    retrieval_result : RetrievalResult,
    model            : str,
    provider         : str,
    chat_history     : list[dict] = None,
    stream           : bool = False
):
    """
    Generate a grounded answer using retrieved context.

    Args:
        question         : the user's question
        retrieval_result : output from retriever.retrieve()
        model            : LLM model name
        provider         : "ollama" | "openai" | "groq"
        chat_history     : previous messages for multi-turn conversation
                           list of {"role": "user"/"assistant", "content": "..."}
        stream           : stream tokens back (only works with ollama)

    Returns:
        str | generator — the answer text (or stream if stream=True)

    Example:
        result = retrieve(question, index, pages, model, provider)
        answer = rag_answer(question, result, model, provider)
        print(answer)
    """
    context = retrieval_result.combined_context()

    # Build the messages list
    messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT}]

    # Add chat history for multi-turn conversation
    if chat_history:
        messages.extend(chat_history)

    # Add current question with context
    user_content = RAG_USER_TEMPLATE.format(
        context=context,
        question=question
    )
    messages.append({"role": "user", "content": user_content})

    return chat(
        messages=messages,
        model=model,
        provider=provider,
        stream=stream
    )


def rag_answer_with_sources(
    question         : str,
    retrieval_result : RetrievalResult,
    model            : str,
    provider         : str,
    chat_history     : list[dict] = None
) -> dict:
    """
    Same as rag_answer but also returns source citations.

    Returns:
        {
            "answer"  : "The sea level is projected to rise by...",
            "sources" : ["3.1 Sea Level Rise (pages 9-11)", ...]
        }
    """
    answer = rag_answer(
        question=question,
        retrieval_result=retrieval_result,
        model=model,
        provider=provider,
        chat_history=chat_history
    )

    return {
        "answer"  : answer,
        "sources" : retrieval_result.source_citations()
    }