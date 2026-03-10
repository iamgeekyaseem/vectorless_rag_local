"""
core/retriever.py — PageIndex Tree Search Retriever
──────────────────────────────────────────────────────────────
Given a question and a DocumentIndex tree, this module uses the
LLM to reason over the tree and find the most relevant pages.

THE 3-STEP RETRIEVAL PROCESS:
  1. NAVIGATE  — LLM reads the tree outline, picks best node(s)
  2. FETCH     — Pull the actual page text for those nodes
  3. VERIFY    — LLM confirms the content actually answers the question

WHY VERIFY?
  Small models like Gemma3:4b sometimes navigate to the wrong node.
  The verify step catches this and tries the next-best option.
  It's a safety net — not needed for GPT-4o but helpful locally.
"""

import json
import re
from core.llm import chat
from core.models import DocumentIndex, IndexNode


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _extract_json(text: str) -> str:
    """Strip markdown fences and return clean JSON string."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _get_pages_text(pages: list[dict], start_page: int, end_page: int) -> str:
    """
    Fetch and format the actual text for a page range.
    Returns a clean string with page markers.
    """
    parts = []
    for page in pages:
        if start_page <= page["page"] <= end_page:
            parts.append(f"--- Page {page['page']} ---\n{page['text']}")
    return "\n\n".join(parts)


def _node_by_id(index: DocumentIndex, node_id: str) -> IndexNode | None:
    """
    Find a node anywhere in the tree by its node_id.
    Uses the flat node list for fast lookup.
    """
    for node in index.all_nodes_flat():
        if node.node_id == node_id:
            return node
    return None


# ─────────────────────────────────────────────────────────────
# STEP 1 — NAVIGATE: LLM reasons over the tree outline
# ─────────────────────────────────────────────────────────────

def navigate_tree(
    question : str,
    index    : DocumentIndex,
    model    : str,
    provider : str,
    top_k    : int = 2
) -> list[str]:
    """
    Ask the LLM to navigate the tree and identify the most
    relevant node IDs for a given question.

    Args:
        question : the user's question
        index    : the DocumentIndex tree
        top_k    : how many nodes to return (default 2)
                   returning 2 gives us a fallback if first is wrong

    Returns:
        list of node_id strings, ordered by relevance
        e.g. ["0005", "0003"]

    HOW THE PROMPT WORKS:
        We give the LLM the full tree outline (just titles + summaries,
        NOT the actual page text — that's the key insight).
        The LLM uses its reasoning to pick the right branch.
        This is exactly how a human would use a table of contents.
    """
    tree_outline = index.to_text_outline()

    prompt = f"""You are a document navigation expert. 
You have a document index (table of contents with summaries) and a question.
Your job is to identify which section(s) of the document are most likely to contain the answer.

QUESTION: {question}

DOCUMENT INDEX:
{tree_outline}

Instructions:
- Read the question carefully
- Look at each section's title and summary
- Identify the {top_k} most relevant sections by their node_id
- Order them from most to least relevant
- Consider: which section would a human expert go to first?

Respond ONLY with valid JSON, no other text:
{{
  "reasoning": "brief explanation of why you chose these sections",
  "node_ids": ["0001", "0002"]
}}
"""

    response = chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        provider=provider
    )

    try:
        clean = _extract_json(response)
        data = json.loads(clean)
        node_ids = data.get("node_ids", [])
        reasoning = data.get("reasoning", "")
        print(f"[retriever] Navigation reasoning: {reasoning}")
        print(f"[retriever] Selected nodes: {node_ids}")
        return node_ids
    except Exception as e:
        print(f"[retriever] Warning: Could not parse navigation response: {e}")
        print(f"[retriever] Raw response: {response}")
        # Fallback: return first leaf node
        leaves = []
        for node in index.nodes:
            leaves.extend(node.all_leaves())
        return [leaves[0].node_id] if leaves else []


# ─────────────────────────────────────────────────────────────
# STEP 2 — FETCH: Get actual page text for selected nodes
# ─────────────────────────────────────────────────────────────

def fetch_node_content(
    node  : IndexNode,
    pages : list[dict]
) -> dict:
    """
    Fetch the actual text content for a node's page range.

    Returns a dict with:
        {
            "node_id"   : "0005",
            "title"     : "3.1 Sea Level Rise",
            "page_range": "pages 9-11",
            "text"      : "full text of pages 9-11..."
        }
    """
    text = _get_pages_text(pages, node.start_page, node.end_page)

    return {
        "node_id"    : node.node_id,
        "title"      : node.title,
        "page_range" : node.page_range(),
        "text"       : text
    }


# ─────────────────────────────────────────────────────────────
# STEP 3 — VERIFY: Confirm the content is actually relevant
# ─────────────────────────────────────────────────────────────

def verify_relevance(
    question : str,
    content  : dict,
    model    : str,
    provider : str
) -> tuple[bool, str]:
    """
    Ask the LLM: "Does this content actually answer the question?"

    This is a cheap yes/no call that saves us from giving the
    RAG pipeline irrelevant context.

    Returns:
        (is_relevant: bool, reason: str)

    Example:
        (True,  "The section directly discusses sea level projections")
        (False, "This section covers weather patterns, not sea levels")
    """
    preview = content["text"][:3000]  # Keep it short for verification

    prompt = f"""Does the following document section contain information that would help answer the question?

QUESTION: {question}

SECTION: {content['title']} ({content['page_range']})
CONTENT PREVIEW:
{preview}

Answer with ONLY valid JSON:
{{
  "is_relevant": true,
  "reason": "one sentence explanation"
}}
"""

    response = chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        provider=provider
    )

    try:
        clean = _extract_json(response)
        data = json.loads(clean)
        is_relevant = bool(data.get("is_relevant", True))
        reason = data.get("reason", "")
        return is_relevant, reason
    except Exception:
        # If we can't parse, assume relevant (better safe than sorry)
        return True, "Could not verify, assuming relevant"


# ─────────────────────────────────────────────────────────────
# MAIN FUNCTION — Full retrieval pipeline
# ─────────────────────────────────────────────────────────────

class RetrievalResult:
    """
    Clean container for retrieval results.
    Passed to the RAG pipeline in the next step.
    """
    def __init__(self):
        self.contents    : list[dict] = []   # list of fetched content dicts
        self.node_ids    : list[str]  = []   # which nodes were retrieved
        self.reasoning   : str        = ""   # why these nodes were chosen
        self.verified    : bool       = False

    def combined_context(self) -> str:
        """
        Merge all retrieved content into one context string.
        This is what gets injected into the RAG prompt.
        """
        parts = []
        for c in self.contents:
            parts.append(
                f"[Source: {c['title']} — {c['page_range']}]\n{c['text']}"
            )
        return "\n\n---\n\n".join(parts)

    def source_citations(self) -> list[str]:
        """Returns a list of source strings for display in the UI"""
        return [f"{c['title']} ({c['page_range']})" for c in self.contents]


def retrieve(
    question    : str,
    index       : DocumentIndex,
    pages       : list[dict],
    model       : str,
    provider    : str,
    top_k       : int  = 2,
    verify      : bool = True
) -> RetrievalResult:
    """
    THE MAIN FUNCTION — full tree search retrieval pipeline.

    Args:
        question : the user's question
        index    : DocumentIndex tree (from indexer.build_index)
        pages    : raw pages (from loader.load_document)
        model    : LLM model name
        provider : "ollama" | "openai" | "groq"
        top_k    : number of nodes to retrieve
        verify   : whether to run the relevance verification step

    Returns:
        RetrievalResult with the relevant page content

    Example:
        result = retrieve(
            question = "What are the effects of sea level rise?",
            index    = my_index,
            pages    = my_pages,
            model    = "gemma3:4b",
            provider = "ollama"
        )
        print(result.combined_context())   # pass this to RAG
        print(result.source_citations())   # ["3.1 Sea Level Rise (pages 9-11)"]
    """
    result = RetrievalResult()

    # ── Step 1: Navigate the tree ─────────────────────────────
    print(f"[retriever] Navigating tree for: '{question}'")
    node_ids = navigate_tree(question, index, model, provider, top_k)

    if not node_ids:
        print("[retriever] Warning: No nodes found, falling back to first leaf")
        all_leaves = []
        for node in index.nodes:
            all_leaves.extend(node.all_leaves())
        node_ids = [all_leaves[0].node_id] if all_leaves else []

    result.node_ids = node_ids

    # ── Step 2: Fetch content for each node ───────────────────
    for node_id in node_ids:
        node = _node_by_id(index, node_id)
        if not node:
            print(f"[retriever] Warning: node_id '{node_id}' not found in tree")
            continue

        content = fetch_node_content(node, pages)

        # ── Step 3: Verify relevance ──────────────────────────
        if verify:
            is_relevant, reason = verify_relevance(question, content, model, provider)
            print(f"[retriever] Node {node_id} relevant: {is_relevant} — {reason}")
            if is_relevant:
                result.contents.append(content)
                result.verified = True
        else:
            result.contents.append(content)

    # If verification filtered everything out, fall back to first node
    if not result.contents and node_ids:
        print("[retriever] All nodes filtered by verify — using first node as fallback")
        node = _node_by_id(index, node_ids[0])
        if node:
            result.contents.append(fetch_node_content(node, pages))

    print(f"[retriever] Retrieved {len(result.contents)} content block(s)")
    return result