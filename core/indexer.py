"""
core/indexer.py — PageIndex Tree Builder
──────────────────────────────────────────────────────────────
Takes a loaded document (list of pages) and uses the LLM to
build a hierarchical tree index — the PageIndex.

HOW IT WORKS (2 phases):

  Phase 1 — SCAN
    We send batches of pages to the LLM and ask:
    "What sections/chapters are in these pages?"
    The LLM identifies structure without reading everything.

  Phase 2 — BUILD
    We take all identified sections and ask the LLM to:
    "Organize these into a clean tree with summaries"
    Result: a full DocumentIndex tree saved as JSON.

WHY BATCHES?
    Gemma3:4b has a limited context window (~8k tokens).
    A whole 50-page document won't fit in one call.
    So we process pages in batches, then merge the results.
"""

import json
import re
from core.llm import chat
from core.models import DocumentIndex, IndexNode


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _pages_to_text(pages: list[dict], start: int, end: int) -> str:
    """
    Convert a slice of pages into a single text string for the LLM.
    Adds clear page markers so the LLM knows where each page starts.

    Output looks like:
        --- Page 3 ---
        text of page 3...

        --- Page 4 ---
        text of page 4...
    """
    parts = []
    for page in pages:
        if start <= page["page"] <= end:
            parts.append(f"--- Page {page['page']} ---\n{page['text']}")
    return "\n\n".join(parts)


def _extract_json(text: str) -> str:
    """
    Extract a JSON block from LLM output.
    LLMs sometimes wrap JSON in ```json ... ``` markdown fences.
    This strips those fences and returns clean JSON.
    """
    # Try to find ```json ... ``` block
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()

    # Try to find a raw { ... } block
    match = re.search(r"(\{[\s\S]*\})", text)
    if match:
        return match.group(1).strip()

    return text.strip()


# ─────────────────────────────────────────────────────────────
# PHASE 1 — SCAN PAGES FOR SECTIONS
# ─────────────────────────────────────────────────────────────

def _scan_batch(
    pages: list[dict],
    start_page: int,
    end_page: int,
    model: str,
    provider: str
) -> list[dict]:
    """
    Ask the LLM to identify sections within a batch of pages.

    Returns a list of raw section dicts like:
    [
        {"title": "Introduction", "start_page": 1, "end_page": 2},
        {"title": "Background",   "start_page": 2, "end_page": 3},
        ...
    ]
    """
    batch_text = _pages_to_text(pages, start_page, end_page)

    prompt = f"""You are analyzing a document. Below are pages {start_page} to {end_page}.

Identify the main sections or chapters in these pages.
For each section, provide:
- title: the section heading or descriptive name
- start_page: which page the section begins on
- end_page: which page the section ends on (inclusive)

Respond ONLY with a valid JSON object in this exact format:
{{
  "sections": [
    {{"title": "Section Name", "start_page": 1, "end_page": 3}},
    {{"title": "Another Section", "start_page": 4, "end_page": 6}}
  ]
}}

If the pages have no clear sections, create one entry for the entire range.
Do not include any text outside the JSON.

DOCUMENT PAGES:
{batch_text[:6000]}
"""

    response = chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        provider=provider
    )

    try:
        clean = _extract_json(response)
        data = json.loads(clean)
        return data.get("sections", [])
    except Exception as e:
        print(f"[indexer] Warning: Could not parse batch {start_page}-{end_page}: {e}")
        # Fallback: treat the whole batch as one section
        return [{"title": f"Pages {start_page}-{end_page}", "start_page": start_page, "end_page": end_page}]


# ─────────────────────────────────────────────────────────────
# PHASE 2 — SUMMARIZE + BUILD TREE
# ─────────────────────────────────────────────────────────────

def _summarize_section(
    pages: list[dict],
    section: dict,
    model: str,
    provider: str
) -> str:
    """
    Ask the LLM to write a short summary for a section.
    Uses the actual page text for that section.
    """
    section_text = _pages_to_text(
        pages,
        section["start_page"],
        section["end_page"]
    )

    prompt = f"""Read the following section titled "{section['title']}" and write a 1-2 sentence summary.
Be concise and focus on the key information covered.
Respond with ONLY the summary text, nothing else.

SECTION TEXT:
{section_text[:4000]}
"""

    summary = chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        provider=provider
    )
    return summary.strip()


def _build_tree_structure(
    sections: list[dict],
    model: str,
    provider: str
) -> list[dict]:
    """
    Ask the LLM to organize flat sections into a nested tree.

    Input (flat list):
        [
            {"title": "Introduction", "start_page": 1, "end_page": 2},
            {"title": "1.1 Background", "start_page": 1, "end_page": 1},
            {"title": "Methods", "start_page": 3, "end_page": 8},
            ...
        ]

    Output (nested):
        [
            {
                "title": "Introduction", "start_page": 1, "end_page": 2,
                "children": [
                    {"title": "1.1 Background", "start_page": 1, "end_page": 1, "children": []}
                ]
            },
            ...
        ]
    """
    sections_text = json.dumps(sections, indent=2)

    prompt = f"""You are organizing document sections into a hierarchical tree structure.

Below is a flat list of sections. Organize them into a nested tree where:
- Top-level nodes are main chapters or major sections
- Child nodes are subsections that fall within a parent's page range
- A node is a child if its pages are fully contained within the parent's page range

Respond ONLY with valid JSON in this exact format:
{{
  "tree": [
    {{
      "title": "Chapter Title",
      "start_page": 1,
      "end_page": 10,
      "children": [
        {{
          "title": "Subsection Title",
          "start_page": 1,
          "end_page": 5,
          "children": []
        }}
      ]
    }}
  ]
}}

SECTIONS TO ORGANIZE:
{sections_text}
"""

    response = chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        provider=provider
    )

    try:
        clean = _extract_json(response)
        data = json.loads(clean)
        return data.get("tree", sections)
    except Exception as e:
        print(f"[indexer] Warning: Could not build tree structure: {e}")
        # Fallback: return flat list as-is
        return [{"title": s["title"], "start_page": s["start_page"],
                 "end_page": s["end_page"], "children": []} for s in sections]


# ─────────────────────────────────────────────────────────────
# ASSEMBLE — Convert raw dicts → Pydantic IndexNode objects
# ─────────────────────────────────────────────────────────────

def _dict_to_node(data: dict, node_id_counter: list, pages: list[dict], model: str, provider: str) -> IndexNode:
    """
    Recursively convert a raw dict into an IndexNode.
    Also fetches a summary from the LLM for each node.

    node_id_counter is a mutable list [int] used as a shared counter
    across recursive calls (Python trick for mutable default).
    """
    node_id_counter[0] += 1
    node_id = str(node_id_counter[0]).zfill(4)  # "0001", "0002", etc.

    # Get summary from LLM
    summary = _summarize_section(
        pages,
        {"start_page": data["start_page"], "end_page": data["end_page"], "title": data["title"]},
        model,
        provider
    )

    # Recursively build children
    children = []
    for child_data in data.get("children", []):
        child_node = _dict_to_node(child_data, node_id_counter, pages, model, provider)
        children.append(child_node)

    return IndexNode(
        node_id    = node_id,
        title      = data["title"],
        start_page = data["start_page"],
        end_page   = data["end_page"],
        summary    = summary,
        nodes      = children
    )


# ─────────────────────────────────────────────────────────────
# MAIN FUNCTION — Build the full PageIndex
# ─────────────────────────────────────────────────────────────

def build_index(
    pages       : list[dict],
    model       : str = "gemma3:4b",
    provider    : str = "ollama",
    batch_size  : int = 5,
    on_progress = None
) -> DocumentIndex:
    """
    THE MAIN FUNCTION — builds a complete PageIndex tree from pages.

    Args:
        pages      : output from loader.load_document()
        model      : LLM model name
        provider   : "ollama" | "openai" | "groq"
        batch_size : how many pages to scan at once (default 5)
                     lower = more LLM calls but better for small models
                     higher = fewer calls but needs bigger context window
        on_progress: optional callback fn(message: str) for UI updates

    Returns:
        DocumentIndex — the full tree index

    Example:
        from core.loader import load_document
        from core.indexer import build_index

        pages = load_document("report.pdf")
        index = build_index(pages, model="gemma3:4b", provider="ollama")
        print(index.to_text_outline())
    """

    def progress(msg):
        print(f"[indexer] {msg}")
        if on_progress:
            on_progress(msg)

    total_pages = len(pages)
    progress(f"Starting PageIndex build for {total_pages} pages...")

    # ── Phase 1: Scan pages in batches ────────────────────────
    progress("Phase 1: Scanning document structure...")
    all_sections = []
    page_numbers = [p["page"] for p in pages]
    min_page = min(page_numbers)
    max_page = max(page_numbers)

    for batch_start in range(min_page, max_page + 1, batch_size):
        batch_end = min(batch_start + batch_size - 1, max_page)
        progress(f"  Scanning pages {batch_start}-{batch_end}...")

        sections = _scan_batch(pages, batch_start, batch_end, model, provider)
        all_sections.extend(sections)

    progress(f"Phase 1 complete: found {len(all_sections)} raw sections")

    # ── Phase 2: Build nested tree structure ──────────────────
    progress("Phase 2: Building tree structure...")
    if len(all_sections) > 1:
        tree_data = _build_tree_structure(all_sections, model, provider)
    else:
        tree_data = [{"title": s["title"], "start_page": s["start_page"],
                      "end_page": s["end_page"], "children": []} for s in all_sections]

    progress(f"Phase 2 complete: {len(tree_data)} top-level nodes")

    # ── Phase 3: Summarize + assemble Pydantic objects ────────
    progress("Phase 3: Summarizing sections...")
    counter = [0]
    index_nodes = []
    for node_data in tree_data:
        progress(f"  Summarizing: {node_data.get('title', '?')}...")
        node = _dict_to_node(node_data, counter, pages, model, provider)
        index_nodes.append(node)

    # ── Get document-level title + description ─────────────────
    progress("Generating document description...")
    first_page_text = pages[0]["text"][:2000] if pages else ""
    doc_prompt = f"""Based on the beginning of this document, provide:
1. A short title (max 10 words)
2. A 1-2 sentence description

Respond ONLY as JSON:
{{"title": "...", "description": "..."}}

DOCUMENT START:
{first_page_text}
"""
    doc_response = chat(
        messages=[{"role": "user", "content": doc_prompt}],
        model=model,
        provider=provider
    )
    try:
        doc_meta = json.loads(_extract_json(doc_response))
        doc_title = doc_meta.get("title", "Untitled Document")
        doc_description = doc_meta.get("description", "")
    except Exception:
        doc_title = "Untitled Document"
        doc_description = ""

    # ── Final assembly ─────────────────────────────────────────
    document_index = DocumentIndex(
        title       = doc_title,
        description = doc_description,
        total_pages = total_pages,
        nodes       = index_nodes
    )

    progress(f"✅ PageIndex complete! Tree has {len(document_index.all_nodes_flat())} nodes total.")
    return document_index