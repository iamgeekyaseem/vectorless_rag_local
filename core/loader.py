"""
core/loader.py — Document Loader
──────────────────────────────────────────────────────────────
Reads PDF and TXT files and returns text page-by-page.

WHY PAGE-BY-PAGE?
  PageIndex builds a tree index that references page numbers.
  e.g. "Section 3.1 covers pages 12-15"
  So we MUST preserve page boundaries — not arbitrary chunks.

OUTPUT FORMAT (always):
  A list of dicts, one per page:
  [
    {"page": 1, "text": "full text of page 1..."},
    {"page": 2, "text": "full text of page 2..."},
    ...
  ]
  This consistent format means the rest of the pipeline
  doesn't care whether the input was PDF or TXT.
"""

import os
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# PDF LOADER
# ─────────────────────────────────────────────────────────────

def load_pdf(file_path: str) -> list[dict]:
    """
    Extract text from a PDF file, one dict per page.

    Args:
        file_path : path to the .pdf file

    Returns:
        [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}, ...]

    HOW IT WORKS:
        PyPDF2 reads the PDF and gives us one page object at a time.
        We extract the text from each page and store it with its
        page number (1-indexed, like humans count pages).
    """
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        raise ImportError("pypdf2 not installed. Run: uv add pypdf2")

    pages = []
    reader = PdfReader(file_path)

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""   # extract_text() can return None

        # Clean up the text a bit
        text = text.strip()

        # Skip completely empty pages (e.g. blank separator pages)
        if not text:
            continue

        pages.append({
            "page": i + 1,          # 1-indexed (page 1, not page 0)
            "text": text
        })

    print(f"[loader] PDF loaded: {len(pages)} pages from '{Path(file_path).name}'")
    return pages


# ─────────────────────────────────────────────────────────────
# TXT LOADER
# ─────────────────────────────────────────────────────────────

def load_txt(file_path: str, chars_per_page: int = 3000) -> list[dict]:
    """
    Load a plain text file and split it into "virtual pages".

    WHY VIRTUAL PAGES?
        TXT files have no real page boundaries.
        We simulate pages by splitting every N characters.
        This keeps the same output format as PDF.

    Args:
        file_path     : path to the .txt file
        chars_per_page: how many characters = 1 "page" (default 3000)
                        ~3000 chars ≈ 1 A4 page of text

    Returns:
        [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}, ...]
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        full_text = f.read()

    # Split into chunks of chars_per_page characters
    # We try to split at paragraph boundaries (\n\n) when possible
    pages = []
    page_num = 1
    start = 0

    while start < len(full_text):
        end = start + chars_per_page

        # If we're not at the end of the file, try to find a
        # clean paragraph break to split at (looks nicer)
        if end < len(full_text):
            # Look for the last double newline before `end`
            clean_break = full_text.rfind("\n\n", start, end)
            if clean_break != -1 and clean_break > start:
                end = clean_break

        chunk = full_text[start:end].strip()

        if chunk:   # skip empty chunks
            pages.append({
                "page": page_num,
                "text": chunk
            })
            page_num += 1

        start = end

    print(f"[loader] TXT loaded: {len(pages)} virtual pages from '{Path(file_path).name}'")
    return pages


# ─────────────────────────────────────────────────────────────
# UNIFIED LOADER — use this everywhere in the project
# ─────────────────────────────────────────────────────────────

def load_document(file_path: str) -> list[dict]:
    """
    THE MAIN FUNCTION — auto-detects file type and loads it.

    Args:
        file_path : path to a .pdf or .txt file

    Returns:
        [{"page": 1, "text": "..."}, ...]

    Example:
        pages = load_document("uploads/my_report.pdf")
        print(f"Loaded {len(pages)} pages")
        print(pages[0])
        # {"page": 1, "text": "Introduction\nThis report covers..."}
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext in (".txt", ".md"):
        return load_txt(file_path)
    else:
        raise ValueError(
            f"Unsupported file type: '{ext}'. "
            f"Supported: .pdf, .txt, .md"
        )


# ─────────────────────────────────────────────────────────────
# HELPER: Save uploaded Streamlit file to disk
# ─────────────────────────────────────────────────────────────

def save_uploaded_file(uploaded_file, upload_dir: str = "uploads") -> str:
    """
    Streamlit gives us an UploadedFile object (not a path).
    This function saves it to disk and returns the file path.

    Args:
        uploaded_file : the object from st.file_uploader()
        upload_dir    : folder to save into (default: "uploads/")

    Returns:
        str: the full path to the saved file

    Example:
        uploaded = st.file_uploader("Upload PDF")
        if uploaded:
            path = save_uploaded_file(uploaded)
            pages = load_document(path)
    """
    os.makedirs(upload_dir, exist_ok=True)
    save_path = os.path.join(upload_dir, uploaded_file.name)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    print(f"[loader] Saved uploaded file to: {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────
# HELPER: Quick stats about loaded pages
# ─────────────────────────────────────────────────────────────

def document_stats(pages: list[dict]) -> dict:
    """
    Returns useful stats about a loaded document.
    Used to show info in the Streamlit UI.

    Returns:
        {
            "total_pages": 12,
            "total_chars": 45230,
            "avg_chars_per_page": 3769,
            "shortest_page": 1,
            "longest_page": 7
        }
    """
    if not pages:
        return {}

    char_counts = [len(p["text"]) for p in pages]

    return {
        "total_pages"        : len(pages),
        "total_chars"        : sum(char_counts),
        "avg_chars_per_page" : int(sum(char_counts) / len(char_counts)),
        "shortest_page"      : pages[char_counts.index(min(char_counts))]["page"],
        "longest_page"       : pages[char_counts.index(max(char_counts))]["page"],
    }