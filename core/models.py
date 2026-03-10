"""
core/models.py — Pydantic Data Models
──────────────────────────────────────────────────────────────
Defines the shape of the PageIndex tree structure.

WHY PYDANTIC?
  Pydantic lets us define data models as Python classes.
  It automatically:
    - Validates that data has the right types
    - Converts to/from JSON (for saving & loading the tree)
    - Gives us nice autocomplete in editors

THINK OF IT LIKE:
  A blueprint for what a tree node MUST look like.
  If the LLM gives us bad data, Pydantic catches it.
"""

from pydantic import BaseModel, Field
from typing import Optional
import json


# ─────────────────────────────────────────────────────────────
# A SINGLE NODE in the tree
# ─────────────────────────────────────────────────────────────

class IndexNode(BaseModel):
    """
    Represents one section/chapter in the document tree.

    Example (leaf node — no children):
        IndexNode(
            node_id    = "0003",
            title      = "2.1 Greenhouse Gases",
            start_page = 3,
            end_page   = 5,
            summary    = "Covers CO2 and methane emission data...",
            nodes      = []
        )

    Example (parent node — has children):
        IndexNode(
            node_id    = "0002",
            title      = "2. Causes of Climate Change",
            start_page = 3,
            end_page   = 8,
            summary    = "Explores main causes including...",
            nodes      = [<IndexNode 2.1>, <IndexNode 2.2>]
        )
    """

    node_id    : str            = Field(description="Unique ID like '0001', '0002'")
    title      : str            = Field(description="Section title")
    start_page : int            = Field(description="First page of this section (1-indexed)")
    end_page   : int            = Field(description="Last page of this section (inclusive)")
    summary    : str            = Field(description="Short summary of this section's content")
    nodes      : list["IndexNode"] = Field(
                                    default=[],
                                    description="Child nodes (sub-sections)"
                                  )

    def page_range(self) -> str:
        """Returns a human-readable page range like 'pages 3-5' or 'page 7'"""
        if self.start_page == self.end_page:
            return f"page {self.start_page}"
        return f"pages {self.start_page}-{self.end_page}"

    def is_leaf(self) -> bool:
        """A leaf node has no children — it points directly to pages"""
        return len(self.nodes) == 0

    def all_leaves(self) -> list["IndexNode"]:
        """Recursively collect all leaf nodes under this node"""
        if self.is_leaf():
            return [self]
        leaves = []
        for child in self.nodes:
            leaves.extend(child.all_leaves())
        return leaves


# ─────────────────────────────────────────────────────────────
# THE FULL DOCUMENT INDEX
# ─────────────────────────────────────────────────────────────

class DocumentIndex(BaseModel):
    """
    The complete PageIndex tree for one document.

    Contains:
        - Document-level metadata
        - The root-level nodes (top chapters/sections)
        - All nested sub-sections inside those nodes

    Example:
        DocumentIndex(
            title       = "Climate Change Report 2024",
            description = "A comprehensive report covering...",
            total_pages = 20,
            nodes       = [<ch1>, <ch2>, <ch3>, <ch4>]
        )
    """

    title       : str             = Field(description="Document title")
    description : str             = Field(description="1-2 sentence description of the whole doc")
    total_pages : int             = Field(description="Total number of pages in the document")
    nodes       : list[IndexNode] = Field(description="Top-level sections/chapters")

    # ── Serialization ─────────────────────────────────────────

    def to_json(self) -> str:
        """Convert the full tree to a JSON string (for saving to disk)"""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "DocumentIndex":
        """Rebuild the tree from a saved JSON string"""
        return cls.model_validate_json(json_str)

    def save(self, path: str):
        """Save the tree index to a .json file"""
        with open(path, "w") as f:
            f.write(self.to_json())
        print(f"[models] Tree index saved to: {path}")

    @classmethod
    def load(cls, path: str) -> "DocumentIndex":
        """Load a previously saved tree index from a .json file"""
        with open(path, "r") as f:
            return cls.from_json(f.read())

    # ── Helpers ───────────────────────────────────────────────

    def all_nodes_flat(self) -> list[IndexNode]:
        """
        Returns ALL nodes in the tree as a flat list.
        Useful for displaying the full tree structure.

        e.g. [root1, child1.1, child1.2, root2, child2.1, ...]
        """
        result = []
        def _recurse(nodes):
            for node in nodes:
                result.append(node)
                _recurse(node.nodes)
        _recurse(self.nodes)
        return result

    def to_text_outline(self) -> str:
        """
        Returns the tree as a readable text outline.
        Used when we pass the tree to the LLM for retrieval.

        Example output:
            [0001] Introduction (pages 1-2)
              Summary: Overview of the report...
              [0002] Background (page 1)
                Summary: Historical context...
        """
        lines = []
        lines.append(f"Document: {self.title}")
        lines.append(f"Description: {self.description}")
        lines.append(f"Total pages: {self.total_pages}")
        lines.append("")

        def _recurse(nodes, depth=0):
            indent = "  " * depth
            for node in nodes:
                lines.append(f"{indent}[{node.node_id}] {node.title} ({node.page_range()})")
                lines.append(f"{indent}  Summary: {node.summary}")
                if node.nodes:
                    _recurse(node.nodes, depth + 1)

        _recurse(self.nodes)
        return "\n".join(lines)


# Allow IndexNode to reference itself (for nested children)
IndexNode.model_rebuild()