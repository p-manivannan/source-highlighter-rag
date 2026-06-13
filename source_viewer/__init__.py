"""Streamlit bridge for the interactive PDF source viewer."""

from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

import streamlit.components.v1 as components


COMPONENT_DIR = Path(__file__).resolve().parent
FRONTEND_DIST = COMPONENT_DIR / "frontend" / "dist"

_citation_component = components.declare_component(
    "citation_component",
    path=FRONTEND_DIST,
)


@lru_cache(maxsize=16)
def _encoded_pdf(path: str, modified_ns: int) -> str:
    """Return a cached base64 representation, invalidated when the file changes."""
    del modified_ns
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def resolve_pdf(data_dir: Path, source_file: str) -> Path:
    """Resolve a citation filename without allowing traversal outside data_dir."""
    root = data_dir.resolve()
    candidate = (root / source_file).resolve()

    if candidate.parent != root:
        raise ValueError("The cited PDF path is outside the configured data directory.")
    if candidate.suffix.casefold() != ".pdf":
        raise ValueError("The cited source is not a PDF.")
    if not candidate.is_file():
        raise FileNotFoundError(f"The cited PDF does not exist: {source_file}")

    return candidate


def render_source_viewer(
    *,
    data_dir: Path,
    source_file: str,
    page_number: int,
    chunk_text: str,
    selection_key: str,
    highlight_mode: str = "exact_quote",
    height: int = 820,
) -> None:
    """Load a validated local PDF and mount the React viewer."""
    pdf_path = resolve_pdf(data_dir, source_file)
    modified_ns = pdf_path.stat().st_mtime_ns
    pdf_base64 = _encoded_pdf(str(pdf_path), modified_ns)

    _citation_component(
        view="source_viewer",
        pdf_base64=pdf_base64,
        document_name=source_file,
        page_number=max(1, int(page_number)),
        chunk_text=chunk_text,
        selection_key=selection_key,
        highlight_mode=highlight_mode,
        key="interactive-pdf-source-viewer",
        height=height,
    )


def render_citation_answer(
    *,
    claims: list[dict],
    citations: list[dict],
    component_key: str,
) -> dict | None:
    """Render claim text with inline citations and return the latest click event."""
    return _citation_component(
        view="citation_answer",
        claims=claims,
        citations=citations,
        key=component_key,
        default=None,
    )
