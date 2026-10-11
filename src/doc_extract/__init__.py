"""doc-extract-eval: compare PDF text extractors and summarize with LLMs.

Pipeline: extract (pypdf / pdfplumber / PyMuPDF / Tesseract OCR)
  → clean → chunk → LLM summarize → structured business JSON.
"""
from .extractors import (
    ExtractionResult,
    EXTRACTORS,
    extract_pypdf,
    extract_pdfplumber,
    extract_pymupdf,
    extract_tesseract,
)
from .metrics import compare_extractors, pick_winner
from .summarizer import (
    summarize_ollama,
    extract_entities,
    classify_document,
)
from .pipeline import run_pipeline, clean_text, chunk_text

__all__ = [
    "ExtractionResult",
    "EXTRACTORS",
    "extract_pypdf",
    "extract_pdfplumber",
    "extract_pymupdf",
    "extract_tesseract",
    "compare_extractors",
    "pick_winner",
    "summarize_ollama",
    "extract_entities",
    "classify_document",
    "run_pipeline",
    "clean_text",
    "chunk_text",
]
