"""End-to-end pipeline: extract → clean → chunk → summarize → structured JSON.

run_pipeline(pdf_path) auto-selects the best extractor, cleans and chunks
the text, then asks an LLM (Ollama) for a business summary plus structured
fields. If Ollama is down, extraction results are still returned with
"summary": None and a clear "llm_error" — the pipeline never crashes on
LLM failure.
"""
from __future__ import annotations

import datetime
import os
import re
import time

from .extractors import ExtractionResult
from .metrics import compare_extractors
from .summarizer import (
    summarize_ollama,
    parse_summary_json,
    extract_entities,
    classify_document,
)


def clean_text(text: str) -> str:
    """Normalize whitespace, de-hyphenate, drop obvious header/footer noise."""
    # De-hyphenate line-break splits: "docu-\nment" -> "document"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    # Collapse whitespace, keep paragraph breaks.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    lines = []
    for line in text.split("\n"):
        s = line.strip()
        # Drop lone page numbers and repeated short headers/footers.
        if re.fullmatch(r"\d{1,3}", s):
            continue
        if re.fullmatch(r"Page \d+ of \d+", s, re.IGNORECASE):
            continue
        lines.append(s)
    return "\n".join(lines).strip()


def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    """Split into overlapping chunks (default 500 chars, 50 overlap)."""
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def run_pipeline(pdf_path: str, summarizer: str = "ollama",
                 force_ocr: bool = False, use_mistral_ocr: bool = False,
                 model: str | None = None) -> dict:
    """Full pipeline. Returns dict with provenance + per-stage timings.

    summarizer: "ollama" (default) or "none" (skip LLM, extraction only).
    """
    t_start = time.perf_counter()
    stages: dict[str, float] = {}

    # 1. Extract (auto-select best).
    t0 = time.perf_counter()
    comparison = compare_extractors(pdf_path, include_ocr=force_ocr,
                                    include_mistral=use_mistral_ocr)
    winner_name = comparison["winner"]["winner"]
    winner_metrics = comparison["results"].get(winner_name, {})
    stages["extract_s"] = round(time.perf_counter() - t0, 3)

    # Recover the winning text: re-run is wasteful; comparison already ran
    # extractors, but metrics only keeps text. Re-extract with the winner
    # for the full text (cheap relative to OCR/LLM).
    from .extractors import EXTRACTORS
    t0 = time.perf_counter()
    raw_text, tables = "", []
    if winner_name and winner_name in EXTRACTORS:
        try:
            full: ExtractionResult = EXTRACTORS[winner_name](pdf_path)
            raw_text = full.text
            tables = full.tables
        except RuntimeError as e:  # e.g. tesseract binaries missing
            winner_metrics = {**winner_metrics, "ocr_error": str(e)}
        except Exception as e:  # noqa: BLE001 — winner must not kill pipeline
            winner_metrics = {**winner_metrics,
                              "extract_error": f"{winner_name} crashed: {e}"}
    else:
        winner_metrics = {**winner_metrics,
                          "extract_error": "no usable extractor winner"}
    stages["extract_full_s"] = round(time.perf_counter() - t0, 3)

    # 2. Clean.
    t0 = time.perf_counter()
    cleaned = clean_text(raw_text)
    stages["clean_s"] = round(time.perf_counter() - t0, 3)

    # 3. Chunk.
    t0 = time.perf_counter()
    chunks = chunk_text(cleaned)
    stages["chunk_s"] = round(time.perf_counter() - t0, 3)

    # Deterministic fields (no LLM needed).
    entities = extract_entities(cleaned)
    doc_type = classify_document(cleaned)

    # 4-5. Summarize + structure.
    llm_error: str | None = None
    summary: str | None = None
    key_points: list[str] = []
    model_used = model or os.environ.get("OLLAMA_MODEL", "qwen3:8b")
    t0 = time.perf_counter()
    if summarizer == "ollama" and cleaned.strip():
        try:
            raw = summarize_ollama(cleaned, model=model_used)
            parsed = parse_summary_json(raw)
            summary = parsed["summary"]
            key_points = parsed["key_points"]
        except RuntimeError as e:
            llm_error = str(e)
    elif not cleaned.strip():
        llm_error = "no text extracted; nothing to summarize"
    stages["summarize_s"] = round(time.perf_counter() - t0, 3)

    stages["total_s"] = round(time.perf_counter() - t_start, 3)

    return {
        "provenance": {
            "pdf": os.path.basename(pdf_path),
            "extractor": winner_name,
            "extractor_reason": comparison["winner"]["reason"],
            "ocr_attempted": comparison["ocr_attempted"],
            "mistral_attempted": comparison["mistral_attempted"],
            "mistral_error": comparison.get("mistral_error"),
            "scanned_like": comparison["scanned_like"],
            "model": model_used if summarizer == "ollama" else None,
            "timestamp": datetime.datetime.now(
                datetime.timezone.utc).isoformat(),
            "n_chunks": len(chunks),
        },
        "extraction": {
            "char_count": len(cleaned),
            "words": len(cleaned.split()),
            "n_tables": len(tables),
            "tables": tables[:5],  # cap for report size
        },
        "document_type": doc_type,
        "entities": entities,
        "summary": summary,
        "key_points": key_points,
        "llm_error": llm_error,
        "comparison": comparison,
        "stages": stages,
    }
