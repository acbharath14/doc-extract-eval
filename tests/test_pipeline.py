"""Tests for the pipeline: clean, chunk, full run, Ollama-down path."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from doc_extract.pipeline import clean_text, chunk_text, run_pipeline


def test_clean_text_dehyphenates():
    assert clean_text("docu-\nment") == "document"
    assert clean_text("hello   world") == "hello world"


def test_clean_text_drops_page_numbers():
    assert clean_text("Title\n42\nMore text") == "Title\nMore text"
    assert "Page 1 of 3" not in clean_text("Intro\nPage 1 of 3\nBody")


def test_chunk_text_basic():
    chunks = chunk_text("a" * 1200, size=500, overlap=50)
    assert len(chunks) == 3
    assert chunks[0] == "a" * 500
    # Overlap: chunk 2 starts 450 chars in.
    assert chunks[1][:50] == chunks[0][-50:]


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("short") == ["short"]


def _sample_pdf(tmp_path):
    sys.path.insert(0, str(Path(__file__).parent.parent / "samples"))
    from invoice_pdf import build
    p = tmp_path / "inv.pdf"
    build(p)
    return str(p)


def test_pipeline_no_llm(tmp_path):
    out = run_pipeline(_sample_pdf(tmp_path), summarizer="none")
    assert out["document_type"] == "invoice"
    assert out["summary"] is None
    assert out["extraction"]["char_count"] > 500
    assert out["extraction"]["n_tables"] >= 1
    assert "extract_s" in out["stages"]
    assert "total_s" in out["stages"]
    assert out["provenance"]["extractor"] in ("pypdf", "pdfplumber", "pymupdf")
    # Entities found in the invoice.
    assert any("2026" in d for d in out["entities"]["dates"])
    assert len(out["entities"]["amounts"]) > 0


def test_pipeline_ollama_down_graceful(tmp_path):
    import urllib.error
    with patch("urllib.request.urlopen",
               side_effect=urllib.error.URLError("refused")):
        out = run_pipeline(_sample_pdf(tmp_path), summarizer="ollama")
    assert out["summary"] is None
    assert out["llm_error"] is not None
    assert "not reachable" in out["llm_error"]
    # Extraction still worked.
    assert out["extraction"]["char_count"] > 500


def _mock_ollama(text: str):
    from unittest.mock import MagicMock
    m = MagicMock()
    m.read.return_value = json.dumps({"response": text}).encode()
    m.__enter__.return_value = m
    return m


def test_pipeline_ollama_success(tmp_path):
    body = json.dumps({"summary": "Invoice for $X.", "key_points": ["Due Oct 31"]})
    with patch("urllib.request.urlopen", return_value=_mock_ollama(body)):
        out = run_pipeline(_sample_pdf(tmp_path), summarizer="ollama")
    assert out["summary"] == "Invoice for $X."
    assert out["key_points"] == ["Due Oct 31"]
    assert out["llm_error"] is None


def test_pipeline_missing_file():
    with pytest.raises(FileNotFoundError):
        run_pipeline("/nope/missing.pdf", summarizer="none")
