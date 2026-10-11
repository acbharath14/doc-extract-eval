"""Tests for metrics: scoring, winner selection, compare_extractors."""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from doc_extract.extractors import ExtractionResult
from doc_extract.metrics import score_result, pick_winner, compare_extractors


def _res(name, text, tables=None, elapsed=0.1, warnings=None):
    return ExtractionResult(name, text, 2, tables=tables or [],
                            elapsed_s=elapsed,
                            warnings=warnings or [])


def test_score_result_fields():
    r = _res("pypdf", "hello world foo", tables=[[["a", "b"], ["c", "d"]]])
    s = score_result(r)
    assert s["char_count"] == len("hello world foo")
    assert s["words"] == 3
    assert s["n_tables"] == 1
    assert s["table_cells"] == 4
    assert s["text_density"] == round(len("hello world foo") / 2, 1)


def test_pick_winner_most_chars():
    a = score_result(_res("pypdf", "x" * 100))
    b = score_result(_res("pdfplumber", "x" * 200))
    w = pick_winner([a, b])
    assert w["winner"] == "pdfplumber"
    assert "pdfplumber" in w["ranked"][0]


def test_pick_winner_tie_break_tables():
    a = score_result(_res("pypdf", "x" * 100, tables=[]))
    b = score_result(_res("pdfplumber", "x" * 100,
                          tables=[[["a"], ["b"]]]))
    w = pick_winner([a, b])
    assert w["winner"] == "pdfplumber"
    assert "tie" in w["reason"].lower()


def test_pick_winner_empty():
    w = pick_winner([])
    assert w["winner"] is None


def test_compare_extractors_on_generated_pdf(tmp_path):
    sys.path.insert(0, str(Path(__file__).parent.parent / "samples"))
    from text_pdf import build
    pdf = tmp_path / "r.pdf"
    build(pdf)
    out = compare_extractors(str(pdf))
    assert set(out["results"]) == {"pypdf", "pdfplumber", "pymupdf"}
    assert out["winner"]["winner"] in ("pypdf", "pdfplumber", "pymupdf")
    assert out["elapsed_s_total"] >= 0
    assert out["ocr_attempted"] is False  # text PDF, no OCR needed


def test_compare_extractors_missing_file():
    with pytest.raises(FileNotFoundError):
        compare_extractors("/nope/missing.pdf")


def test_compare_triggers_ocr_for_scanned_like(tmp_path):
    # Blank PDF → all extractors find ~0 chars → OCR attempted.
    from fpdf import FPDF
    p = tmp_path / "blank.pdf"
    FPDF().output(str(p))
    with patch("doc_extract.metrics.EXTRACTORS") as mock_ext:
        # Fake extractors returning empty results.
        def fake(name):
            def _fn(path):
                return ExtractionResult(name, "", 1, elapsed_s=0.01)
            return _fn
        mock_ext.__getitem__.side_effect = lambda n: fake(n)
        # Tesseract will raise RuntimeError (no binaries) → ocr_error noted.
        out = compare_extractors(str(p))
    assert out["scanned_like"] is True
    assert out["ocr_attempted"] is True
