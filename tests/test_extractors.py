"""Tests for the extractor backends. Uses generated sample PDFs."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from doc_extract.extractors import (
    ExtractionResult,
    extract_pypdf,
    extract_pdfplumber,
    extract_pymupdf,
    extract_tesseract,
    EXTRACTORS,
)

# Generate fixtures once per session.
@pytest.fixture(scope="session")
def report_pdf(tmp_path_factory):
    sys.path.insert(0, str(Path(__file__).parent.parent / "samples"))
    from text_pdf import build
    out = tmp_path_factory.mktemp("pdfs") / "report.pdf"
    build(out)
    return str(out)


@pytest.fixture(scope="session")
def invoice_pdf(tmp_path_factory):
    sys.path.insert(0, str(Path(__file__).parent.parent / "samples"))
    from invoice_pdf import build
    out = tmp_path_factory.mktemp("pdfs") / "invoice.pdf"
    build(out)
    return str(out)


@pytest.fixture(scope="session")
def empty_pdf(tmp_path_factory):
    from fpdf import FPDF
    out = tmp_path_factory.mktemp("pdfs") / "empty.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.output(str(out))
    return str(out)


def _assert_result_shape(res: ExtractionResult, name: str):
    assert isinstance(res, ExtractionResult)
    assert res.extractor == name
    assert isinstance(res.text, str)
    assert isinstance(res.n_pages, int) and res.n_pages >= 0
    assert isinstance(res.tables, list)
    assert res.char_count == len(res.text)
    assert isinstance(res.elapsed_s, float) and res.elapsed_s >= 0
    assert isinstance(res.warnings, list)


def test_pypdf_extracts_report(report_pdf):
    res = extract_pypdf(report_pdf)
    _assert_result_shape(res, "pypdf")
    assert res.n_pages == 3
    assert "Acme Corporation" in res.text
    assert res.char_count > 1000


def test_pdfplumber_extracts_tables(invoice_pdf):
    res = extract_pdfplumber(invoice_pdf)
    _assert_result_shape(res, "pdfplumber")
    assert len(res.tables) >= 1
    # Line-items table has header + 5 items.
    assert len(res.tables[0]) >= 5
    assert "INV-2026-0847" in res.text


def test_pymupdf_extracts_text_and_tables(invoice_pdf):
    res = extract_pymupdf(invoice_pdf)
    _assert_result_shape(res, "pymupdf")
    assert res.char_count > 500
    assert len(res.tables) >= 1


def test_pypdf_no_tables_by_design(invoice_pdf):
    res = extract_pypdf(invoice_pdf)
    assert res.tables == []


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        extract_pypdf("/nonexistent/doc.pdf")
    with pytest.raises(FileNotFoundError):
        extract_pdfplumber("/nonexistent/doc.pdf")
    with pytest.raises(FileNotFoundError):
        extract_pymupdf("/nonexistent/doc.pdf")


def test_empty_pdf_warns(empty_pdf):
    res = extract_pypdf(empty_pdf)
    _assert_result_shape(res, "pypdf")
    assert res.text.strip() == ""
    assert any("no extractable text" in w for w in res.warnings)


def test_extractors_registered():
    assert set(EXTRACTORS) == {"pypdf", "pdfplumber", "pymupdf", "tesseract"}


def test_tesseract_missing_binaries_graceful(tmp_path, monkeypatch):
    # Simulate missing binaries without touching the real PATH check.
    import doc_extract.extractors as ex
    monkeypatch.setattr(ex, "_ocr_binaries_available",
                        lambda: (False, ["tesseract"]))
    # Need a real file for the _check_file gate.
    from fpdf import FPDF
    p = tmp_path / "x.pdf"
    FPDF().output(str(p))
    with pytest.raises(RuntimeError, match="tesseract"):
        extract_tesseract(str(p))
