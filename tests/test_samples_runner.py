"""Tests for sample PDF generation scripts and the CLI runner."""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

SAMPLES = Path(__file__).parent.parent / "samples"


def test_text_pdf_generates_valid_pdf(tmp_path):
    out = tmp_path / "report.pdf"
    r = subprocess.run([sys.executable, str(SAMPLES / "text_pdf.py"),
                        "--out", str(out)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert out.stat().st_size > 1000
    from pypdf import PdfReader
    assert len(PdfReader(str(out)).pages) == 3


def test_invoice_pdf_generates_valid_pdf(tmp_path):
    out = tmp_path / "invoice.pdf"
    r = subprocess.run([sys.executable, str(SAMPLES / "invoice_pdf.py"),
                        "--out", str(out)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert out.stat().st_size > 1000
    from pypdf import PdfReader
    assert len(PdfReader(str(out)).pages) == 2


def test_runner_compare_cli(tmp_path):
    from samples.text_pdf import build
    pdf = tmp_path / "r.pdf"
    build(pdf)
    from doc_extract.runner import main
    out = tmp_path / "cmp.json"
    code = main(["compare", str(pdf), "--output", str(out)])
    assert code == 0
    import json
    data = json.loads(out.read_text())
    assert "winner" in data
    assert data["winner"]["winner"] in ("pypdf", "pdfplumber", "pymupdf")


def test_runner_extract_cli(tmp_path, capsys):
    from samples.invoice_pdf import build
    pdf = tmp_path / "i.pdf"
    build(pdf)
    from doc_extract.runner import main
    code = main(["extract", str(pdf), "--backend", "pypdf"])
    assert code == 0
    assert "pypdf" in capsys.readouterr().out


def test_runner_missing_file():
    from doc_extract.runner import main
    assert main(["compare", "/nope/missing.pdf"]) == 1
    assert main(["extract", "/nope/missing.pdf"]) == 1
