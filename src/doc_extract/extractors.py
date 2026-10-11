"""Unified PDF text extractors.

Five backends behind one interface so results are directly comparable:

- pypdf       — pure Python, fast, good for born-digital text PDFs, no tables
- pdfplumber  — best table extraction, slower, good text
- pymupdf     — fastest overall, good text + layout, basic tables
- tesseract   — local OCR for scanned/image PDFs via pytesseract + pdf2image
- mistral_ocr — cloud OCR via the Mistral API (free tier; needs MISTRAL_API_KEY)

Each extractor returns an ExtractionResult. All handle missing files,
encrypted PDFs, and empty PDFs gracefully.
"""
from __future__ import annotations

import os
import re
import shutil
import time
from dataclasses import dataclass, field


@dataclass
class ExtractionResult:
    extractor: str                     # "pypdf" | "pdfplumber" | "pymupdf" | "tesseract" | "mistral_ocr"
    text: str                          # extracted text (may be "")
    n_pages: int                       # pages processed
    tables: list = field(default_factory=list)  # list of tables; each table is list of rows
    char_count: int = 0                # derived, but stored for convenience
    elapsed_s: float = 0.0
    warnings: list = field(default_factory=list)
    ocr_confidence: float | None = None  # mean word confidence 0-100, tesseract only

    def __post_init__(self) -> None:
        self.char_count = len(self.text)


def _check_file(pdf_path: str) -> None:
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"No such file: {pdf_path}")


def extract_pypdf(pdf_path: str) -> ExtractionResult:
    """Extract text with pypdf (pure Python, fast, no table support)."""
    _check_file(pdf_path)
    from pypdf import PdfReader

    t0 = time.perf_counter()
    warnings: list[str] = []
    try:
        reader = PdfReader(pdf_path)
    except Exception as e:
        return ExtractionResult("pypdf", "", 0, elapsed_s=time.perf_counter() - t0,
                                warnings=[f"could not open PDF: {e}"])
    if reader.is_encrypted:
        return ExtractionResult("pypdf", "", 0, elapsed_s=time.perf_counter() - t0,
                                warnings=["PDF is encrypted; text extraction skipped"])

    parts: list[str] = []
    n_pages = len(reader.pages)
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception as e:
            warnings.append(f"page {i + 1}: extraction failed ({e})")
            txt = ""
        if not txt.strip():
            warnings.append(f"page {i + 1} had no extractable text")
        parts.append(txt)
    text = "\n".join(parts)
    return ExtractionResult("pypdf", text, n_pages,
                            elapsed_s=time.perf_counter() - t0,
                            warnings=warnings)


def extract_pdfplumber(pdf_path: str) -> ExtractionResult:
    """Extract text + tables with pdfplumber (best tables, slower)."""
    _check_file(pdf_path)
    import pdfplumber

    t0 = time.perf_counter()
    warnings: list[str] = []
    tables: list = []
    try:
        pdf = pdfplumber.open(pdf_path)
    except Exception as e:
        return ExtractionResult("pdfplumber", "", 0, elapsed_s=time.perf_counter() - t0,
                                warnings=[f"could not open PDF: {e}"])
    # pdfplumber raises on encrypted PDFs at open or first access.
    try:
        n_pages = len(pdf.pages)
    except Exception as e:
        return ExtractionResult("pdfplumber", "", 0,
                                elapsed_s=time.perf_counter() - t0,
                                warnings=[f"PDF is encrypted or unreadable: {e}"])

    parts: list[str] = []
    for i, page in enumerate(pdf.pages):
        try:
            txt = page.extract_text() or ""
            page_tables = page.extract_tables() or []
        except Exception as e:
            warnings.append(f"page {i + 1}: extraction failed ({e})")
            txt, page_tables = "", []
        if not txt.strip():
            warnings.append(f"page {i + 1} had no extractable text")
        # Keep only non-trivial tables (at least 2 rows, 2 cols).
        for t in page_tables:
            if len(t) >= 2 and any(len(r) >= 2 for r in t):
                tables.append([[c or "" for c in row] for row in t])
        parts.append(txt)
    pdf.close()
    text = "\n".join(parts)
    return ExtractionResult("pdfplumber", text, n_pages, tables=tables,
                            elapsed_s=time.perf_counter() - t0,
                            warnings=warnings)


def extract_pymupdf(pdf_path: str) -> ExtractionResult:
    """Extract text with PyMuPDF (fastest, good layout, basic tables)."""
    _check_file(pdf_path)
    import pymupdf

    t0 = time.perf_counter()
    warnings: list[str] = []
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        return ExtractionResult("pymupdf", "", 0,
                                elapsed_s=time.perf_counter() - t0,
                                warnings=[f"could not open PDF: {e}"])
    if doc.needs_pass:
        doc.close()
        return ExtractionResult("pymupdf", "", 0,
                                elapsed_s=time.perf_counter() - t0,
                                warnings=["PDF is encrypted; text extraction skipped"])

    parts: list[str] = []
    tables: list = []
    n_pages = doc.page_count
    for i, page in enumerate(doc):
        try:
            txt = page.get_text() or ""
            for t in page.find_tables().tables or []:
                rows = t.extract()
                if rows and len(rows) >= 2:
                    tables.append(rows)
        except Exception as e:
            warnings.append(f"page {i + 1}: extraction failed ({e})")
            txt = ""
        if not txt.strip():
            warnings.append(f"page {i + 1} had no extractable text")
        parts.append(txt)
    doc.close()
    text = "\n".join(parts)
    return ExtractionResult("pymupdf", text, n_pages, tables=tables,
                            elapsed_s=time.perf_counter() - t0,
                            warnings=warnings)


def _ocr_binaries_available() -> tuple[bool, list[str]]:
    """Check for tesseract + poppler binaries. Returns (ok, missing)."""
    missing = []
    if shutil.which("tesseract") is None:
        missing.append("tesseract")
    # pdf2image needs pdftoppm (poppler).
    if shutil.which("pdftoppm") is None:
        missing.append("poppler (pdftoppm)")
    return (not missing, missing)


def extract_tesseract(pdf_path: str, max_pages: int = 10,
                      dpi: int = 200) -> ExtractionResult:
    """OCR a PDF with Tesseract via pdf2image.

    Intended for scanned/image PDFs where text extractors find nothing.
    Caps at max_pages for speed. Raises RuntimeError with install hints
    when binaries are missing.
    """
    _check_file(pdf_path)
    ok, missing = _ocr_binaries_available()
    if not ok:
        raise RuntimeError(
            "OCR requires missing binaries: " + ", ".join(missing) + ". "
            "Install with: conda install -c conda-forge tesseract poppler "
            "(or: apt install tesseract-ocr poppler-utils / brew install tesseract poppler)"
        )

    from pdf2image import convert_from_path
    import pytesseract
    from PIL import Image

    t0 = time.perf_counter()
    warnings: list[str] = []
    try:
        images: list[Image.Image] = convert_from_path(
            pdf_path, dpi=dpi, first_page=1, last_page=max_pages)
    except Exception as e:
        return ExtractionResult("tesseract", "", 0,
                                elapsed_s=time.perf_counter() - t0,
                                warnings=[f"could not rasterize PDF: {e}"])

    parts: list[str] = []
    confidences: list[float] = []
    for i, img in enumerate(images):
        try:
            txt = pytesseract.image_to_string(img) or ""
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            confs = [float(c) for c in data.get("conf", [])
                     if isinstance(c, (int, float)) and c >= 0]
            if confs:
                confidences.append(sum(confs) / len(confs))
        except Exception as e:
            warnings.append(f"page {i + 1}: OCR failed ({e})")
            txt = ""
        if not txt.strip():
            warnings.append(f"page {i + 1}: OCR found no text")
        parts.append(txt)

    # Page count: we only rasterized up to max_pages; report honestly.
    try:
        from pypdf import PdfReader
        total_pages = len(PdfReader(pdf_path).pages)
    except Exception:
        total_pages = len(images)
    if total_pages > max_pages:
        warnings.append(f"OCR capped at {max_pages} of {total_pages} pages")

    text = "\n".join(parts)
    mean_conf = (sum(confidences) / len(confidences)) if confidences else None
    return ExtractionResult("tesseract", text, len(images),
                            elapsed_s=time.perf_counter() - t0,
                            warnings=warnings,
                            ocr_confidence=round(mean_conf, 1) if mean_conf else None)


def _parse_markdown_tables(md: str) -> list:
    """Pull GitHub-flavored markdown tables out of OCR markdown.

    Returns a list of tables; each table is a list of rows (lists of cells).
    Separator rows (|---|---|) are skipped. Only tables with >=2 data rows
    are kept, matching the pdfplumber convention.
    """
    tables: list = []
    current: list = []
    in_table = False
    for line in md.split("\n"):
        if re.match(r"^\s*\|.*\|\s*$", line):
            if re.match(r"^\s*\|[\s\-:|]+\|\s*$", line):
                continue  # separator row
            current.append([c.strip() for c in line.strip().strip("|").split("|")])
            in_table = True
        else:
            if in_table and len(current) >= 2:
                tables.append(current)
            current, in_table = [], False
    if in_table and len(current) >= 2:
        tables.append(current)
    return tables


def extract_mistral_ocr(pdf_path: str) -> ExtractionResult:
    """OCR a PDF with the Mistral cloud OCR API.

    Needs the MISTRAL_API_KEY environment variable (free tier available at
    console.mistral.ai). Uploads the PDF, runs mistral-ocr-latest, and
    returns the markdown output as text plus parsed tables.

    Raises RuntimeError with a setup hint when the API key is missing.
    """
    _check_file(pdf_path)
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Mistral OCR needs the MISTRAL_API_KEY environment variable. "
            "Get a free key at https://console.mistral.ai, then export "
            "MISTRAL_API_KEY=<your key>."
        )

    from mistralai import Mistral

    t0 = time.perf_counter()
    warnings: list[str] = []
    client = Mistral(api_key=api_key)
    try:
        with open(pdf_path, "rb") as fh:
            uploaded = client.files.upload(
                file={"file_name": os.path.basename(pdf_path),
                      "content": fh.read()})
        signed = client.files.get_signed_url(file_id=uploaded.id)
        result = client.ocr.process(
            model="mistral-ocr-latest",
            document={"type": "document_url",
                      "document_url": signed.url})
    except Exception as e:
        return ExtractionResult("mistral_ocr", "", 0,
                                elapsed_s=time.perf_counter() - t0,
                                warnings=[f"Mistral OCR API failed: {e}"])

    pages = getattr(result, "pages", []) or []
    md_parts = [getattr(p, "markdown", "") or "" for p in pages]
    text = "\n\n".join(md_parts)
    tables = _parse_markdown_tables(text)
    n_pages = len(pages)
    if not text.strip():
        warnings.append("Mistral OCR returned no text")
    return ExtractionResult("mistral_ocr", text, n_pages, tables=tables,
                            elapsed_s=time.perf_counter() - t0,
                            warnings=warnings)


EXTRACTORS: dict[str, callable] = {
    "pypdf": extract_pypdf,
    "pdfplumber": extract_pdfplumber,
    "pymupdf": extract_pymupdf,
    "tesseract": extract_tesseract,
    "mistral_ocr": extract_mistral_ocr,
}

#: Extractors that do OCR (cloud or local). Treated specially by the
#: winner logic: only eligible when the PDF looks scanned.
OCR_EXTRACTORS = {"tesseract", "mistral_ocr"}
