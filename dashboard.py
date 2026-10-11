"""Streamlit dashboard for doc-extract-eval.

Upload a PDF → run all extractors side-by-side → inspect text per extractor
→ run the LLM pipeline for a structured business summary.

Run:  streamlit run dashboard.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from doc_extract.metrics import compare_extractors
from doc_extract.pipeline import run_pipeline
from doc_extract.extractors import EXTRACTORS

st.set_page_config(page_title="Doc Extract Eval", layout="wide")
st.title("Document Extraction Eval")
st.caption("Compare PDF extractors, then summarize with an LLM.")

ROOT = Path(__file__).parent

# --- Sample generation ---

with st.sidebar:
    st.header("Samples")
    st.caption("Generate the bundled sample PDFs on the fly.")
    col_a, col_b = st.sidebar.columns(2)
    report_path: Path | None = None
    invoice_path: Path | None = None
    if col_a.button("Report PDF"):
        report_path = ROOT / "samples" / "business_report.pdf"
        r = subprocess.run([sys.executable, str(ROOT / "samples" / "text_pdf.py"),
                            "--out", str(report_path)],
                           capture_output=True, text=True)
        if r.returncode == 0:
            st.sidebar.success(f"Generated {report_path.name}")
        else:
            st.sidebar.error(r.stderr[:200])
    if col_b.button("Invoice PDF"):
        invoice_path = ROOT / "samples" / "invoice.pdf"
        r = subprocess.run([sys.executable, str(ROOT / "samples" / "invoice_pdf.py"),
                            "--out", str(invoice_path)],
                           capture_output=True, text=True)
        if r.returncode == 0:
            st.sidebar.success(f"Generated {invoice_path.name}")
        else:
            st.sidebar.error(r.stderr[:200])

# --- PDF input ---

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
pdf_path: str | None = None
if uploaded:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(uploaded.read())
    tmp.close()
    pdf_path = tmp.name
    st.caption(f"Uploaded: {uploaded.name}")
elif (ROOT / "samples" / "business_report.pdf").exists():
    if st.checkbox("Use generated business_report.pdf", value=False):
        pdf_path = str(ROOT / "samples" / "business_report.pdf")

if not pdf_path:
    st.info("Upload a PDF (or generate a sample from the sidebar) to begin.")
    st.stop()

# --- Extractor comparison ---

force_ocr = st.checkbox("Force Tesseract OCR", value=False,
                        help="Run OCR even when text extractors succeed.")
if st.button("Compare extractors", type="primary"):
    with st.spinner("Running pypdf, pdfplumber, PyMuPDF…"):
        try:
            comp = compare_extractors(pdf_path, include_ocr=force_ocr)
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()
    st.session_state["comparison"] = comp

    # Keep raw texts for the per-extractor tabs.
    texts: dict[str, str] = {}
    for name in comp["results"]:
        try:
            texts[name] = EXTRACTORS[name](pdf_path).text[:8000]
        except Exception as e:
            texts[name] = f"(extraction failed: {e})"
    st.session_state["texts"] = texts

comp = st.session_state.get("comparison")
if comp:
    st.subheader("Extractor comparison")
    winner = comp["winner"]["winner"]
    rows = []
    for name, m in comp["results"].items():
        rows.append({
            "extractor": name + (" 🏆" if name == winner else ""),
            "chars": f"{m['char_count']:,}",
            "words": f"{m['words']:,}",
            "tables": m["n_tables"],
            "table cells": m["table_cells"],
            "chars/page": m["text_density"],
            "time (s)": m["elapsed_s"],
            "warnings": m["n_warnings"],
            "OCR conf.": m["ocr_confidence"] if m["ocr_confidence"] else "—",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption(f"Winner: **{winner}** — {comp['winner']['reason']}")
    if comp["ocr_error"]:
        st.warning(f"OCR skipped: {comp['ocr_error']}")
    elif comp["scanned_like"]:
        st.info("PDF looks scanned (all text extractors found <50 chars/page) — "
                "OCR ran automatically.")

    st.subheader("Extracted text per extractor")
    tabs = st.tabs(list(st.session_state["texts"]))
    for tab, (name, text) in zip(tabs, st.session_state["texts"].items()):
        with tab:
            st.text_area(f"{name} output", text, height=300,
                         key=f"text_{name}", disabled=True)

# --- LLM pipeline ---

st.subheader("LLM pipeline")
st.caption("Extract → clean → chunk → summarize with Ollama → structured JSON. "
           "Needs `ollama serve` running locally.")
model = st.text_input("Ollama model", value="qwen3:8b")
if st.button("Run LLM pipeline"):
    with st.spinner(f"Summarizing with {model}… (this can take a minute)"):
        import os
        os.environ["OLLAMA_MODEL"] = model
        try:
            result = run_pipeline(pdf_path, force_ocr=force_ocr)
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

    prov = result["provenance"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Extractor", prov["extractor"])
    c2.metric("Document type", result["document_type"])
    c3.metric("Pipeline time", f"{result['stages']['total_s']}s")

    if result["llm_error"]:
        st.warning(f"LLM unavailable: {result['llm_error']}")
        st.info("Extraction results below are still valid — only the "
                "summary step was skipped.")
    else:
        st.subheader("Summary")
        st.write(result["summary"])
        if result["key_points"]:
            st.subheader("Key points")
            for kp in result["key_points"]:
                st.markdown(f"- {kp}")

    st.subheader("Entities")
    st.json(result["entities"])

    with st.expander("Full pipeline JSON"):
        slim = {k: v for k, v in result.items() if k != "comparison"}
        st.json(slim)
