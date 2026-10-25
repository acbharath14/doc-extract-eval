# doc-extract-eval

Compare PDF text extractors (pypdf, pdfplumber, PyMuPDF, Tesseract OCR,
Mistral cloud OCR) on real documents, then pipe the output through an LLM
for structured business summaries. Upload a doc, watch each stage, see which
extractor wins.

This demos the classic document-AI flow: **extract → clean → chunk →
summarize → structured data** — the same approach behind invoice processing,
contract review, and report digestion pipelines.

## Quickstart

```bash
pip install -r requirements.txt

# Generate sample PDFs
python samples/text_pdf.py
python samples/invoice_pdf.py

# Compare extractors on a PDF
python -m doc_extract.runner compare samples/business_report.pdf

# Include Tesseract and Mistral OCR in the comparison
python -m doc_extract.runner compare samples/invoice.pdf --ocr --mistral --output reports/invoice.json

# Score extractors against expected text (winner picked by F1, not char count)
python -m doc_extract.runner compare samples/invoice.pdf --ground-truth samples/invoice.txt

# Full pipeline: extract → summarize with Ollama (needs `ollama serve`)
python -m doc_extract.runner pipeline samples/invoice.pdf --output reports/invoice.json

# Extraction only, no LLM
python -m doc_extract.runner pipeline samples/invoice.pdf --no-llm

# Single extractor
python -m doc_extract.runner extract samples/invoice.pdf --backend pdfplumber --text

# Run tests
pytest tests/ -v

# Dashboard
streamlit run dashboard.py
```

No `PYTHONPATH` needed — `pytest.ini` sets `pythonpath = src`.

## OCR setup

**Tesseract (local, free).** `pytesseract` on pip is only a wrapper — you
also need the `tesseract` and `poppler` system binaries:

```bash
conda install -c conda-forge tesseract poppler
```

(Homebrew dropped Intel Macs, so conda-forge is the reliable source there.)
If the binaries aren't on your PATH after install, add conda's bin directory:

```bash
export PATH="$PATH:$(conda info --base)/bin"
```

To make it permanent, add that line to your `~/.zshrc` (or `~/.bashrc`).

**Mistral OCR (cloud, free tier).** Get a key at
[console.mistral.ai](https://console.mistral.ai), then:

```bash
export MISTRAL_API_KEY=your-key-here
```

Note: `requirements.txt` pins `mistralai>=1.0,<2.0` — v3 restructured the
entire SDK API. The free tier is rate-limited; if you hit 429s, wait a few
minutes and retry, or check your quota in the Mistral console.

## What it does

**1. Compare extractors.** Runs pypdf, pdfplumber, and PyMuPDF on your PDF
and scores each on characters extracted, tables found, speed, and warnings.
Tesseract OCR runs automatically when all text extractors find <50 chars/page
(likely a scanned PDF), or when forced with `--ocr`. Mistral cloud OCR is
available too — opt-in via the dashboard checkbox or `include_mistral=True`
(needs `MISTRAL_API_KEY`; free tier at console.mistral.ai). The winner is picked
by evidence: most characters, then most tables, then speed.

**Ground-truth scoring.** Drop expected text in a `<name>.txt` next to the PDF
(or paste it in the dashboard, or pass `--ground-truth file.txt`), and each
extractor is scored on word-level precision / recall / F1. With ground truth,
the winner is picked by **F1** — accuracy over verbosity.

**2. Run the pipeline.** Auto-selects the best extractor, cleans the text
(whitespace, de-hyphenation, page-number stripping), chunks it, then asks a
local Ollama model for a business summary plus structured fields:

```json
{
  "summary": "Q3 revenue grew 22.8% year-over-year…",
  "key_points": ["Enterprise division led growth", …],
  "entities": {"dates": ["October 31, 2026"], "amounts": ["$12,400.00"], …},
  "document_type": "invoice"
}
```

Every run records provenance: which extractor won and why, which model was
used, per-stage timings.

**3. Visualize.** The Streamlit dashboard gives you side-by-side extractor
comparison, per-extractor text tabs, and the LLM summary with entities —
plus one-click sample generation.

## The extractors

| Extractor | Speed | Tables | Best for |
|---|---|---|---|
| pypdf | Fast | No | Born-digital text PDFs |
| pdfplumber | Slower | Best | Invoices, reports with tables |
| PyMuPDF | Fastest | Basic | Large PDFs, layout fidelity |
| Tesseract OCR | Slow | No | Scanned / image-only PDFs, offline |
| Mistral OCR | API-bound | Parsed from markdown | Scanned PDFs, best accuracy (needs API key) |

See [METHODOLOGY.md](METHODOLOGY.md) for when to use which and how the
metrics work.

## Honest limits

- **OCR accuracy depends on scan quality.** 200+ DPI, straight pages, good
  contrast. Bad scans → bad text, no matter the engine.
- **Table extraction is heuristic.** Merged cells, nested tables, and
  borderless layouts confuse every extractor. Verify against the source.
- **Entity regex is approximate.** The name finder will catch some false
  positives ("Contact Jane Smith"). It's a starting point, not NER.
- **Document classifier is keyword-based.** Ties and jargon-heavy docs
  return "unknown" rather than guessing wrong.
- **LLM summaries need a local Ollama.** Without `ollama serve`, the
  pipeline returns extraction results with `"summary": null` and a clear
  error — it never crashes, but there's no summary.
- **Mistral OCR needs quota.** The free tier is rate-limited; 429s are
  reported as warnings, not crashes. Check your quota in the Mistral console.

## CI

No workflow is shipped yet — this repo has no `.github` directory, so
nothing runs on PRs. The workflow text to paste via the GitHub web UI is
in the hygiene PR description (workflow files cannot be pushed via the
API).

## Graceful degradation

Nothing in the pipeline crashes on a missing dependency. One bad extractor
can't kill a comparison, a missing API key becomes a warning, and a dead
Ollama still returns extraction results. Every failure is recorded in the
report's `warnings` / `llm_error` fields — inspectable, not silent.

## Project structure

```
├── src/doc_extract/
│   ├── extractors.py   # pypdf / pdfplumber / pymupdf / tesseract / mistral_ocr, one interface
│   ├── metrics.py      # scoring + winner selection
│   ├── pipeline.py     # extract → clean → chunk → summarize → JSON
│   ├── summarizer.py   # Ollama client, entity regex, doc classifier
│   └── runner.py       # CLI: compare | pipeline | extract
├── samples/
│   ├── text_pdf.py     # generates 3-page business report
│   ├── invoice_pdf.py  # generates 2-page invoice with tables
│   └── README.md       # how to make a scanned PDF for OCR demo
├── tests/              # 60 tests, no network (everything mocked)
├── dashboard.py        # Streamlit UI
├── METHODOLOGY.md      # approaches, trade-offs, metrics explained
```
