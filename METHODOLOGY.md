# Methodology

How doc-extract-eval works, why each choice was made, and where the
sharp edges are.

## The extraction problem

PDFs are a presentation format, not a data format. Text inside a PDF is a
set of positioned glyphs — there are no paragraphs, no tables, no reading
order guaranteed. Every extractor is reconstructing structure from geometry,
and they make different trade-offs.

## Extractor approaches

**pypdf** reads the content stream directly. Fast and dependency-light, but
it sees raw text-showing operators with no layout analysis. Good when the
PDF was generated from text (Word → PDF, report generators). No table
support — tables come out as jumbled text runs.

**pdfplumber** builds on pdfminer.six and adds explicit table detection:
it looks for ruling lines (or text alignment, with `text` strategy) to
infer rows and columns. Slowest of the three text extractors, but the only
one I'd trust for invoice line items. Its `extract_tables()` returns
structured rows you can actually compute on.

**PyMuPDF (fitz)** is a full PDF engine in C. Fastest text extraction and
good at preserving reading order via its layout analysis. `find_tables()`
gives basic table detection — better than nothing, worse than pdfplumber
for complex tables.

**Tesseract OCR** doesn't read the PDF at all — pdf2image rasterizes each
page to PNG (needs poppler), then Tesseract recognizes text from pixels.
Slow (seconds per page), accuracy lives and dies by scan quality. Only
worth it when there's no embedded text. We gate it: OCR runs automatically
only when every text extractor finds <50 chars/page, or when forced.

### When to use which

| Situation | Pick |
|---|---|
| Born-digital report, just need text | pypdf (fastest to code) or PyMuPDF (fastest to run) |
| Invoice / statement with tables | pdfplumber |
| Large PDF, speed matters | PyMuPDF |
| Scanned paper, photo of a doc | Tesseract OCR |
| Unknown / mixed batch | Run all, pick the winner by metrics (that's what `compare` does) |

## Winner selection

`pick_winner` ranks extractors by:

1. **Character count** (primary) — coverage is king; an extractor that
   finds more text is usually doing better layout reconstruction.
2. **Table count** (tie-break) — for tabular docs, structured tables
   beat raw text.
3. **Speed** (second tie-break) — faster wins when quality ties.
4. **Warning count** (final tie-break) — fewer extraction failures.

OCR results are only *eligible* to win when the PDF looks scanned
(all text extractors <50 chars/page). Otherwise a slow OCR run with
mediocre text shouldn't beat a fast clean text extraction.

## Text cleaning

`clean_text` does three things:

1. **De-hyphenation** — "docu-\nment" → "document". Line-break hyphens
   are a layout artifact, not content.
2. **Whitespace normalization** — collapse runs, preserve paragraph breaks.
3. **Header/footer stripping** — lone page numbers and "Page N of M"
   patterns are dropped. This is heuristic; fancy headers survive.

## Chunking

500 characters with 50-char overlap. Small enough to fit comfortably in a
local LLM's context alongside the prompt, large enough to keep sentences
intact most of the time. The overlap guards against splitting a key phrase
across a boundary. For very long docs you'd want semantic chunking
(by heading/section) — character chunking is the simple baseline.

## LLM summarization

We prompt the model for JSON with `summary` + `key_points`, temperature 0.1
for determinism, and parse defensively (markdown fences stripped, garbage
falls back to raw text). The 12k-char prompt cap keeps local models happy.

**Why Ollama, not an API:** the whole point is a free, local, private
pipeline. No keys, no per-token cost, docs never leave the machine. The
trade-off is speed — qwen3:8b does ~15-25s per summary chunk on CPU.

## Entity extraction

Regex, not NER. Dates (ISO, US, written-out), currency amounts ($, €, £,
USD/EUR/GBP), and capitalized 2-3 word sequences as name candidates.
It's noisy by design — documented in the README. For production you'd swap
in spaCy or a dedicated NER model; the regex gives you a baseline to beat.

## Document classification

Keyword signals per type (invoice/bill-to/subtotal → invoice; agreement/
whereas/indemnif → contract; …). Ties and zero-signal docs return
"unknown" rather than guessing. Simple, explainable, wrong less often
than a fancier classifier with no training data.

## Eval philosophy

This repo doesn't claim any extractor is "best" universally — it measures
per document and picks by evidence. The honest version of "which extractor
should I use" is "it depends on the document," and the comparison table
is how you find out for yours.
