"""CLI for doc-extract-eval.

  python -m doc_extract.runner compare doc.pdf [--ocr] [--mistral] [--output out.json]
  python -m doc_extract.runner pipeline doc.pdf [--no-llm] [--model qwen3:8b] [--output out.json]
  python -m doc_extract.runner extract doc.pdf --backend pdfplumber
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .extractors import EXTRACTORS, ExtractionResult
from .metrics import compare_extractors
from .pipeline import run_pipeline


def _result_to_json(res: ExtractionResult) -> dict:
    return {
        "extractor": res.extractor,
        "char_count": res.char_count,
        "n_pages": res.n_pages,
        "n_tables": len(res.tables),
        "elapsed_s": round(res.elapsed_s, 3),
        "warnings": res.warnings,
        "ocr_confidence": res.ocr_confidence,
        "text_preview": res.text[:500],
    }


def cmd_compare(args: argparse.Namespace) -> int:
    try:
        out = compare_extractors(args.pdf, include_ocr=args.ocr,
                                 include_mistral=args.mistral)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    # Slim the JSON: drop full warnings lists per extractor into counts.
    slim = {
        "winner": out["winner"],
        "ocr_attempted": out["ocr_attempted"],
        "ocr_error": out["ocr_error"],
        "mistral_attempted": out["mistral_attempted"],
        "mistral_error": out["mistral_error"],
        "scanned_like": out["scanned_like"],
        "elapsed_s_total": out["elapsed_s_total"],
        "results": out["results"],
    }
    print(json.dumps(slim, indent=2))
    if args.output:
        p = Path(args.output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    return 0


def cmd_pipeline(args: argparse.Namespace) -> int:
    try:
        out = run_pipeline(args.pdf,
                           summarizer="none" if args.no_llm else "ollama",
                           force_ocr=args.ocr,
                           use_mistral_ocr=args.mistral,
                           model=args.model)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    # Don't dump the whole comparison blob to stdout; keep the report.
    stdout_view = {k: v for k, v in out.items() if k != "comparison"}
    stdout_view["comparison_summary"] = {
        "winner": out["comparison"]["winner"],
        "ocr_attempted": out["comparison"]["ocr_attempted"],
    }
    print(json.dumps(stdout_view, indent=2)[:6000])
    if args.output:
        p = Path(args.output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Full report written to {args.output}", file=sys.stderr)
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    try:
        res = EXTRACTORS[args.backend](args.pdf)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(json.dumps(_result_to_json(res), indent=2))
    if args.text:
        print("--- TEXT ---")
        print(res.text[:4000])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Document extraction eval harness")
    sub = parser.add_subparsers(dest="command", required=True)

    p_cmp = sub.add_parser("compare", help="compare all extractors on a PDF")
    p_cmp.add_argument("pdf")
    p_cmp.add_argument("--ocr", action="store_true",
                       help="force Tesseract OCR even for text PDFs")
    p_cmp.add_argument("--mistral", action="store_true",
                       help="include Mistral cloud OCR (needs MISTRAL_API_KEY)")
    p_cmp.add_argument("--output", default=None)
    p_cmp.set_defaults(func=cmd_compare)

    p_pipe = sub.add_parser("pipeline", help="full extract → summarize pipeline")
    p_pipe.add_argument("pdf")
    p_pipe.add_argument("--no-llm", action="store_true",
                        help="skip LLM summarization (extraction only)")
    p_pipe.add_argument("--ocr", action="store_true")
    p_pipe.add_argument("--mistral", action="store_true",
                        help="include Mistral cloud OCR (needs MISTRAL_API_KEY)")
    p_pipe.add_argument("--model", default=None,
                        help="Ollama model (default: $OLLAMA_MODEL or qwen3:8b)")
    p_pipe.add_argument("--output", default=None)
    p_pipe.set_defaults(func=cmd_pipeline)

    p_ext = sub.add_parser("extract", help="single extractor run")
    p_ext.add_argument("pdf")
    p_ext.add_argument("--backend",
                       choices=sorted(EXTRACTORS), default="pdfplumber")
    p_ext.add_argument("--text", action="store_true",
                       help="also print extracted text")
    p_ext.set_defaults(func=cmd_extract)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
