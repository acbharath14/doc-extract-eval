"""Extraction quality metrics and cross-extractor comparison.

compare_extractors(pdf_path) runs every applicable extractor and scores them
on the same dimensions so the "best" choice is evidence, not habit.
"""
from __future__ import annotations

import time

from .extractors import EXTRACTORS, OCR_EXTRACTORS, ExtractionResult
from .scoring import score_against_ground_truth, load_ground_truth


def _words(text: str) -> int:
    return len(text.split())


def _table_cells(tables: list) -> int:
    return sum(len(row) for table in tables for row in table)


def score_result(res: ExtractionResult,
                 ground_truth: str | None = None) -> dict:
    """Per-extractor quality metrics from one ExtractionResult.

    When ground_truth is supplied, word-level precision/recall/F1 are
    included under the "gt" key.
    """
    words = _words(res.text)
    cells = _table_cells(res.tables)
    density = round(res.char_count / res.n_pages, 1) if res.n_pages else 0.0
    out = {
        "extractor": res.extractor,
        "char_count": res.char_count,
        "words": words,
        "n_pages": res.n_pages,
        "n_tables": len(res.tables),
        "table_cells": cells,
        "text_density": density,          # chars per page
        "elapsed_s": round(res.elapsed_s, 3),
        "n_warnings": len(res.warnings),
        "warnings": res.warnings,
        "ocr_confidence": res.ocr_confidence,
    }
    if ground_truth:
        out["gt"] = score_against_ground_truth(res.text, ground_truth)
    return out


def pick_winner(scored: list[dict]) -> dict:
    """Pick the winning extractor.

    With ground truth: highest F1 wins (accuracy over verbosity).
    Without: most characters extracted (coverage).
    Tie-breaks: more tables, then faster, then fewer warnings.
    Returns {"winner": name, "reason": str, "ranked": [names...]}.
    OCR extractors (tesseract, mistral_ocr) are only eligible when the PDF
    looks scanned — otherwise OCR results are informational only.
    """
    if not scored:
        return {"winner": None, "reason": "no results", "ranked": []}

    eligible = [s for s in scored
                if s["extractor"] not in OCR_EXTRACTORS or s.get("_ocr_eligible")]
    pool = eligible or scored  # fall back to everything if nothing eligible

    has_gt = any("gt" in s for s in pool)

    def key(s: dict):
        if has_gt and "gt" in s:
            # F1 first, then recall (coverage of expected text).
            return (s["gt"]["f1"], s["gt"]["recall"],
                    s["n_tables"], -s["elapsed_s"], -s["n_warnings"])
        return (s["char_count"], s["n_tables"], -s["elapsed_s"],
                -s["n_warnings"])

    ranked = sorted(pool, key=key, reverse=True)
    winner = ranked[0]
    notes = []
    if has_gt and "gt" in winner:
        reason = (f"highest F1 vs ground truth ({winner['gt']['f1']:.3f}; "
                  f"P={winner['gt']['precision']:.3f}, "
                  f"R={winner['gt']['recall']:.3f})")
    else:
        reason = (f"most characters ({winner['char_count']:,}), "
                  f"{winner['n_tables']} tables, {winner['elapsed_s']}s")
    if (len(ranked) > 1 and not has_gt
            and ranked[1]["char_count"] == winner["char_count"]):
        notes.append(f"tied on chars with {ranked[1]['extractor']}; "
                     f"tables broke the tie ({winner['n_tables']} vs "
                     f"{ranked[1]['n_tables']})")
    if notes:
        reason += "; " + "; ".join(notes)
    return {"winner": winner["extractor"], "reason": reason,
            "ranked": [s["extractor"] for s in ranked],
            "scored_by": "ground_truth_f1" if has_gt else "char_count"}


def compare_extractors(pdf_path: str, include_ocr: bool = False,
                       include_mistral: bool = False,
                       ground_truth: str | None = None,
                       gt_path: str | None = None) -> dict:
    """Run all text extractors on pdf_path; add OCR when forced or needed.

    Local OCR (tesseract) runs when include_ocr=True, or automatically when
    every text extractor finds <50 chars/page (likely scanned). Cloud OCR
    (mistral_ocr) runs only when include_mistral=True and MISTRAL_API_KEY
    is set — it costs API credit, so it is never automatic.

    Ground truth: pass expected text directly, or a gt_path to a .txt file.
    When absent, a sibling <basename>.txt is auto-loaded if present. With
    ground truth, the winner is picked by F1 instead of char count.

    Returns:
      {"results": {name: metrics}, "winner": {...}, "elapsed_s_total": ...}
    """
    t0 = time.perf_counter()
    results: dict[str, ExtractionResult] = {}
    for name in ("pypdf", "pdfplumber", "pymupdf"):
        try:
            results[name] = EXTRACTORS[name](pdf_path)
        except (FileNotFoundError, RuntimeError):
            raise
        except Exception as e:  # one bad extractor must not kill comparison
            results[name] = ExtractionResult(
                name, "", 0, warnings=[f"extractor crashed: {e}"])

    # Decide whether local OCR is warranted.
    densities = [r.char_count / r.n_pages for r in results.values()
                 if r.n_pages]
    scanned_like = bool(densities) and all(d < 50 for d in densities)
    run_ocr = include_ocr or scanned_like

    ocr_error: str | None = None
    if run_ocr:
        try:
            results["tesseract"] = EXTRACTORS["tesseract"](pdf_path)
        except RuntimeError as e:
            ocr_error = str(e)  # missing binaries → note, not crash
        except FileNotFoundError:
            raise
        except Exception as e:
            results["tesseract"] = ExtractionResult(
                "tesseract", "", 0, warnings=[f"OCR crashed: {e}"])

    mistral_error: str | None = None
    if include_mistral:
        try:
            results["mistral_ocr"] = EXTRACTORS["mistral_ocr"](pdf_path)
        except RuntimeError as e:
            mistral_error = str(e)  # missing API key → note, not crash
        except FileNotFoundError:
            raise
        except Exception as e:
            results["mistral_ocr"] = ExtractionResult(
                "mistral_ocr", "", 0,
                warnings=[f"Mistral OCR crashed: {e}"])

    scored = []
    if ground_truth is None:
        ground_truth = load_ground_truth(pdf_path, gt_path)
    for name, res in results.items():
        s = score_result(res, ground_truth)
        s["_ocr_eligible"] = (name in OCR_EXTRACTORS and scanned_like)
        scored.append(s)

    out = {
        "results": {s["extractor"]: {k: v for k, v in s.items()
                                    if not k.startswith("_")}
                    for s in scored},
        "winner": pick_winner(scored),
        "ocr_attempted": run_ocr,
        "ocr_error": ocr_error,
        "mistral_attempted": include_mistral,
        "mistral_error": mistral_error,
        "scanned_like": scanned_like,
        "ground_truth_used": bool(ground_truth),
        "elapsed_s_total": round(time.perf_counter() - t0, 3),
    }
    return out
