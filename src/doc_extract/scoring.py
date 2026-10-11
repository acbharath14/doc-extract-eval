"""Ground-truth scoring for extractor evaluation.

When the user supplies expected text for a PDF, each extractor's output is
scored with word-level precision / recall / F1 against it. This answers
"who extracted *right*" rather than "who extracted *more*".

All comparisons are on normalized tokens: lowercased, punctuation stripped,
whitespace collapsed. Multisets (Counter) are used so repeated words count.
"""
from __future__ import annotations

import re
from collections import Counter


def normalize_text(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into word tokens."""
    text = text.lower()
    # Keep alphanumerics and apostrophes inside words; drop the rest.
    tokens = re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text)
    return tokens


def score_against_ground_truth(extracted: str, expected: str) -> dict:
    """Word-level precision / recall / F1 of extracted vs expected text.

    Returns {"precision": float, "recall": float, "f1": float,
             "n_expected": int, "n_extracted": int, "n_correct": int}.
    All scores are 0.0–1.0, rounded to 4 decimals.
    """
    exp_tokens = Counter(normalize_text(expected))
    ext_tokens = Counter(normalize_text(extracted))

    n_expected = sum(exp_tokens.values())
    n_extracted = sum(ext_tokens.values())
    n_correct = sum((exp_tokens & ext_tokens).values())

    precision = n_correct / n_extracted if n_extracted else 0.0
    recall = n_correct / n_expected if n_expected else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "n_expected": n_expected,
        "n_extracted": n_extracted,
        "n_correct": n_correct,
    }


def load_ground_truth(pdf_path: str, gt_path: str | None = None) -> str | None:
    """Find expected text for a PDF.

    Priority: explicit gt_path > sibling <basename>.txt > None.
    Returns the file contents, or None when no ground truth is available.
    """
    import os

    candidates: list[str] = []
    if gt_path:
        candidates.append(gt_path)
    base, _ = os.path.splitext(pdf_path)
    candidates.append(base + ".txt")

    for c in candidates:
        if c and os.path.isfile(c):
            try:
                with open(c, encoding="utf-8") as fh:
                    text = fh.read().strip()
                if text:
                    return text
            except (OSError, UnicodeDecodeError):
                continue
    return None
