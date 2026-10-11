"""Generate a 3-page business report PDF (text-based, with a table).

Usage:  python samples/text_pdf.py [--out path]
Requires: fpdf2
"""
from __future__ import annotations

import argparse
from pathlib import Path

from fpdf import FPDF


class ReportPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "Acme Corp  *  Quarterly Business Review  *  Q3 2026",
                  align="C")
        self.ln(12)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()} of {{nb}}", align="C")

    def heading(self, text: str, level: int = 1):
        sizes = {1: 18, 2: 14, 3: 11}
        self.set_font("Helvetica", "B", sizes[level])
        self.set_text_color(20, 60, 120)
        self.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(3)

    def kpi_table(self, rows: list[list[str]]):
        self.set_font("Helvetica", "B", 10)
        self.set_fill_color(20, 60, 120)
        self.set_text_color(255, 255, 255)
        widths = [70, 35, 35, 50]
        for i, cell in enumerate(rows[0]):
            self.cell(widths[i], 8, cell, border=1, fill=True, align="C")
        self.ln()
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        fill = False
        for row in rows[1:]:
            if fill:
                self.set_fill_color(235, 242, 250)
            for i, cell in enumerate(row):
                self.cell(widths[i], 7, cell, border=1, fill=fill,
                          align="C" if i else "L")
            self.ln()
            fill = not fill
        self.ln(4)


LOREM = (
    "Acme Corporation delivered a strong third quarter, with revenue growing "
    "across all three business segments. The enterprise division led the way, "
    "driven by multi-year contract renewals and expansion within existing "
    "accounts. Management attributes the performance to disciplined execution "
    "and continued investment in the platform."
)

PARA2 = (
    "Operating margin expanded by two hundred basis points year-over-year as "
    "automation initiatives reduced manual processing costs. The company "
    "onboarded forty-two new enterprise customers during the quarter, "
    "including three Fortune 500 accounts. Churn remained below two percent, "
    "the lowest level in company history."
)

PARA3 = (
    "Looking ahead to the fourth quarter, management expects seasonal strength "
    "in the retail segment and continued momentum in enterprise. Guidance for "
    "full-year revenue was raised to between $48.2M and $49.5M. The board "
    "approved a $5M share repurchase program, reflecting confidence in the "
    "company's cash generation."
)


def build(out_path: Path) -> Path:
    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # Page 1 -- title + executive summary.
    pdf.add_page()
    pdf.ln(20)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 14, "Quarterly Business Review", new_x="LMARGIN",
             new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, "Q3 2026  *  Acme Corporation", new_x="LMARGIN",
             new_y="NEXT", align="C")
    pdf.ln(10)
    pdf.heading("Executive Summary", level=2)
    pdf.body(LOREM)
    pdf.body(PARA2)

    # Page 2 -- KPI table + findings.
    pdf.add_page()
    pdf.heading("Key Performance Indicators", level=2)
    pdf.body("Performance against plan for the quarter ended September 30, 2026:")
    pdf.kpi_table([
        ["Metric", "Q3 2026", "Q3 2025", "Change"],
        ["Revenue", "$12.4M", "$10.1M", "+22.8%"],
        ["Gross margin", "71.2%", "69.4%", "+1.8pp"],
        ["Enterprise customers", "312", "270", "+42"],
        ["Net churn", "1.8%", "2.4%", "-0.6pp"],
        ["Cash on hand", "$18.9M", "$14.2M", "+33.1%"],
    ])
    pdf.heading("Findings", level=2)
    pdf.body(PARA3)

    # Page 3 -- methodology + appendix note.
    pdf.add_page()
    pdf.heading("Methodology", level=2)
    pdf.body(
        "Figures in this report are derived from the company's ERP system and "
        "have been reconciled to the general ledger. Year-over-year "
        "comparisons use constant currency. Customer counts exclude trial "
        "accounts with less than 30 days of tenure."
    )
    pdf.heading("Appendix", level=2)
    pdf.body(
        "Detailed segment breakdowns, regional splits, and the full chart "
        "pack are available in the accompanying spreadsheet. Questions about "
        "this report should be directed to investor relations at "
        "ir@acme-corp.example.com."
    )

    pdf.output(str(out_path))
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="samples/business_report.pdf")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    build(out)
    print(f"Wrote {out} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
