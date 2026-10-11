"""Generate a 2-page invoice PDF (line-items table, totals, dates, addresses).

Usage:  python samples/invoice_pdf.py [--out path]
Requires: fpdf2
"""
from __future__ import annotations

import argparse
from pathlib import Path

from fpdf import FPDF

ITEMS = [
    ("Professional services -- Q3 platform implementation", "40 hrs", "$185.00", "$7,400.00"),
    ("Annual platform license (Enterprise tier)", "1", "$12,000.00", "$12,000.00"),
    ("Data migration -- legacy CRM records", "1", "$3,250.00", "$3,250.00"),
    ("Training -- administrator workshop (2 days)", "2", "$950.00", "$1,900.00"),
    ("Priority support add-on (quarterly)", "1", "$1,100.00", "$1,100.00"),
]


class InvoicePDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 20)
        self.set_text_color(20, 60, 120)
        self.cell(0, 12, "INVOICE", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(20, 60, 120)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()} of {{nb}}", align="C")

    def addr_block(self, title: str, lines: list[str], x: float):
        self.set_xy(x, self.get_y())
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(60, 60, 60)
        self.cell(90, 6, title, new_x="LMARGIN", new_y="NEXT")
        self.set_x(x)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        for line in lines:
            self.set_x(x)
            self.cell(90, 5.5, line, new_x="LMARGIN", new_y="NEXT")

    def meta_row(self, label: str, value: str):
        self.set_font("Helvetica", "B", 10)
        self.cell(40, 6, label)
        self.set_font("Helvetica", "", 10)
        self.cell(0, 6, value, new_x="LMARGIN", new_y="NEXT")

    def items_table(self):
        widths = [95, 25, 30, 40]
        headers = ["Description", "Qty", "Unit Price", "Amount"]
        self.set_font("Helvetica", "B", 10)
        self.set_fill_color(20, 60, 120)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(widths[i], 8, h, border=1, fill=True,
                      align="C" if i else "L")
        self.ln()
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(30, 30, 30)
        fill = False
        for row in ITEMS:
            if fill:
                self.set_fill_color(235, 242, 250)
            h = 7
            # Description may wrap; use multi_cell for first column.
            x0, y0 = self.get_x(), self.get_y()
            self.multi_cell(widths[0], h, row[0], border=1, fill=fill)
            h_desc = self.get_y() - y0
            self.set_xy(x0 + widths[0], y0)
            for i, cell in enumerate(row[1:], start=1):
                self.cell(widths[i], h_desc, cell, border=1, fill=fill,
                          align="R")
            self.ln()
            fill = not fill
        self.ln(2)

    def totals(self, subtotal: float, tax: float, total: float):
        self.set_x(120)
        for label, val in [("Subtotal:", f"${subtotal:,.2f}"),
                           ("Tax (8.5%):", f"${tax:,.2f}")]:
            self.set_font("Helvetica", "", 10)
            self.cell(35, 6, label, align="R")
            self.cell(35, 6, val, align="R", new_x="LMARGIN", new_y="NEXT")
            self.set_x(120)
        self.set_font("Helvetica", "B", 11)
        self.set_fill_color(20, 60, 120)
        self.set_text_color(255, 255, 255)
        self.set_x(120)
        self.cell(35, 8, "Amount Due:", align="R", fill=True)
        self.cell(35, 8, f"${total:,.2f}", align="R", fill=True,
                  new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(30, 30, 30)


def build(out_path: Path) -> Path:
    pdf = InvoicePDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Addresses.
    y = pdf.get_y()
    pdf.addr_block("From:", ["Acme Corporation", "100 Market Street, Suite 400",
                             "Herndon, VA 20170", "billing@acme-corp.example.com"],
                   x=10)
    pdf.set_y(y)
    pdf.addr_block("Bill To:", ["Globex Industries", "Attn: Accounts Payable",
                                "500 Commerce Blvd", "Reston, VA 20190"],
                   x=110)
    pdf.ln(8)

    # Meta.
    pdf.meta_row("Invoice Number:", "INV-2026-0847")
    pdf.meta_row("Invoice Date:", "October 1, 2026")
    pdf.meta_row("Due Date:", "October 31, 2026")
    pdf.meta_row("Purchase Order:", "PO-55231")
    pdf.ln(6)

    # Line items.
    pdf.items_table()

    subtotal = 7400.00 + 12000.00 + 3250.00 + 1900.00 + 1100.00
    tax = round(subtotal * 0.085, 2)
    total = subtotal + tax
    pdf.totals(subtotal, tax, total)
    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(0, 5, "Payment terms: Net 30. Please remit to the address "
                         "above or wire to the account on file. A 1.5% monthly "
                         "late fee applies to overdue balances.")

    # Page 2 -- remittance slip + notes.
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(20, 60, 120)
    pdf.cell(0, 10, "Remittance Slip", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(30, 30, 30)
    pdf.multi_cell(0, 6,
        "Please detach and return this slip with your payment.\n\n"
        f"Invoice: INV-2026-0847\nAmount Due: ${total:,.2f}\n"
        "Due Date: October 31, 2026\n\n"
        "Make checks payable to Acme Corporation.\n"
        "For questions, contact billing@acme-corp.example.com or "
        "call (703) 555-0142.")
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Notes", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6,
        "Services were performed between July 1 and September 30, 2026, "
        "under Master Services Agreement MSA-2024-0117. The platform license "
        "covers the period October 1, 2026 through September 30, 2027.")

    pdf.output(str(out_path))
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="samples/invoice.pdf")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    build(out)
    print(f"Wrote {out} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
