"""
ReportLab styling, typography, color palettes, and NumberedCanvas for QuantTerminal reports.
"""

from __future__ import annotations
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, HRFlowable

import os
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily

# Register DejaVu Sans for full Unicode glyph support (₹, %, σ, β, ≤, ≥, →, ±)
DEJAVU_DIR = "/usr/share/fonts/truetype/dejavu"
FONT_NORMAL = "DejaVuSans"
FONT_BOLD = "DejaVuSans-Bold"

try:
    if os.path.exists(os.path.join(DEJAVU_DIR, "DejaVuSans.ttf")):
        pdfmetrics.registerFont(TTFont("DejaVuSans", os.path.join(DEJAVU_DIR, "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", os.path.join(DEJAVU_DIR, "DejaVuSans-Bold.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVuSans-Oblique", os.path.join(DEJAVU_DIR, "DejaVuSans-Oblique.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVuSans-BoldOblique", os.path.join(DEJAVU_DIR, "DejaVuSans-BoldOblique.ttf")))
        registerFontFamily("DejaVuSans", normal="DejaVuSans", bold="DejaVuSans-Bold", italic="DejaVuSans-Oblique", boldItalic="DejaVuSans-BoldOblique")
    else:
        FONT_NORMAL = "Helvetica"
        FONT_BOLD = "Helvetica-Bold"
except Exception:
    FONT_NORMAL = "Helvetica"
    FONT_BOLD = "Helvetica-Bold"

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 36.0  # 0.5 inch margins
PRINTABLE_WIDTH = PAGE_WIDTH - 2 * MARGIN
PRINTABLE_HEIGHT = PAGE_HEIGHT - 2 * MARGIN

# Institutional Color Palette
PRIMARY = colors.HexColor("#0F172A")       # Deep Navy
SECONDARY = colors.HexColor("#334155")     # Dark Slate
ACCENT_BLUE = colors.HexColor("#0284C7")   # Bright Cyan/Sky
ACCENT_GREEN = colors.HexColor("#059669")  # Emerald
ACCENT_RED = colors.HexColor("#DC2626")    # Crimson
ACCENT_AMBER = colors.HexColor("#D97706")  # Amber
BG_LIGHT = colors.HexColor("#F8FAFC")      # Off-white / light slate
BG_CARD = colors.HexColor("#F1F5F9")       # Light card background
BORDER = colors.HexColor("#CBD5E1")        # Muted border
TEXT_MUTED = colors.HexColor("#64748B")    # Slate text
TEXT_MAIN = colors.HexColor("#0F172A")     # Dark text

# Global title/section mapping for running headers
PAGE_TOPIC_MAP = {
    1: "EXECUTIVE ANALYTICS SUMMARY",
    2: "DATA & METHODOLOGY",
    3: "MARKET & PRICE OVERVIEW",
    4: "TECHNICAL ANALYSIS & MOMENTUM",
    5: "RETURN & STATISTICAL ANALYSIS",
    6: "VOLATILITY DYNAMICS & GARCH",
    7: "QUANTITATIVE RISK ANALYTICS",
    8: "REGIME DETECTION & MARKOV STATES",
    9: "TIME SERIES DECOMPOSITION",
    10: "TIME SERIES MODEL BENCHMARK",
    11: "FORECAST HORIZON ANALYSIS",
    12: "MACHINE LEARNING FORECASTING",
    13: "DEEP LEARNING NEURAL MODELS",
    14: "STRATEGY BACKTESTING ENGINE",
    15: "MULTI-STRATEGY COMPARISON",
    16: "MONTE CARLO STOCHASTIC SIMULATION",
    17: "STOCHASTIC MODELS & TAIL RISK",
    18: "PORTFOLIO ALLOCATION & RISK",
    19: "FACTOR RESEARCH & STATISTICAL ARBITRAGE",
    20: "RESEARCH SUMMARY & METHODOLOGY LIMITATIONS",
}


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute total pages and draw consistent
    running headers and footers ("Page X of Y") with institutional styling.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []
        self.doc_ticker = "ASSET"
        self.doc_date = datetime.now().strftime("%Y-%m-%d")

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages: int):
        self.saveState()
        page_num = self._pageNumber
        topic = PAGE_TOPIC_MAP.get(page_num, "QUANTITATIVE RESEARCH")

        # Running Header (Pages 2-20)
        if page_num > 1:
            self.setFont(FONT_BOLD, 7)
            self.setFillColor(ACCENT_BLUE)
            self.drawString(MARGIN, PAGE_HEIGHT - 24, "QUANTTERMINAL INSTITUTIONAL RESEARCH")

            self.setFont(FONT_NORMAL, 7)
            self.setFillColor(TEXT_MUTED)
            self.drawRightString(PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 24, f"{self.doc_ticker} • {topic}")

            self.setStrokeColor(BORDER)
            self.setLineWidth(0.5)
            self.line(MARGIN, PAGE_HEIGHT - 28, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 28)

        # Running Footer (All Pages)
        self.setStrokeColor(BORDER)
        self.setLineWidth(0.5)
        self.line(MARGIN, 28, PAGE_WIDTH - MARGIN, 28)

        self.setFont(FONT_NORMAL, 6.5)
        self.setFillColor(TEXT_MUTED)
        self.drawString(
            MARGIN, 18,
            "QuantTerminal Research Engine • Algorithmic Quantitative Analytics • Not Personalized Financial Advice"
        )
        self.drawRightString(
            PAGE_WIDTH - MARGIN, 18,
            f"Page {page_num} of {total_pages}"
        )
        self.restoreState()


def get_report_styles():
    """Build and return standardized institutional ParagraphStyles."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        "CoverTag",
        parent=styles["Normal"],
        fontName=FONT_BOLD,
        fontSize=8,
        leading=10,
        textColor=ACCENT_BLUE,
        spaceAfter=4
    ))

    styles.add(ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName=FONT_BOLD,
        fontSize=18,
        leading=22,
        textColor=PRIMARY,
        spaceAfter=4
    ))

    styles.add(ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName=FONT_NORMAL,
        fontSize=8.5,
        leading=11,
        textColor=SECONDARY,
        spaceAfter=8
    ))

    styles.add(ParagraphStyle(
        "PageSectionLabel",
        parent=styles["Normal"],
        fontName=FONT_BOLD,
        fontSize=7,
        leading=9,
        textColor=ACCENT_BLUE,
        spaceAfter=2
    ))

    styles.add(ParagraphStyle(
        "PageHeading",
        parent=styles["Normal"],
        fontName=FONT_BOLD,
        fontSize=13,
        leading=16,
        textColor=PRIMARY,
        spaceAfter=3
    ))

    styles.add(ParagraphStyle(
        "PageSubheading",
        parent=styles["Normal"],
        fontName=FONT_NORMAL,
        fontSize=7.5,
        leading=10,
        textColor=TEXT_MUTED,
        spaceAfter=8
    ))

    styles.add(ParagraphStyle(
        "SectionSubheader",
        parent=styles["Normal"],
        fontName=FONT_BOLD,
        fontSize=8.5,
        leading=11,
        textColor=SECONDARY,
        spaceBefore=4,
        spaceAfter=4
    ))

    styles.add(ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName=FONT_NORMAL,
        fontSize=7.5,
        leading=10.5,
        textColor=TEXT_MAIN,
        spaceAfter=4
    ))

    styles.add(ParagraphStyle(
        "ReportBodyMuted",
        parent=styles["Normal"],
        fontName=FONT_NORMAL,
        fontSize=7,
        leading=9.5,
        textColor=TEXT_MUTED
    ))

    styles.add(ParagraphStyle(
        "CalloutBoxText",
        parent=styles["Normal"],
        fontName=FONT_NORMAL,
        fontSize=7,
        leading=9.5,
        textColor=SECONDARY
    ))

    styles.add(ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName=FONT_NORMAL,
        fontSize=7.5,
        leading=9.5,
        textColor=TEXT_MAIN
    ))

    styles.add(ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName=FONT_BOLD,
        fontSize=7.5,
        leading=9.5,
        textColor=PRIMARY
    ))

    styles.add(ParagraphStyle(
        "TableHead",
        parent=styles["Normal"],
        fontName=FONT_BOLD,
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white
    ))

    styles.add(ParagraphStyle(
        "KPICardLabel",
        parent=styles["Normal"],
        fontName=FONT_BOLD,
        fontSize=6.5,
        leading=8,
        textColor=TEXT_MUTED,
        alignment=1  # Centered
    ))

    styles.add(ParagraphStyle(
        "KPICardValue",
        parent=styles["Normal"],
        fontName=FONT_BOLD,
        fontSize=11,
        leading=13,
        textColor=PRIMARY,
        alignment=1  # Centered
    ))

    styles.add(ParagraphStyle(
        "KPICardSub",
        parent=styles["Normal"],
        fontName=FONT_NORMAL,
        fontSize=6,
        leading=7.5,
        textColor=TEXT_MUTED,
        alignment=1  # Centered
    ))

    styles.add(ParagraphStyle(
        "ReportDisclaimer",
        parent=styles["Normal"],
        fontName=FONT_NORMAL,
        fontSize=6.5,
        leading=9,
        textColor=colors.HexColor("#7F1D1D")
    ))

    return styles


def create_callout_box(text: str, style, title: str = "KEY FINDING") -> Table:
    """Helper to produce a clean bordered highlight/takeaway box."""
    content = [
        [Paragraph(f"<b>{title}:</b> {text}", style)]
    ]
    t = Table(content, colWidths=[PRINTABLE_WIDTH])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0F9FF")),
        ("BOX", (0, 0), (-1, -1), 0.75, ACCENT_BLUE),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t
