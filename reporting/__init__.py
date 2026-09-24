"""
QuantTerminal Institutional Reporting Engine.
Provides complete 20-page quantitative analytics PDF generation.
"""

from reporting.report_data import build_report_data
from reporting.pdf_generator import generate_pdf_report
from reporting.report_styles import NumberedCanvas, get_report_styles

__all__ = [
    "build_report_data",
    "generate_pdf_report",
    "NumberedCanvas",
    "get_report_styles",
]
