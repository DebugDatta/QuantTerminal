"""PDF report generation for quant-terminal reports.

Function:
    export_pdf - Render a section -> (DataFrame, Chart) mapping into a
                 multi-section PDF report with reportlab.

Contract sources:
    docs/REPORTS_EXPORT.md       - export_pdf(report_data, filename,
                                   include_sections=None); reportlab;
                                   documented section list and ordering;
                                   title page with project name / ticker /
                                   date range; each section begins a new
                                   page; tables with alternating row colors;
                                   charts embedded as PNG; page numbers,
                                   header with ticker name, footer with
                                   generation date and data source
    docs/ARCHITECTURE.md         - reports/pdf.py pinned as "# export_pdf"
    docs/TASK_DIVISION.md        - reports/pdf.py -> REPORTS_EXPORT
    docs/REPORTS_EXPORT.md       - kaleido/Pillow fallback: charts become
                                   tables-only with a warning when the deps
                                   are absent
    docs/MODEL_CONFIDENCE.md     - badge rendered "in Reports ... as a colored
                                   badge in Excel/PDF exports"

Notes
-----
- report_data maps a section name to either a DataFrame (tables-only) or a
  (DataFrame, chart) 2-tuple. Sections are emitted in the documented
  PDF_SECTIONS order; unknown keys follow in insertion order.
- include_sections filters sections (None keeps everything supplied). The
  user-facing section names on Page 17 differ from these report section
  names (e.g. 'Strategy' vs 'Strategy Performance'); the view layer maps
  them (see REVIEW-LATER).
- Chart flowables are produced through the chart object's write_image
  (Plotly + kaleido). When kaleido/Pillow are absent the documented
  tables-only fallback warning is emitted and the table is still rendered.
- Confidence badges are rendered by filling the cells of a column whose
  normalised name is confidence/badge/level, colouring them with the
  documented badge fills. Badge computation is not part of reports/.
- Title-page ticker / date range are best-effort fields read from the
  Executive Summary frame (rows labelled Ticker and Date Range/Period);
  if absent, a generic title page is produced.
"""

from __future__ import annotations

import re
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd

DEFAULT_BASE_DIR = "exports"

# Documented PDF section inventory and order (docs/REPORTS_EXPORT.md).
PDF_SECTIONS = (
    "Executive Summary",
    "Statistics",
    "Technical Analysis",
    "Risk",
    "Portfolio",
    "Strategy Performance",
    "Forecasting",
    "Machine Learning",
    "Regimes",
    "Monte Carlo",
    "Factor Research",
    "Statistical Arbitrage",
)

BADGE_FILLS = {
    "high": "#C6EFCE",
    "medium": "#FFEB9C",
    "low": "#FFC7CE",
}

RETURNS_NAME_RE = re.compile(r"return|pnl|cagr|yield|drawdown")
PRICE_NAME_RE = re.compile(r"(^|_)(close|open|high|low|price)(_|$)")
BADGE_NAME_RE = re.compile(r"^(confidence|badge|level)$")
HEADER_BG = "#0F172A"
HEADER_FG = "#F8FAFC"
ALTERNATE_BG = "#F1F5F9"

WARN_TABLES_ONLY = "kaleido/Pillow not installed; charts fall back to tables-only."


def _timestamp() -> str:
    return datetime.now().replace(microsecond=0).isoformat(sep=" ")


def _load_reportlab():
    try:
        import reportlab  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "export_pdf requires reportlab (documented dependency for PDF "
            "export in docs/REPORTS_EXPORT.md). Install reportlab into the "
            "project environment to enable PDF reports."
        ) from exc
    return reportlab


def _ensure_ext(name: str, ext: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("filename must be a non-empty string")
    stem = name.strip()
    return stem if stem.lower().endswith(ext.lower()) else f"{stem}{ext}"


def _normalise(col: str) -> str:
    return str(col).strip().lower().replace(" ", "_")


def _badge_level(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if s == "🟢":
        return "high"
    if s == "🟡":
        return "medium"
    if s == "🔴":
        return "low"
    low = s.lower()
    if low in {"high", "medium", "low"}:
        return low
    return None


def _executive_summary_meta(frame: pd.DataFrame):
    ticker, date_range = None, None
    for label in frame.index:
        if not isinstance(label, str):
            continue
        low = label.lower()
        if "ticker" in low and ticker is None:
            ticker = _first_value(frame.loc[label])
        elif ("date range" in low or "period" in low) and date_range is None:
            date_range = _first_value(frame.loc[label])
    return ticker, date_range


def _first_value(seq) -> str | None:
    for v in seq:
        if v is not None and not (isinstance(v, float) and pd.isna(v)):
            return str(v)
    return None


def _format_cell(col_norm: str, value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, (int, float, bool)):
        if RETURNS_NAME_RE.search(col_norm):
            return f"{float(value) * 100:.2f}%"
        if PRICE_NAME_RE.search(col_norm):
            return f"{float(value):.2f}"
        if isinstance(value, float):
            return f"{value:.4g}"
    text = str(value)
    if len(text) > 60:
        return text[:57] + "..."
    return text


def _build_table(flow, df: pd.DataFrame) -> None:
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    include_index = not isinstance(df.index, pd.RangeIndex)
    columns = list(df.columns)
    header = []
    if include_index:
        header.append(str(df.index.name or ""))
    header.extend([str(c) for c in columns])

    badge_cols = {}
    for j, col in enumerate(columns):
        if BADGE_NAME_RE.match(_normalise(col)):
            badge_cols[j] = _normalise(col)

    data = [header]
    for i in range(df.shape[0]):
        row = []
        if include_index:
            row.append(str(df.index[i]))
        for j, col in enumerate(columns):
            value = df.iloc[i, j]
            level = _badge_level(value) if j in badge_cols else None
            if level:
                text = _format_cell(_normalise(col), value)
                row.append(f"{level.capitalize()} ({BADGE_FILLS[level]})")
                continue
            row.append(_format_cell(_normalise(col), value))
        data.append(row)

    table = Table(data, repeatRows=1)
    col_count = len(header)
    bg_alternating = [
        (
            "BACKGROUND",
            (0, found),
            (col_count - 1, found),
            colors.HexColor(ALTERNATE_BG),
        )
        for found in range(2, len(data), 2)
    ]
    style = [
        ("BACKGROUND", (0, 0), (col_count - 1, 0), colors.HexColor(HEADER_BG)),
        ("TEXTCOLOR", (0, 0), (col_count - 1, 0), colors.HexColor(HEADER_FG)),
        ("FONTNAME", (0, 0), (col_count - 1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (col_count - 1, len(data) - 1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (col_count - 1, len(data) - 1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ] + bg_alternating
    # badge fills: colour the text cell background per documented level.
    for j, col_norm in badge_cols.items():
        start_col = j + (1 if include_index else 0)
        for i in range(1, len(data)):
            cell_text = data[i][start_col]
            level = cell_text.split(" (")[0].lower() if " (" in cell_text else None
            if level in BADGE_FILLS:
                style.append(
                    ("BACKGROUND", (start_col, i), (start_col, i), colors.HexColor(BADGE_FILLS[level]))
                )
    table.setStyle(TableStyle(style))
    flow.append(table)


def _chart_image(chart) -> object | None:
    try:
        import kaleido  # noqa: F401
        import PIL  # noqa: F401
    except ImportError:
        return None
    writer = getattr(chart, "write_image", None)
    if writer is None:
        return None
    try:
        from reportlab.platypus import Image
        import tempfile
        import os

        tmp = tempfile.mktemp(suffix=".png")
        writer(tmp)
        img = Image(tmp)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return img
    except Exception:
        return None


def export_pdf(
    report_data: dict,
    filename: str,
    include_sections: list[str] | None = None,
    base_dir: str | Path = DEFAULT_BASE_DIR,
) -> str:
    """Render and save a multi-section PDF report.

    Parameters
    ----------
    report_data : dict[str, pd.DataFrame | (pd.DataFrame, chart)]
        Section name -> DataFrame or (DataFrame, chart-flowable source). A
        chart is embedded only when it exposes write_image and kaleido/Pillow
        are installed; otherwise the documented tables-only fallback applies.
    filename : str
        Output filename (.pdf appended if missing), written under
        base_dir/pdf/.
    include_sections : list[str] or None
        Subset of section names to include. None includes every supplied
        section. Names absent from report_data are ignored.
    base_dir : str or Path, default "exports"
        Root that receives the pdf/ subfolder.

    Returns
    -------
    str
        Path of the written PDF.
    """
    _load_reportlab()
    if not isinstance(report_data, dict):
        raise TypeError("report_data must be a dict of section name -> frame")
    if not report_data:
        raise ValueError("report_data must contain at least one section")
    cleaned = {}
    for name, value in report_data.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("report_data keys must be non-empty section names")
        if isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError("section values must be DataFrame or (DataFrame, chart)")
            frame, chart = value
        else:
            frame, chart = value, None
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                f"report_data values must be pandas DataFrames or "
                f"(DataFrame, chart); got {type(frame).__name__} for {name!r}"
            )
        cleaned[name] = (frame, chart)

    if include_sections is not None:
        if not isinstance(include_sections, (list, tuple)) or not all(
            isinstance(s, str) for s in include_sections
        ):
            raise TypeError("include_sections must be a list of section names")
        wanted = set(include_sections)
        cleaned = {k: v for k, v in cleaned.items() if k in wanted}

    known = set(PDF_SECTIONS)
    ordered = [s for s in PDF_SECTIONS if s in cleaned]
    ordered += [k for k in cleaned if k not in known]

    out_name = _ensure_ext(filename, ".pdf")
    out_path = Path(base_dir) / "pdf" / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.platypus import (
        HRFlowable,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
    )

    ticker, date_range = None, None
    if "Executive Summary" in cleaned:
        ticker, date_range = _executive_summary_meta(cleaned["Executive Summary"][0])
    header_text = ticker or "QuantTerminal"
    generated_ts = _timestamp()
    n_sections = len(ordered)

    def _on_page(canvas: Canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColorRGB(0x0F / 255, 0x17 / 255, 0x2A / 255)
        canvas.drawString(15 * mm, A4[1] - 12 * mm, header_text)
        canvas.setFillColorRGB(0.45, 0.45, 0.45)
        canvas.drawRightString(
            A4[0] - 15 * mm, A4[1] - 12 * mm,
            f"Generated {generated_ts}  |  Data source: computed analytics",
        )
        canvas.setStrokeColorRGB(0.85, 0.85, 0.85)
        canvas.line(15 * mm, A4[1] - 15 * mm, A4[0] - 15 * mm, A4[1] - 15 * mm)
        canvas.drawString(15 * mm, 12 * mm, f"{header_text}  |  Page {doc.page}")
        canvas.restoreState()

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading_style = styles["Heading1"]
    normal = styles["Normal"]

    doc = SimpleDocTemplate(str(out_path), pagesize=A4)

    story = []
    story.append(Paragraph("QuantTerminal Report", title_style))
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(f"Report: {Path(out_name).stem}", normal))
    if ticker:
        story.append(Paragraph(f"Ticker: {ticker}", normal))
    if date_range:
        story.append(Paragraph(f"Date range: {date_range}", normal))
    story.append(Paragraph(f"Generated: {generated_ts}", normal))
    story.append(Spacer(1, 4 * mm))
    story.append(
        HRFlowable(width="100%", thickness=1, color="#0F172A", spaceAfter=6)
    )
    story.append(Paragraph("All data derived from computed analytics.", normal))

    chart_fallback_warned = False
    for idx, section in enumerate(ordered):
        if idx or story:
            story.append(PageBreak())
        story.append(Paragraph(section, heading_style))
        story.append(Spacer(1, 3 * mm))
        frame, chart = cleaned[section]
        if frame.empty:
            story.append(Paragraph("No data for this section.", normal))
        else:
            _build_table(story, frame)
        if chart is not None:
            image = _chart_image(chart)
            if image is not None:
                story.append(Spacer(1, 4 * mm))
                story.append(image)
            elif not chart_fallback_warned:
                warnings.warn(WARN_TABLES_ONLY, stacklevel=2)
                chart_fallback_warned = True

    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    return str(out_path)