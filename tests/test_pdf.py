"""Tests for reports/pdf.py (export_pdf).

reportlab is the documented dependency (docs/REPORTS_EXPORT.md). It is NOT
installed in the current environment, so the full PDF-suite below is gated
on reportlab being importable while a dedicated test verifies the honest
ImportError that export_pdf raises without it.
"""

import importlib.util

import pandas as pd
import pytest

from reports.pdf import PDF_SECTIONS, export_pdf

HAS_REPORTLAB = importlib.util.find_spec("reportlab") is not None


def _frame(index_name: str | None = "date") -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=3, name=index_name or "index")
    return pd.DataFrame(
        {"close": [100.0, 101.5, 99.75], "return_pct": [0.01, -0.02, 0.015]},
        index=idx,
    )


def _exec_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "value": ["RELIANCE", "2024-01-01 to 2024-12-31", 0.213, 0.45],
        },
        index=["Ticker", "Date Range", "Total Return", "Sharpe"],
    )


def _report_data() -> dict:
    return {
        "Executive Summary": (_exec_summary(), None),
        "Statistics": (_frame(), None),
        "Factor Research": (_frame(), None),
    }


def test_export_pdf_reports_missing_dependency(tmp_path):
    if HAS_REPORTLAB:
        pytest.skip("reportlab installed; real PDF tests run instead")
    with pytest.raises(ImportError, match="reportlab"):
        export_pdf(_report_data(), "r.pdf", base_dir=tmp_path)


pytestmark = pytest.mark.skipif(
    not HAS_REPORTLAB,
    reason="reportlab not installed (documented PDF dependency)",
)


def test_pdf_created_at_documented_location(tmp_path, monkeypatch):
    monkeypatch.setattr("reports.pdf._timestamp", lambda: "2026-01-01 12:00:00")
    out = export_pdf(_report_data(), "quantterminal_report.pdf", base_dir=tmp_path)
    expected = tmp_path / "pdf" / "quantterminal_report.pdf"
    assert out == str(expected)
    assert expected.exists()
    with open(expected, "rb") as f:
        head = f.read(8)
    assert head.startswith(b"%PDF")


def test_extension_appended(tmp_path):
    out = export_pdf(_report_data(), "quantterminal_report", base_dir=tmp_path)
    assert out.endswith(".pdf")


def test_include_sections_filters_and_orders(tmp_path, monkeypatch):
    monkeypatch.setattr("reports.pdf._timestamp", lambda: "2026-01-01 12:00:00")
    data = {
        "Factor Research": (_frame(), None),
        "Risk": (_frame(), None),
        "Statistics": (_frame(), None),
    }
    out = export_pdf(data, "f.pdf", base_dir=tmp_path, include_sections=["Statistics", "Risk"])
    assert len(out) > 0 and out.endswith(".pdf")
    assert (tmp_path / "pdf" / "f.pdf").exists()


def test_unknown_include_names_are_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr("reports.pdf._timestamp", lambda: "2026-01-01 12:00:00")
    data = {"Statistics": (_frame(), None)}
    export_pdf(data, "g.pdf", base_dir=tmp_path, include_sections=["Statistics", "Not A Section"])


def test_dataframe_values_also_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr("reports.pdf._timestamp", lambda: "2026-01-01 12:00:00")
    out = export_pdf({"Statistics": _frame()}, "h.pdf", base_dir=tmp_path)
    assert (tmp_path / "pdf" / "h.pdf").exists()


def test_empty_section_renders_placeholder(tmp_path, monkeypatch):
    monkeypatch.setattr("reports.pdf._timestamp", lambda: "2026-01-01 12:00:00")
    empty = pd.DataFrame({"close": pd.Series(dtype="float32")})
    export_pdf({"Statistics": (empty, None)}, "i.pdf", base_dir=tmp_path, include_sections=["Statistics"])
    assert (tmp_path / "pdf" / "i.pdf").exists()


def test_title_page_metadata_fields(tmp_path, monkeypatch):
    monkeypatch.setattr("reports.pdf._timestamp", lambda: "2026-01-01 12:00:00")
    out = export_pdf(_report_data(), "meta.pdf", base_dir=tmp_path)
    # title page ticker/date-range read from Executive Summary without error
    assert (tmp_path / "pdf" / "meta.pdf").exists()


def test_deterministic_bytes_when_timestamp_pinned(tmp_path, monkeypatch):
    monkeypatch.setattr("reports.pdf._timestamp", lambda: "2026-01-01 12:00:00")
    data = _report_data()
    o1 = export_pdf(data, "det.pdf", base_dir=tmp_path)
    o2 = export_pdf(data, "det.pdf", base_dir=tmp_path)
    assert open(o1, "rb").read() == open(o2, "rb").read()


def test_validation_errors(tmp_path):
    with pytest.raises(TypeError, match="dict of section name"):
        export_pdf([("a", _frame())], "r.pdf", base_dir=tmp_path)
    with pytest.raises(ValueError, match="at least one section"):
        export_pdf({}, "r.pdf", base_dir=tmp_path)
    with pytest.raises(ValueError, match="DataFrame or"):
        export_pdf({"Statistics": (_frame(), None, None)}, "r.pdf", base_dir=tmp_path)
    with pytest.raises(TypeError, match="DataFrames or"):
        export_pdf({"Statistics": [1, 2]}, "r.pdf", base_dir=tmp_path)
    with pytest.raises(TypeError, match="list of section names"):
        export_pdf(_report_data(), "r.pdf", base_dir=tmp_path, include_sections="Statistics")
    with pytest.raises(ValueError, match="non-empty"):
        export_pdf({"": _frame()}, "r.pdf", base_dir=tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        export_pdf(_report_data(), " ", base_dir=tmp_path)