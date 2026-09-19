"""Tests for reports/excel.py (export_excel).

openpyxl is the documented dependency (docs/REPORTS_EXPORT.md). It is NOT
installed in the current environment, so the full workbook-suite below is
gated on openpyxl being importable while a dedicated test verifies the
honest ImportError that export_excel raises without it (no silent
substitution of another package or format).
"""

import importlib.util
import zipfile

import pandas as pd
import pytest

from reports.excel import (
    BADGE_FILLS,
    EXCEL_SHEETS,
    export_excel,
)

HAS_OPENPYXL = importlib.util.find_spec("openpyxl") is not None


def _frame(index_name: str | None = "date") -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=3, name=index_name or "index")
    return pd.DataFrame(
        {
            "close": [100.0, 101.5, 99.75],
            "return_pct": [0.01, -0.02, 0.015],
            "volume": [1000, 1200, 900],
        },
        index=idx,
    )


def _confidence_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "garch_coef": [0.1, 0.3, 0.2],
            "confidence": ["high", "medium", "low"],
        }
    )


def test_export_excel_reports_missing_dependency(tmp_path):
    if HAS_OPENPYXL:
        pytest.skip("openpyxl installed; real workbook tests run instead")
    with pytest.raises(ImportError, match="openpyxl"):
        export_excel({"Executive Summary": _frame()}, "r.xlsx", base_dir=tmp_path)


pytestmark = pytest.mark.skipif(
    not HAS_OPENPYXL,
    reason="openpyxl not installed (documented Excel dependency)",
)


@pytest.fixture
def openpyxl():
    import openpyxl

    return openpyxl


def _reload(path, openpyxl):
    return openpyxl.load_workbook(path)


def test_workbook_created_at_documented_location(tmp_path, openpyxl):
    out = export_excel({"Returns": _frame()}, "quantterminal_report.xlsx", base_dir=tmp_path)
    expected = tmp_path / "excel" / "quantterminal_report.xlsx"
    assert out == str(expected)
    assert expected.exists()
    assert zipfile.is_zipfile(expected)


def test_extension_is_appended(tmp_path, openpyxl):
    out = export_excel({"Returns": _frame()}, "quantterminal_report", base_dir=tmp_path)
    assert out.endswith(".xlsx")


def test_sheet_names_subset_in_documented_order(tmp_path, openpyxl):
    data = {"Risk": _frame(), "Factors": _frame(), "Returns": _frame()}
    out = export_excel(data, "r.xlsx", base_dir=tmp_path)
    wb = _reload(out, openpyxl)
    assert wb.sheetnames == ["Returns", "Risk", "Factors"]


def test_unknown_keys_appended_after_known_sheets(tmp_path, openpyxl):
    data = {"Executive Summary": _frame(), "Custom Section": _frame()}
    out = export_excel(data, "r.xlsx", base_dir=tmp_path)
    wb = _reload(out, openpyxl)
    assert wb.sheetnames == ["Executive Summary", "Custom Section"]


def test_headers_bold_and_frozen(tmp_path, openpyxl):
    out = export_excel({"Returns": _frame()}, "r.xlsx", base_dir=tmp_path)
    wb = _reload(out, openpyxl)
    ws = wb["Returns"]
    assert ws.freeze_panes == "A2"
    assert ws["A1"].font.bold
    assert ws["B1"].font.bold


def test_representative_values_round_trip(tmp_path, openpyxl):
    df = _frame()
    out = export_excel({"Returns": df}, "r.xlsx", base_dir=tmp_path)
    wb = _reload(out, openpyxl)
    ws = wb["Returns"]
    assert abs(ws["B2"].value - 100.0) < 1e-9
    assert abs(ws["B4"].value - 99.75) < 1e-9
    assert ws["C2"].value == pytest.approx(0.01)


def test_named_index_written_with_label(tmp_path, openpyxl):
    df = _frame(index_name="date")
    out = export_excel({"Returns": df}, "r.xlsx", base_dir=tmp_path)
    wb = _reload(out, openpyxl)
    ws = wb["Returns"]
    assert ws["A1"].value == "date"
    assert ws["A2"].value == pd.Timestamp("2024-01-01")
    assert ws["B2"].value == pytest.approx(100.0)


def test_range_index_index_column_omitted(tmp_path, openpyxl):
    df = pd.DataFrame({"close": [1.0, 2.0]})
    out = export_excel({"Returns": df}, "r.xlsx", base_dir=tmp_path)
    wb = _reload(out, openpyxl)
    ws = wb["Returns"]
    assert ws["A1"].value == "close"
    assert ws["A2"].value == pytest.approx(1.0)


def test_returns_percentage_and_price_formatting(tmp_path, openpyxl):
    out = export_excel({"Returns": _frame()}, "r.xlsx", base_dir=tmp_path)
    wb = _reload(out, openpyxl)
    ws = wb["Returns"]
    assert ws["B2"].number_format == "0.00"
    assert ws["C2"].number_format == "0.00%"


def test_conditional_formatting_green_red(tmp_path, openpyxl):
    out = export_excel({"Returns": _frame()}, "r.xlsx", base_dir=tmp_path)
    wb = _reload(out, openpyxl)
    ws = wb["Returns"]
    rules = list(ws.conditional_formatting)
    assert len(rules) == 1
    ops = [r.operator for r in rules[0].rules]
    assert "greaterThan" in ops and "lessThan" in ops


def test_confidence_badge_fills(tmp_path, openpyxl):
    out = export_excel({"Statistics": _confidence_frame()}, "r.xlsx", base_dir=tmp_path)
    wb = _reload(out, openpyxl)
    ws = wb["Statistics"]
    high_cell = ws["B2"]
    assert high_cell.fill.start_color.rgb in (f"00{BADGE_FILLS['high']}", BADGE_FILLS["high"].upper())
    assert ws["B3"].fill.start_color.rgb.endswith(BADGE_FILLS["medium"])


def test_charts_sheet_warns_tables_only(tmp_path, openpyxl):
    charts = pd.DataFrame({"chart": ["path/to/equity.png"]})
    with pytest.warns(Warning, match="tables-only"):
        out = export_excel({"Charts": charts}, "r.xlsx", base_dir=tmp_path)
    wb = _reload(out, openpyxl)
    assert "Charts" in wb.sheetnames


def test_overwrite_same_path_succeeds(tmp_path, openpyxl):
    out1 = export_excel({"Returns": _frame()}, "r.xlsx", base_dir=tmp_path)
    out2 = export_excel({"Returns": _frame()}, "r.xlsx", base_dir=tmp_path)
    assert out1 == out2
    wb = _reload(out2, openpyxl)
    assert wb.sheetnames == ["Returns"]


def test_empty_dataframe_writes_header_only(tmp_path, openpyxl):
    df = pd.DataFrame({"close": pd.Series(dtype="float32"), "note": pd.Series(dtype="object")})
    out = export_excel({"Returns": df}, "r.xlsx", base_dir=tmp_path)
    wb = _reload(out, openpyxl)
    ws = wb["Returns"]
    assert ws["A1"].value == "close"
    assert ws["A2"].value is None


def test_validation_errors(tmp_path, openpyxl):
    with pytest.raises(TypeError, match="dict of section name"):
        export_excel([("a", _frame())], "r.xlsx", base_dir=tmp_path)
    with pytest.raises(ValueError, match="at least one section"):
        export_excel({}, "r.xlsx", base_dir=tmp_path)
    with pytest.raises(TypeError, match="DataFrames"):
        export_excel({"Returns": [1, 2]}, "r.xlsx", base_dir=tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        export_excel({"": _frame()}, "r.xlsx", base_dir=tmp_path)


def test_deterministic_structure(tmp_path, openpyxl):
    data = {"Returns": _frame(), "Risk": _frame()}
    out1 = export_excel(data, "d.xlsx", base_dir=tmp_path)
    out2 = export_excel(data, "d.xlsx", base_dir=tmp_path)
    wb1 = _reload(out1, openpyxl)
    wb2 = _reload(out2, openpyxl)
    assert wb1.sheetnames == wb2.sheetnames
    for sheet in wb1.sheetnames:
        for row1, row2 in zip(wb1[sheet].iter_rows(values_only=True), wb2[sheet].iter_rows(values_only=True)):
            assert row1 == row2