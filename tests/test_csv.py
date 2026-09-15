"""Tests for reports/csv.py (export_csv).

Independent checks against the docs/REPORTS_EXPORT.md contract: single
DataFrame -> one CSV, dict of DataFrames -> one CSV per key, timestamped
folder under exports/csv/, index preserved, NaN as empty field, deterministic
output, and validation of malformed inputs.
"""

import pandas as pd
import pytest

import reports.csv as rc


@pytest.fixture
def absent_ts(monkeypatch):
    monkeypatch.setattr(rc, "_timestamp", lambda: "20260101_120000")


def _frame(days: int = 4, name: str = "date", start: str = "2024-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=days, name=name)
    return pd.DataFrame(
        {
            "close": [100.0, 101.5, 99.75, 102.0],
            "return_pct": [0.01, 0.015, -0.0175, 0.0225],
            "note": ["a", "b", "c", "d"],
        },
        index=idx,
    ).iloc[:days]


# --------------------------------------------------------------- single frame


def test_single_dataframe_path_and_location(tmp_path, absent_ts):
    df = _frame()
    out = rc.export_csv(df, "quantterminal.csv", base_dir=tmp_path)
    expected = tmp_path / "csv" / "quantterminal_20260101_120000" / "quantterminal.csv"
    assert out == str(expected)
    assert expected.exists()


def test_extension_is_appended_when_missing(tmp_path, absent_ts):
    out = rc.export_csv(_frame(), "report", base_dir=tmp_path)
    assert out.endswith(".csv")


def test_single_csv_round_trips_exactly(tmp_path, absent_ts):
    df = _frame()
    out = rc.export_csv(df, "quantterminal.csv", base_dir=tmp_path)
    back = pd.read_csv(out, index_col=0, parse_dates=True)
    pd.testing.assert_frame_equal(back, df, check_dtype=False, check_freq=False)
    assert back.index.name == "date"


def test_index_preserved_in_file(tmp_path, absent_ts):
    out = rc.export_csv(_frame(name="date"), "x.csv", base_dir=tmp_path)
    text = pd.read_csv(out, index_col=0, parse_dates=True)
    assert list(text.index) == list(_frame(name="date").index)


def test_nan_serialised_as_empty_field(tmp_path, absent_ts, monkeypatch):
    df = _frame()
    df.loc[df.index[1], "note"] = None
    out = rc.export_csv(df, "x.csv", base_dir=tmp_path)
    lines = out and open(out).read().splitlines()
    fields = lines[2].split(",")
    assert fields[-1] == ""  # NaN last column -> empty field


def test_empty_dataframe_writes_header_only(tmp_path, absent_ts):
    df = pd.DataFrame({"close": pd.Series(dtype="float32"), "note": pd.Series(dtype="object")})
    out = rc.export_csv(df, "e.csv", base_dir=tmp_path)
    lines = open(out).read().splitlines()
    assert len(lines) == 1
    assert "close" in lines[0] and "note" in lines[0]


# ------------------------------------------------------------------ dict


def test_dict_writes_one_csv_per_key(tmp_path, absent_ts):
    df = _frame()
    out = rc.export_csv({"prices": df, "returns": df.iloc[1:]}, "quantterminal", base_dir=tmp_path)
    folder = tmp_path / "csv" / "quantterminal_20260101_120000"
    assert out == [str(folder / "prices.csv"), str(folder / "returns.csv")]
    assert (folder / "prices.csv").exists()
    assert (folder / "returns.csv").exists()


def test_dict_key_with_extension_not_doubled(tmp_path, absent_ts):
    df = _frame()
    out = rc.export_csv({"prices.csv": df}, "q", base_dir=tmp_path)
    assert out == [str(tmp_path / "csv" / "q_20260101_120000" / "prices.csv")]


def test_dict_preserves_duplicate_index_when_requested(tmp_path, absent_ts):
    df = _frame().iloc[[0, 0, 1, 2]]
    out = rc.export_csv({"dup": df}, "q", base_dir=tmp_path)
    back = pd.read_csv(out[0], index_col=0, parse_dates=True)
    assert len(back) == 4


def test_nested_filename_creates_subfolders(tmp_path, absent_ts):
    df = _frame()
    out = rc.export_csv(df, "nested/report.csv", base_dir=tmp_path)
    expected = tmp_path / "csv" / "report_20260101_120000" / "nested" / "report.csv"
    assert out == str(expected)
    assert expected.exists()


# --------------------------------------------------------------- determinism


def test_equal_inputs_produce_identical_bytes(tmp_path, absent_ts):
    df = _frame()
    out1 = rc.export_csv(df, "a.csv", base_dir=tmp_path)
    out2 = rc.export_csv(df, "a.csv", base_dir=tmp_path)
    assert open(out1, "rb").read() == open(out2, "rb").read()


def test_dict_order_is_preserved(tmp_path, absent_ts):
    df = _frame()
    out = rc.export_csv({"z": df, "a": df}, "q", base_dir=tmp_path)
    assert [str(p).split("/")[-1] for p in out] == ["z.csv", "a.csv"]


# --------------------------------------------------------------- validation


def test_invalid_data_type():
    with pytest.raises(TypeError, match="DataFrame or a dict"):
        rc.export_csv([1, 2, 3], "x.csv", base_dir=".")


def test_empty_dict_rejected(tmp_path):
    with pytest.raises(ValueError, match="at least one DataFrame"):
        rc.export_csv({}, "x.csv", base_dir=tmp_path)


def test_non_dataframe_dict_value_rejected(tmp_path):
    with pytest.raises(TypeError, match="DataFrames"):
        rc.export_csv({"x": [1, 2]}, "x.csv", base_dir=tmp_path)


def test_empty_or_blank_filename_rejected(tmp_path):
    with pytest.raises(ValueError, match="non-empty"):
        rc.export_csv(_frame(), "", base_dir=tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        rc.export_csv(_frame(), "   ", base_dir=tmp_path)


def test_blank_dict_key_rejected(tmp_path):
    with pytest.raises(ValueError, match="non-empty strings"):
        rc.export_csv({"": _frame()}, "x.csv", base_dir=tmp_path)