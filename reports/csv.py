"""CSV export for quant-terminal reports.

Function:
    export_csv - Export a DataFrame or a dict of DataFrames to CSV file(s).

Contract sources:
    docs/REPORTS_EXPORT.md       - export_csv(data, filename); single
                                   DataFrame -> single CSV, dict of
                                   DataFrames -> multiple CSVs with
                                   filenames derived from the keys,
                                   all written to a timestamped folder
                                   under exports/csv/
    docs/ARCHITECTURE.md         - reports/csv.py pinned as "# export_csv"
    docs/TASK_DIVISION.md        - reports/csv.py -> REPORTS_EXPORT
    docs/STREAMLIT_PAGES.md      - Page 17: report builds write exportable
                                   files; a Download button appears with
                                   the produced filename
    docs/REPORTS_EXPORT.md       - Export Location: exports/csv/<stamp>/
                                   priced, returns, ...  .csv files

Notes
-----
- Export is pure serialization: values are written exactly as computed by
  the analytics modules. No analysis is recalculated here.
- Index is always written (to_csv index=True) so time-series export keeps
  its date/asset labels, matching PANEL: multi-asset data indexed by date
  (docs/DATA_LAYER.md).
- NaN is serialised as an empty field (pandas default).
- The timestamped folder uses the filename stem plus a %Y%m%d_%H%M%S stamp,
  matching the documented examples (quantterminal_20260101_120000).
  `_timestamp()` is a module function exactly so tests can pin the stamp.

REVIEW-LATER (REPORTS_EXPORT gaps):
- The documented workflow shows a per-section folder under exports/csv/;
  we write a single timestamped folder per export containing the files.
- Whether the timestamp should be generated once per export (our choice)
  or per file is not pinned by the docs.
- Return value (path(s)) is not documented; we return the written path for
  a single frame and the ordered list of written paths for a dict.
- A custom export root (Page 18 "Export Path" setting) is not wired into
  this library: base_dir defaults to the documented exports/ root.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

DEFAULT_BASE_DIR = "exports"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


def _timestamp() -> str:
    """Local timestamp used for the export folder name (tests pin this)."""
    return datetime.now().strftime(TIMESTAMP_FORMAT)


def _ensure_py_ext(name: str, ext: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("filename must be a non-empty string")
    stem = name.strip()
    return stem if stem.lower().endswith(ext.lower()) else f"{stem}{ext}"


def export_csv(
    data: pd.DataFrame | dict,
    filename: str,
    base_dir: str | Path = DEFAULT_BASE_DIR,
) -> str | list[str]:
    """Export CSV file(s) under a timestamped folder in exports/csv/.

    Parameters
    ----------
    data : pd.DataFrame or dict of pd.DataFrame
        A single DataFrame exports one CSV named `filename`. A dict exports
        one CSV per key, each file named after its key (".csv" appended if
        missing). dict keys must map to pandas DataFrame values.
    filename : str
        Output filename. For a single DataFrame this is the file name; for a
        dict it names the timestamped folder (stem). ".csv" is appended when
        missing.
    base_dir : str or Path, default "exports"
        Root that receives the csv/ subfolder. Kept keyword-only-usable so a
        future custom export root (Page 18) can plug in here.

    Returns
    -------
    str or list[str]
        Path of the single written file, or the ordered list of written
        file paths for a dict input.
    """
    if isinstance(data, pd.DataFrame):
        file = _export_single(data, filename, Path(base_dir))
        return str(file)
    if isinstance(data, dict):
        return [str(p) for p in _export_dict(data, filename, Path(base_dir))]
    raise TypeError("data must be a pandas DataFrame or a dict of DataFrames")


def _export_single(df: pd.DataFrame, filename: str, base_dir: Path) -> Path:
    out_name = _ensure_py_ext(filename, ".csv")
    out_dir = base_dir / "csv" / f"{Path(out_name).stem}_{_timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / out_name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)
    return path


def _export_dict(data: dict, filename: str, base_dir: Path) -> list[Path]:
    if not data:
        raise ValueError("data dict must contain at least one DataFrame")
    out_name = _ensure_py_ext(filename, ".csv")
    out_dir = base_dir / "csv" / f"{Path(out_name).stem}_{_timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for key, df in data.items():
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"dict values must be pandas DataFrames; got {type(df).__name__} "
                f"for key {key!r}"
            )
        if not isinstance(key, str) or not key.strip():
            raise ValueError("data dict keys must be non-empty strings")
        file_name = _ensure_py_ext(key.strip(), ".csv")
        path = out_dir / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=True)
        paths.append(path)
    return paths