"""Page 10 — Statistical Arbitrage.

Contract sources:
    docs/STREAMLIT_PAGES.md §10  - sidebar controls (Universe text_area,
                                    Exchange, Pair Search Method, Top Pairs,
                                    Entry/Exit Z-Score), tables (Pair Rankings,
                                    Cointegration Results with half-life badge,
                                    Current Spread) and charts (Spread with
                                    z-score bands, Price ratio, Trading signals
                                    on spread, Correlation scatter).
    docs/STATISTICAL_MODELS.md   - §7 cointegration output fields (test
                                    statistic, p-value, hedge ratio, spread
                                    residuals), §9 distance metrics, §10
                                    half-life definition and speed table.
    docs/MODEL_CONFIDENCE.md     - Cointegration confidence badge table
                                    (observations / EG p-value / half-life).
    docs/BIAS_MITIGATION.md      - B4 spurious-regression gate (Part C maps
                                    B4 to Page 10); B1 current-constituents
                                    label; B6 short-sale caveat.
    docs/ARCHITECTURE.md         - statarb module ownership: pairs.py
                                    (find_pairs, pair_distance),
                                    cointegration.py (engle_granger, johansen),
                                    spread.py (calc_spread, calc_zscore,
                                    mean_reversion_signals).
    docs/TASK_DIVISION.md        - StatArb module dir assigned to Pranav.

All analytics are consumed from the shared statarb APIs only — no formula is
duplicated on this page:
    statarb/pairs.py         - find_pairs (pair ranking)
    statarb/cointegration.py - engle_granger (B4 gate + hedge ratio)
    statarb/spread.py        - calc_spread, calc_zscore, mean_reversion_signals,
                               half_life

REVIEW-LATER (documented conflicts/dependencies; also surfaced in-page):
    P10-1  Page filename inferred from the repository naming convention
           ("NN_Name.py"); no doc explicitly pins pages/10_Statistical_Arbitrage.py.
    P10-2  ARCHITECTURE.md lists statarb/__init__.py; the repo has none
           (namespace package). The shared statarb modules are imported
           directly and are untouched.
    P10-3  §10 selectbox "Correlation"/"Distance" maps to find_pairs aliases
           pearson/euclidean (statarb/pairs.py docstring); the two
           nomenclatures do not match §9's "Pearson Distance / Euclidean
           Distance" exactly.
    P10-4  "Cointegration" pair search has no shared ranked-by-p-value API;
           this page loops engle_granger per pair itself. Johansen is not
           surfaced: §7's p-value / hedge-ratio / residuals fields are
           authored for Engle-Granger (see statarb/cointegration.py docstring),
           and §10's Cointegration Results row is Engle-Granger-shaped.
    P10-5  Confidence badge: MODEL_CONFIDENCE.md specifies a three-check table
           (obs / EG p / half-life) but no combination rule and no floor. This
           page combines as the *worst* (least-confidence) check and treats a
           non-mean-reverting (NaN) half-life as Low — an inference.
    P10-6  Half-life NaN (theta outside (-1, 0)) -> "not mean-reverting": the
           speed table (STATISTICAL_MODELS §10) yields no label and the badge
           yields Low; both are reported, per the documented distinction in
           STATISTICAL_MODELS.md §10.
    P10-7  z-score window is fixed at 20 trading days (= documented Mean
           Reversion lookback default, STRATEGIES_BACKTESTING.md §9, and
           statarb/spread.py DEFAULT_WINDOW). §10 lists no z-window control,
           so none was added.
    P10-8  The §10 universe is a free-form text_area. Ticker resolution for
           bare symbols is a local helper (data/fundamentals.py resolve_ticker
           is absent). A cap of 25 tickers is an implementation guard for
           response time on the per-pair Engle-Granger search.
    P10-9  Engle-Granger enforces a 100-observation floor
           (statarb/cointegration.py); pairs below it are skipped and flagged.
           Under "Correlation"/"Distance", pairs are ranked by distance and
           the Cointegration Results table then reports Engle-Granger ONLY for
           the displayed top pairs.
    P10-10 Price ratio uses Close levels (A / B, inner-joined); the
           correlation scatter uses simple returns, not levels (B4), and no
           bare R^2 on trending levels is reported.
"""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "utils")):
    if _p not in sys.path:
        sys.path.append(_p)

try:
    from utils.helper import inject_custom_theme, load_data, fetch_stocks
except ImportError:
    from helper import inject_custom_theme, load_data, fetch_stocks

try:
    from statarb.pairs import find_pairs
    from statarb.cointegration import engle_granger
    from statarb.spread import (
        calc_spread,
        calc_zscore,
        mean_reversion_signals,
        half_life,
        DEFAULT_WINDOW,
    )
except ImportError:
    from pairs import find_pairs
    from cointegration import engle_granger
    from spread import (
        calc_spread,
        calc_zscore,
        mean_reversion_signals,
        half_life,
        DEFAULT_WINDOW,
    )

EXCHANGE_OPTIONS = ("Auto", "NSE", "BSE", "Global")
SEARCH_METHODS = ("Correlation", "Distance", "Cointegration")
_SEARCH_METHOD = {"Correlation": "pearson", "Distance": "euclidean"}

# LOOKBACK is not specified by the §10 docs; 5y comfortably exceeds the
# Engle-Granger 100-observation floor and the z-score warm-up (P10-X, see file
# docstring note): period chosen to standardize across the stat-arb modules.
LOOKBACK_PERIOD = "5y"
LOOKBACK_INTERVAL = "1d"

# §10 Entry/Exit z-score ranges.
ENTRY_Z_MIN, ENTRY_Z_MAX, ENTRY_Z_DEFAULT = 1.0, 3.0, 2.0
EXIT_Z_MIN, EXIT_Z_MAX, EXIT_Z_DEFAULT = 0.1, 2.0, 0.5

# Top pairs slider.
TOP_PAIRS_MIN, TOP_PAIRS_MAX, TOP_PAIRS_DEFAULT = 1, 15, 5

# Guards.
MAX_TICKERS = 25
MIN_UNIVERSE = 2

BADGE_HIGH, BADGE_MED, BADGE_LOW = "🟢", "🟡", "🔴"
_BADGE_RANK = {BADGE_LOW: 0, BADGE_MED: 1, BADGE_HIGH: 2}
_BADGE_LABEL = {BADGE_HIGH: "High", BADGE_MED: "Medium", BADGE_LOW: "Low"}


# ── Formatting ───────────────────────────────────────────────────────────────
def _to_float(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _fmt(value, digits: int = 4) -> str:
    v = _to_float(value)
    return "N/A" if v is None else f"{v:.{digits}f}"


def _fmt_pct(value, digits: int = 2) -> str:
    v = _to_float(value)
    return "N/A" if v is None else f"{v * 100:.{digits}f}%"


def _fmt_date(value) -> str:
    return value.date().isoformat() if hasattr(value, "date") else str(value)


def _fmt_pair(a, b) -> str:
    return f"{a} ↔ {b}"


# ── Universe & data ──────────────────────────────────────────────────────────
def _snapshot_symbols(exchange: str):
    """Per-exchange symbol lookup for bare-ticker resolution (local helper)."""
    us = fetch_stocks("US")
    ind = fetch_stocks("India")
    nse = set(ind.loc[ind["Exchange"] == "NSE", "Symbol"].astype(str))
    bse = set(ind.loc[ind["Exchange"] == "BSE", "Symbol"].astype(str))
    glob = set(us["Symbol"].astype(str))
    return nse, bse, glob


def _resolve_ticker(token: str, exchange: str, nse, bse, glob) -> str:
    """Attach .NS/.BO suffix to bare symbols using the snapshot (best effort)."""
    tok = token.strip()
    if not tok:
        return ""
    if "." in tok:
        return tok
    up = tok.upper()
    if exchange in ("Auto", "NSE") and up in nse:
        return f"{up}.NS"
    if exchange == "BSE" and up in bse:
        return f"{up}.BO"
    if exchange in ("Auto", "Global") and up in glob:
        return up
    return tok


def _parse_tickers(text: str) -> list:
    parts = [p.strip() for p in text.replace("\n", ",").split(",")]
    return [p for p in parts if p]


@st.cache_data(show_spinner=False)
def _load_ohlcv(ticker: str) -> pd.DataFrame:
    df = load_data(ticker, period=LOOKBACK_PERIOD, interval=LOOKBACK_INTERVAL)
    if df is None or df.empty or "Close" not in df.columns:
        return pd.DataFrame()
    out = df[["Close"]].copy()
    out.index = pd.to_datetime(out.index)
    return out


def _build_close_panel(frames: dict):
    """Inner-join Close panels on shared trading days (DATA_LAYER rule)."""
    valid = {
        t: df
        for t, df in frames.items()
        if not df.empty and "Close" in df.columns and len(df) >= 3
    }
    if len(valid) < MIN_UNIVERSE:
        return None, []
    idx = None
    for df in valid.values():
        d = df["Close"].dropna()
        idx = d.index if idx is None else idx.intersection(d.index)
    if idx is None or len(idx) < 3:
        return None, []
    panel = pd.DataFrame(
        {t: df["Close"].reindex(idx) for t, df in valid.items()}
    ).sort_index()
    panel = panel.dropna(how="any")
    if panel.shape[1] < MIN_UNIVERSE or len(panel) < 3:
        return None, []
    return panel, sorted(valid.keys())


# ── Pair ranking ─────────────────────────────────────────────────────────────
def _pair_label_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Pair"] = [_fmt_pair(a, b) for a, b in zip(out["asset_1"], out["asset_2"])]
    return out.drop(columns=["asset_1", "asset_2"])


def _rank_by_correlation(panel: pd.DataFrame) -> pd.DataFrame:
    """find_pairs pearson distance (returns-based, B4-safe)."""
    if panel.shape[1] < MIN_UNIVERSE:
        return pd.DataFrame()
    result = find_pairs(panel, method="pearson")
    out = _pair_label_frame(result)
    out = out.rename(columns={"correlation": "Correlation", "distance": "Distance"})
    return out[["Pair", "Correlation", "Distance", "n", "rank"]].rename(
        columns={"rank": "Rank"}
    )


def _rank_by_distance(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.shape[1] < MIN_UNIVERSE:
        return pd.DataFrame()
    result = find_pairs(panel, method="euclidean")
    out = _pair_label_frame(result)
    out = out.rename(columns={"distance": "Distance"})
    return out[["Pair", "Distance", "n", "rank"]].rename(columns={"rank": "Rank"})


def _eg_for_pair(a: str, b: str, panel: pd.DataFrame):
    return engle_granger(panel[a], panel[b])


def _rank_by_cointegration(panel: pd.DataFrame, eg_cache: dict = None):
    """Engle-Granger pair search: rank pairs by coint p (asc)."""
    cache = eg_cache if eg_cache is not None else {}
    cols = list(panel.columns)
    rows, skipped = [], []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            key = tuple(sorted((str(a), str(b))))
            try:
                eg = cache.get(key)
                if not isinstance(eg, dict):
                    eg = _eg_for_pair(a, b, panel)
                    cache[key] = eg
            except (ValueError, TypeError) as exc:
                cache[key] = ("error", str(exc))
                skipped.append((_fmt_pair(a, b), str(exc)))
                continue
            rows.append({
                "Pair": _fmt_pair(a, b),
                "Coint p": eg["p_value"],
                "Is Cointegrated": eg["is_cointegrated"],
                "Hedge Ratio": eg["hedge_ratio"],
                "n": eg["n"],
            })
    if not rows:
        return pd.DataFrame(), skipped
    df = pd.DataFrame(rows).sort_values(
        "Coint p", ascending=True, na_position="last", kind="mergesort"
    )
    df = df.reset_index(drop=True)
    df["Rank"] = np.arange(1, len(df) + 1, dtype=int)
    return df, skipped


# ── Cointegration results + badge ────────────────────────────────────────────
def _coint_badge(eg: dict, hl: float):
    """MODEL_CONFIDENCE.md cointegration badge; worst check wins (P10-5)."""
    n = int(eg["n"])
    p = eg["p_value"]
    checks = {}

    if n >= 250:
        checks["Observations"] = BADGE_HIGH
    elif n >= 100:
        checks["Observations"] = BADGE_MED
    else:
        checks["Observations"] = BADGE_LOW

    if p < 0.01:
        checks["EG p-value"] = BADGE_HIGH
    elif p < 0.05:
        checks["EG p-value"] = BADGE_MED
    else:
        checks["EG p-value"] = BADGE_LOW

    if hl is None or not np.isfinite(hl) or pd.isna(hl):
        checks["Half-life"] = BADGE_LOW
    elif 5.0 <= hl <= 60.0:
        checks["Half-life"] = BADGE_HIGH
    elif (1.0 <= hl < 5.0) or (60.0 < hl <= 120.0):
        checks["Half-life"] = BADGE_MED
    else:
        checks["Half-life"] = BADGE_LOW

    overall = min(check for _, check in checks.items())
    return overall, _BADGE_LABEL[overall], checks


def _build_coint_table(
    ranked_pairs: list, panel: pd.DataFrame, eg_cache: dict
) -> pd.DataFrame:
    """Engle-Granger detail rows for the displayed pairs (statistic/p/hl/badge)."""
    rows = []
    for a, b in ranked_pairs:
        key = tuple(sorted((str(a), str(b))))
        if key not in eg_cache:
            try:
                eg_cache[key] = _eg_for_pair(a, b, panel)
            except (ValueError, TypeError) as exc:
                eg_cache[key] = ("error", str(exc))
        eg = eg_cache[key]
        if not isinstance(eg, dict):
            rows.append({
                "Pair": _fmt_pair(a, b),
                "Test Statistic": "N/A",
                "p-value": "N/A",
                "Is Cointegrated": "N/A",
                "Hedge Ratio": "N/A",
                "Half-Life": "N/A",
                "Speed (STATISTICAL_MODELS §10)": "Skipped",
                "Confidence": "N/A",
                "n": "N/A",
            })
            continue
        hl_result = half_life(eg["residuals"])
        hl_days = hl_result["half_life"]
        speed = hl_result["interpretation"]
        if speed is None:
            speed = "Not mean-reverting"
        badge, label, _checks = _coint_badge(eg, _to_float(hl_days))
        rows.append({
            "Pair": _fmt_pair(a, b),
            "Test Statistic": _fmt(eg["test_statistic"], 3),
            "p-value": _fmt(eg["p_value"], 4),
            "Is Cointegrated": (
                "Yes" if eg["is_cointegrated"] else "No"
            ),
            "Hedge Ratio": _fmt(eg["hedge_ratio"], 3),
            "Half-Life": (
                f"{_fmt(hl_days, 1)} d" if _to_float(hl_days) is not None else "N/A"
            ),
            "Speed (STATISTICAL_MODELS §10)": speed,
            "Confidence": f"{badge} {label}",
            "n": int(eg["n"]),
        })
    return pd.DataFrame(rows)


def _selected_pairs(ranked: pd.DataFrame, method: str, top_n: int):
    """(a, b) tuples for the top-ranked displayed pairs."""
    pairs = []
    if method == "Cointegration":
        for label in ranked["Pair"].head(top_n):
            a, b = label.split(" ↔ ")
            pairs.append((a, b))
    else:
        for label in ranked["Pair"].head(top_n):
            a, b = label.split(" ↔ ")
            pairs.append((a, b))
    return pairs


# ── Spread / z-score / signals ───────────────────────────────────────────────
def _pair_analytics(a: str, b: str, panel: pd.DataFrame, eg_cache: dict,
                    entry_z: float, exit_z: float):
    """Spread, z-score, signals and price ratio for one pair."""
    key = tuple(sorted((a, b)))
    eg = eg_cache.get(key)
    if not isinstance(eg, dict):
        eg = _eg_for_pair(a, b, panel)
        eg_cache[key] = eg
    spread = calc_spread(panel[a], panel[b], eg["hedge_ratio"], eg["constant"])
    zs = calc_zscore(spread, int(DEFAULT_WINDOW))
    signals = mean_reversion_signals(zs, entry_z=entry_z, exit_z=exit_z)
    ratio = (panel[a] / panel[b]).reindex(spread.index)
    return eg, spread, zs, signals, ratio


# ── Figures ──────────────────────────────────────────────────────────────────
def _style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _spread_bands_figure(spread: pd.Series, zs: pd.Series,
                         entry_z: float, exit_z: float) -> go.Figure:
    fig = go.Figure()
    if spread.empty or zs.empty:
        return _style(fig)
    x = spread.index
    fig.add_trace(go.Scatter(
        x=x, y=spread.to_numpy(dtype=float), name="Spread ($A-\\alpha-\\beta B$)",
        line=dict(color="#38BDF8", width=1.6), yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=zs.to_numpy(dtype=float), name="Z-Score",
        line=dict(color="#FDE047", width=1.2), yaxis="y2",
    ))
    for level, color, dash, label in (
        (entry_z, "#F43F5E", "dash", f"+{entry_z:g} (entry)"),
        (-entry_z, "#00E676", "dash", f"-{entry_z:g} (entry)"),
        (exit_z, "rgba(244,63,94,0.5)", "dot", f"+{exit_z:g} (exit)"),
        (-exit_z, "rgba(0,230,118,0.5)", "dot", f"-{exit_z:g} (exit)"),
    ):
        fig.add_hline(
            y=level, line_dash=dash, line_color=color, yref="y2",
            annotation_text=label, annotation_position="top left",
        )
    fig.update_layout(
        yaxis=dict(title="Spread", domain=[0, 1]),
        yaxis2=dict(title="Z-Score", overlaying="y", side="right", showgrid=False),
    )
    fig.update_xaxes(title="Date")
    return _style(fig, 440)


def _price_ratio_figure(ratio: pd.Series) -> go.Figure:
    fig = go.Figure()
    if ratio is None or ratio.empty:
        return _style(fig)
    r = ratio.dropna().astype(float)
    if r.empty:
        return _style(fig)
    fig.add_trace(go.Scatter(
        x=r.index, y=r.to_numpy(), name="Price ratio (A / B)",
        line=dict(color="#A78BFA", width=1.8),
    ))
    mean = float(r.mean())
    if np.isfinite(mean):
        fig.add_hline(
            y=mean, line_dash="dot", line_color="rgba(255,255,255,0.4)",
            annotation_text=f"mean {mean:.3f}",
        )
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="Close A / Close B")
    return _style(fig)


def _signals_figure(spread: pd.Series, signals: pd.Series) -> go.Figure:
    fig = go.Figure()
    if spread.empty or signals.empty:
        return _style(fig)
    x = spread.index
    fig.add_trace(go.Scatter(
        x=x, y=spread.to_numpy(dtype=float), name="Spread",
        line=dict(color="#38BDF8", width=1.4),
    ))
    sig = signals.to_numpy(dtype=int)
    sp = spread.to_numpy(dtype=float)
    long_idx = np.where(np.diff(sig, prepend=0) == 1)[0]
    short_idx = np.where(np.diff(sig, prepend=0) == -1)[0]
    if len(long_idx):
        fig.add_trace(go.Scatter(
            x=x[long_idx], y=sp[long_idx], mode="markers", name="Enter long spread",
            marker=dict(symbol="triangle-up", color="#00E676", size=10),
        ))
    if len(short_idx):
        fig.add_trace(go.Scatter(
            x=x[short_idx], y=sp[short_idx], mode="markers", name="Enter short spread",
            marker=dict(symbol="triangle-down", color="#F43F5E", size=10),
        ))
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="Spread")
    return _style(fig)


def _correlation_scatter_figure(a: pd.Series, b: pd.Series) -> go.Figure:
    fig = go.Figure()
    if a is None or b is None or a.empty or b.empty:
        return _style(fig)
    joined = pd.concat([a.rename("A"), b.rename("B")], axis=1, join="inner").dropna()
    ret = joined.pct_change().dropna()
    if len(ret) < 3:
        return _style(fig)
    x = ret["A"].to_numpy(dtype=float)
    y = ret["B"].to_numpy(dtype=float)
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers", name="Daily returns",
        marker=dict(size=3.5, color="rgba(56,189,248,0.55)"),
    ))
    try:
        fit = np.polyfit(x, y, 1)
        span = np.array([np.nanmin(x), np.nanmax(x)])
        fig.add_trace(go.Scatter(
            x=span, y=np.polyval(fit, span), mode="lines", name="OLS fit",
            line=dict(color="#FDE047", width=1.5, dash="dash"),
        ))
    except Exception:
        fit = None
    from scipy import stats as _stats
    rho = float("nan")
    if len(x) > 2 and float(np.std(x)) > 0 and float(np.std(y)) > 0:
        rho = float(_stats.pearsonr(x, y).statistic)
    note = f"Pearson ρ (returns): {_fmt(rho, 3)} · n={len(x)}"
    fig.add_annotation(
        xref="paper", yref="paper", x=0.02, y=0.98, showarrow=False,
        text=note, font=dict(size=11), align="left",
    )
    fig.update_xaxes(title="Leg A daily return")
    fig.update_yaxes(title="Leg B daily return")
    return _style(fig)


def _position_label(signal: int, a: str, b: str) -> str:
    if signal == 1:
        return f"Long spread (long {a} / short {b})"
    if signal == -1:
        return f"Short spread (short {a} / long {b})"
    return "Flat"


# ── Page body ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    st.set_page_config(
        page_title="Statistical Arbitrage - QuantTerminal",
        page_icon="🔗",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_theme()

    st.title("🔗 Statistical Arbitrage")
    st.caption(
        "Identify mean-reverting pairs from OHLCV data. This page reports "
        "pair rankings, cointegration results and z-score trading signals — a "
        "historical measurement, not a forecast or a recommendation."
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.header("🔗 Pair Settings")
    exchange = st.sidebar.selectbox("Exchange", EXCHANGE_OPTIONS)
    universe_text = st.sidebar.text_area(
        "Universe",
        value="RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, ICICIBANK.NS, SBIN.NS",
        help="Comma or newline separated tickers (e.g., RELIANCE.NS, TCS.NS, "
             "HDFCBANK.NS). Bare symbols are matched to the snapshot where "
             "possible.",
    )
    method = st.sidebar.selectbox("Pair Search Method", SEARCH_METHODS)
    top_n = st.sidebar.slider(
        "Top Pairs",
        min_value=TOP_PAIRS_MIN,
        max_value=TOP_PAIRS_MAX,
        value=TOP_PAIRS_DEFAULT,
    )
    entry_z = st.sidebar.slider(
        "Entry Z-Score",
        min_value=ENTRY_Z_MIN,
        max_value=ENTRY_Z_MAX,
        value=ENTRY_Z_DEFAULT,
        step=0.1,
    )
    exit_z = st.sidebar.slider(
        "Exit Z-Score",
        min_value=EXIT_Z_MIN,
        max_value=EXIT_Z_MAX,
        value=EXIT_Z_DEFAULT,
        step=0.1,
    )

    # ── Guardrails ───────────────────────────────────────────────────────────
    if entry_z <= exit_z:
        st.warning("Entry Z-Score must be greater than Exit Z-Score.")
        st.stop()

    raw_tickers = _parse_tickers(universe_text)
    if len(raw_tickers) < MIN_UNIVERSE:
        st.warning("Enter at least two tickers to run pair analysis.")
        st.stop()

    nse, bse, glob = _snapshot_symbols(exchange)
    tickers = [t for t in (_resolve_ticker(t, exchange, nse, bse, glob) for t in raw_tickers) if t]
    dedup = list(dict.fromkeys(tickers))

    if len(dedup) > MAX_TICKERS:
        st.warning(
            f"Universe cap: analyzing the first {MAX_TICKERS} of {len(dedup)} "
            "tickers for response time (REVIEW-LATER P10-8)."
        )
        dedup = dedup[:MAX_TICKERS]
    if len(dedup) < MIN_UNIVERSE:
        st.warning("At least two parseable tickers are needed.")
        st.stop()

    frames = {t: _load_ohlcv(t) for t in dedup}
    frames = {t: d for t, d in frames.items() if not d.empty}
    if len(frames) < MIN_UNIVERSE:
        st.warning("No OHLCV data could be loaded for at least two tickers.")
        st.stop()

    panel, valid_tickers = _build_close_panel(frames)
    if panel is None:
        st.warning(
            "Fewer than two tickers share enough trading days. Alignments "
            "inner-join on common dates (DATA_LAYER)."
        )
        st.stop()

    dropped = [t for t in dedup if t not in valid_tickers]
    if dropped:
        st.sidebar.warning(f"{len(dropped)} ticker(s) dropped for missing data.")
    st.sidebar.success(f"Loaded {len(valid_tickers)} ticker(s)")

    st.caption(
        "Universe: current snapshot constituents and user-provided tickers — "
        "current-constituents only, backfill/survivorship applies (B1). "
        f"Aligned window: {_fmt_date(panel.index[0])} → "
        f"{_fmt_date(panel.index[-1])} ({len(panel)} trading days)."
    )

    # ── Pair Rankings ─────────────────────────────────────────────────────────
    st.subheader("Pair Rankings")
    eg_cache = {}
    skipped_pairs = []
    if method == "Correlation":
        ranked = _rank_by_correlation(panel)
        if ranked.empty:
            ranked = pd.DataFrame()
        else:
            ranked = ranked.head(top_n)
        caption = (
            "Pearson Distance 1−|ρ| on daily simple returns (B4-safe; "
            f"{_fmt_pair(panel.columns[0], panel.columns[1])}-style pairwise "
            "inner-join via statarb/pairs.find_pairs). Lower distance = more "
            "similar."
        )
    elif method == "Distance":
        ranked = _rank_by_distance(panel)
        if not ranked.empty:
            ranked = ranked.head(top_n)
        caption = (
            "Euclidean Distance on normalized price levels (P_t / P_0, "
            "STATISTICAL_MODELS §9). Lower distance = closer paths."
        )
    else:  # Cointegration
        ranked, skipped_pairs = _rank_by_cointegration(panel, eg_cache)
        if not ranked.empty:
            ranked = ranked.head(top_n)
        caption = (
            "Engle-Granger (statarb/cointegration) ranked by MacKinnon p-value "
            "ascending. Pairs with fewer than 100 aligned observations are "
            "skipped (documented EG floor)."
        )

    if ranked.empty:
        st.info("No pairs could be ranked by the selected method.")
        st.stop()
    show_ranked = ranked.copy()
    for col in ("Correlation", "Distance", "Coint p", "Hedge Ratio"):
        if col in show_ranked.columns:
            show_ranked[col] = show_ranked[col].apply(lambda v: _fmt(v, 4))
    if "n" in show_ranked.columns:
        show_ranked["n"] = show_ranked["n"].apply(str)
    st.dataframe(show_ranked, use_container_width=True, hide_index=True)
    st.caption(caption)
    if skipped_pairs:
        st.info(
            f"{len(skipped_pairs)} pair(s) skipped (insufficient aligned "
            "observations for Engle-Granger)."
        )

    # ── Cointegration Results ─────────────────────────────────────────────────
    st.subheader("Cointegration Results")
    pick_pairs = _selected_pairs(ranked, method, top_n)
    coint_df = _build_coint_table(pick_pairs, panel, eg_cache)
    if coint_df.empty:
        st.info(
            "Engle-Granger results require at least one pair with 100+ aligned "
            "observations."
        )
    else:
        st.dataframe(coint_df, use_container_width=True, hide_index=True)
        st.caption(
            "Engle-Granger: statistic, MacKinnon p-value, β hedge ratio "
            "(spread = A − α − β·B), and half-life from the differenced "
            "regression Δspread = θ·spread + ε (STATISTICAL_MODELS §10). "
            "Speed comes from the §10 table; Confidence is the §10"
            " MODEL_CONFIDENCE badge (observations · EG p-value · half-life) "
            "combined as the worst check (REVIEW-LATER P10-5). Is Cointegrated "
            "= 5% EG rejection (B4 gate)."
        )

    # ── Selected pair: current spread ─────────────────────────────────────────
    st.divider()
    st.subheader("Current Spread — Selected Pair")
    if not pick_pairs:
        st.info("No analyzable pairs available.")
        st.stop()
    pair_labels = [_fmt_pair(a, b) for a, b in pick_pairs]
    chosen_label = st.selectbox("Analyze pair", pair_labels)
    a, b = chosen_label.split(" ↔ ")

    try:
        eg, spread, zs, signals, ratio = _pair_analytics(
            a, b, panel, eg_cache, entry_z, exit_z
        )
    except (ValueError, TypeError) as exc:
        st.error(f"Pair analytics failed: {exc}")
        st.stop()

    hl_result = half_life(eg["residuals"])
    hl_days = hl_result["half_life"]
    speed = hl_result["interpretation"] or "Not mean-reverting"
    badge, badge_label, _checks = _coint_badge(eg, _to_float(hl_days))

    z_last = _to_float(zs.iloc[-1]) if not zs.empty else None
    spread_last = _to_float(spread.iloc[-1]) if not spread.empty else None
    sig_last = int(signals.iloc[-1]) if not signals.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Spread", _fmt(spread_last, 4))
    c2.metric("Current Z-Score", _fmt(z_last, 3))
    c3.metric("Position", _position_label(sig_last, a, b))
    c4.metric(
        "Half-Life",
        f"{_fmt(hl_days, 1)} d" if _to_float(hl_days) is not None else "N/A",
        help=f"{badge} {badge_label} confidence · {speed}",
    )

    if spread.empty or zs.empty:
        st.info(
            "Spread/z-score series are empty for this pair."
        )
    else:
        tail = pd.DataFrame({
            "Date": [_fmt_date(d) for d in spread.index[-10:]],
            "Spread": spread.iloc[-10:].apply(lambda v: _fmt(v, 4)),
            "Z-Score": zs.iloc[-10:].apply(lambda v: _fmt(v, 3)),
            "Signal": signals.iloc[-10:].apply(
                lambda v: {1: "Long", -1: "Short", 0: "Flat"}[int(v)]
            ),
        })
        st.dataframe(tail, use_container_width=True, hide_index=True)
        st.caption(
            f"Z-Score window = {DEFAULT_WINDOW} trading days (documented Mean "
            "Reversion lookback default; no §10 control exists — P10-7). "
            "+1 = long the spread (long {A}, short {B}); −1 = short the spread; "
            "0 = flat. Entry at ±{entry_z:g}, exit within ±{exit_z:g} "
            "(statarb/spread.mean_reversion_signals). A short leg implies "
            "short-selling; India retail default is long-only, and global runs "
            "carry a US margin/borrow-cost caveat (B6).".format(
                A=a, B=b, entry_z=entry_z, exit_z=exit_z
            )
        )

    # ── Charts ────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Spread with Z-Score Bands")
    st.plotly_chart(
        _spread_bands_figure(spread, zs, entry_z, exit_z),
        use_container_width=True,
    )

    st.subheader("Price Ratio")
    st.plotly_chart(_price_ratio_figure(ratio), use_container_width=True)

    st.subheader("Trading Signals on Spread")
    st.plotly_chart(_signals_figure(spread, signals), use_container_width=True)

    st.subheader("Correlation Scatter (Both Legs)")
    st.plotly_chart(
        _correlation_scatter_figure(panel[a], panel[b]), use_container_width=True
    )

    # ── Bias controls / REVIEW-LATER ─────────────────────────────────────────
    with st.expander("Bias controls (B4 · B1 · B6) and REVIEW-LATER"):
        st.markdown(
            "- **B4 Spurious-regression gate** — implemented: pair correlations "
            "use daily simple returns, not trending levels (statarb/pairs.py); "
            "level relationships are only reported through the Engle-Granger "
            "gate (`Is Cointegrated`, 5% decision); no bare R² is emitted for "
            "trending price levels. BIAS_MITIGATION.md Part C maps B4 → "
            "Page 10.\n"
            "- **B1 Point-in-time guard** — universes are current-constituents "
            "only; `data/fundamentals.py` / `data/snapshot.py` are absent on "
            "this branch, so only the documented label is shown.\n"
            "- **B6 Backtest realism** — this page shows signals, not a "
            "backtest; shorting a leg assumes short-sale availability (India "
            "retail long-only default; US margin/borrow-cost caveat). No "
            "liquidity/cost modeling is performed here.\n"
            "- **B2/B3/B5/B10/B11** — Part C maps these to pages other than 10; "
            "no Page-10 requirement exists and none is fabricated.\n"
            "- **Confidence badge** — MODEL_CONFIDENCE.md gives no combination "
            "rule or floor for the cointegration checks; worst-of and "
            "NaN-half-life-as-Low are inferences (P10-5/P10-6).\n"
            "- **Pair search naming** — §10 'Correlation'/'Distance' map to "
            "find_pairs aliases pearson/euclidean; 'Cointegration' loops "
            "engle_granger per pair (no shared ranked-by-p API exists) "
            "(P10-3/P10-4).\n"
            "- **Z-Score window** — fixed at 20 days, the documented Mean "
            "Reversion lookback default; §10 lists no z-window control (P10-7).\n"
            "- **Ticker cap** — 25-ticker guard with disclosure (P10-8); "
            "bare-symbol resolution is a local helper (P10-8).\n"
            "- **ARCHITECTURE.md** lists `statarb/__init__.py`, which the repo "
            "does not contain; shared statarb modules are imported unmodified "
            "(P10-2)."
        )