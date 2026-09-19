"""Page 9 — Factor Research.

Contract sources:
    docs/STREAMLIT_PAGES.md §9    - sidebar controls, tables and charts.
    docs/FACTOR_RESEARCH.md       - factor formulas (§1-5), parameter
                                    defaults/ranges, look-ahead prevention
                                    ("Close(t) -> Open(t+1)"), ranking methods,
                                    IC definition/output, Factor Returns table
                                    columns, visual outputs.
    docs/BIAS_MITIGATION.md       - B1 (current-constituents label), B2
                                    (Newey-West IC t-stat, lags rule).
    docs/MODEL_CONFIDENCE.md      - no factor-research badge rule exists.

Factor formulas/ranking/IC are consumed from the shared APIs only:
    factor/factors.py  - momentum_factor, trend_factor, vol_factor,
                         reversal_factor, liquidity_factor
    factor/scores.py   - factor_rankings, information_coefficient,
                         newey_west_lags

REVIEW-LATER (documented conflicts/dependencies; also surfaced in-page):
    P9-1  Page filename inferred from repository naming convention; no doc
          explicitly pins pages/09_Factor_Research.py.
    P9-2  B3/VIF conflict: BIAS_MITIGATION.md Part C maps B3 to Page 9, but
          neither FACTOR_RESEARCH.md nor STREAMLIT_PAGES §9 specifies VIF and
          machine_learning/features.py is absent -> no VIF implemented.
    P9-3  "New Capabilities" (Stock Screener, Composite Health-Score)
          assigned to Page 9 but require absent config.py /
          data/fundamentals.py / data/snapshot.py / badge helper -> not
          implemented (only the Factor Research spec is built).
    P9-4  B11 maps to a nonexistent factor/ic.py; IC lives in
          factor/scores.py.
    P9-5  B1 provenance files (data/fundamentals.py, data/snapshot.py) are
          absent; this page is OHLCV-only, so only the documented
          "current-constituents only" label is shown.
    P9-6  IC headline t-stat ambiguity: t_stat_icir vs t_stat_nw. Both are
          reported; no single headline is claimed.
    P9-7  IC uses the shared information_coefficient() API, whose forward
          return is Close(t+1)..Close(t+1+h). FACTOR_RESEARCH L98/L100
          describe positions opened at Open(t+1); the portfolio table uses
          Open(t+1)..Open(t+1+h) since Open data is available. The two
          conventions differ and both are disclosed in the UI.
    P9-8  B4/B5/B10 are assigned to Pranav at the person level, but
          BIAS_MITIGATION.md Part C maps them to pages other than 9; no
          Page-9 requirement exists.
    P9-9  FACTOR_RESEARCH.md "Visual Outputs" lists a quintile/decile return
          bar chart that STREAMLIT_PAGES §9 does not; the page spec governs
          so it is omitted.
    P9-10 Page-9 data lookback is unspecified; 5y is used to cover the
          documented momentum lookback maximum (756 trading days).
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
    from factor.factors import (
        momentum_factor,
        trend_factor,
        vol_factor,
        reversal_factor,
        liquidity_factor,
        MOMENTUM_LOOKBACK,
        MIN_MOMENTUM_LOOKBACK,
        MAX_MOMENTUM_LOOKBACK,
        MOMENTUM_SKIP_MONTHS,
        MIN_MOMENTUM_SKIP,
        MAX_MOMENTUM_SKIP,
        TREND_FAST_WINDOW,
        MIN_TREND_FAST,
        MAX_TREND_FAST,
        TREND_SLOW_WINDOW,
        MIN_TREND_SLOW,
        MAX_TREND_SLOW,
        VOL_WINDOW,
        MIN_VOL_WINDOW,
        MAX_VOL_WINDOW,
        VOL_ESTIMATORS,
        REVERSAL_LOOKBACK,
        MIN_REVERSAL_LOOKBACK,
        MAX_REVERSAL_LOOKBACK,
        LIQUIDITY_WINDOW,
        MIN_LIQUIDITY_WINDOW,
        MAX_LIQUIDITY_WINDOW,
    )
    from factor.scores import (
        factor_rankings,
        information_coefficient,
        newey_west_lags,
        RANKING_METHODS,
        MIN_CROSS_SECTION,
    )
except ImportError:
    from factors import (
        momentum_factor,
        trend_factor,
        vol_factor,
        reversal_factor,
        liquidity_factor,
        MOMENTUM_LOOKBACK,
        MIN_MOMENTUM_LOOKBACK,
        MAX_MOMENTUM_LOOKBACK,
        MOMENTUM_SKIP_MONTHS,
        MIN_MOMENTUM_SKIP,
        MAX_MOMENTUM_SKIP,
        TREND_FAST_WINDOW,
        MIN_TREND_FAST,
        MAX_TREND_FAST,
        TREND_SLOW_WINDOW,
        MIN_TREND_SLOW,
        MAX_TREND_SLOW,
        VOL_WINDOW,
        MIN_VOL_WINDOW,
        MAX_VOL_WINDOW,
        VOL_ESTIMATORS,
        REVERSAL_LOOKBACK,
        MIN_REVERSAL_LOOKBACK,
        MAX_REVERSAL_LOOKBACK,
        LIQUIDITY_WINDOW,
        MIN_LIQUIDITY_WINDOW,
        MAX_LIQUIDITY_WINDOW,
    )
    from scores import (
        factor_rankings,
        information_coefficient,
        newey_west_lags,
        RANKING_METHODS,
        MIN_CROSS_SECTION,
    )

EXCHANGE_OPTIONS = ("Auto", "NSE", "BSE", "Global")
FACTOR_OPTIONS = ("Momentum", "Trend", "Volatility", "Reversal", "Liquidity")
RANKING_OPTIONS = ("Quintile", "Decile")
_RANKING_METHOD = {"Quintile": "quintile", "Decile": "decile"}

# Lookback is not specified by the Page 9 docs; 5y covers the documented
# momentum lookback maximum (756 days) plus a rebalance window (P9-10).
LOOKBACK_PERIOD = "5y"
LOOKBACK_INTERVAL = "1d"

# Rebalance frequency defaults/ranges (FACTOR_RESEARCH.md §1-5):
#   momentum / trend / volatility: 21-252, default 63
#   reversal / liquidity:           1-63,  default 21
_REBALANCE_SPECS = {
    "Momentum": (21, 252, 63),
    "Trend": (21, 252, 63),
    "Volatility": (21, 252, 63),
    "Reversal": (1, 63, 21),
    "Liquidity": (1, 63, 21),
}


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


# ── Universe & data ──────────────────────────────────────────────────────────
def _build_universe(exchange: str) -> pd.DataFrame:
    """(ticker, label) universe from the snapshot datasets (Page 5 pattern).

    Auto -> India NSE + Global symbols; NSE/BSE filter the India snapshot;
    Global keeps only the international set.
    """
    ind = fetch_stocks("India")
    us = fetch_stocks("US")
    rows = []

    if not ind.empty and exchange in ("Auto", "NSE"):
        nse = ind[ind["Exchange"] == "NSE"]
        for _, r in nse.iterrows():
            ticker = f"{str(r['Symbol']).strip()}.NS"
            rows.append((ticker, f"{r['Description']} ({ticker})"))

    if not ind.empty and exchange == "BSE":
        bse = ind[ind["Exchange"] == "BSE"]
        for _, r in bse.iterrows():
            ticker = f"{str(r['Symbol']).strip()}.BO"
            rows.append((ticker, f"{r['Description']} ({ticker})"))

    if not us.empty and exchange in ("Auto", "Global"):
        for _, r in us.iterrows():
            ticker = str(r["Symbol"]).strip()
            rows.append((ticker, f"{r['Description']} ({ticker})"))

    frame = pd.DataFrame(rows, columns=["ticker", "label"]).drop_duplicates(
        subset=["ticker"]
    )
    if not frame.empty:
        frame = frame[frame["label"].notna()]
        frame["label"] = frame["label"].astype(str)
    return frame


@st.cache_data(show_spinner=False)
def _load_ohlcv(ticker: str) -> pd.DataFrame:
    df = load_data(ticker, period=LOOKBACK_PERIOD, interval=LOOKBACK_INTERVAL)
    if df is None or df.empty or "Close" not in df.columns:
        return pd.DataFrame()
    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    out = df[keep].copy()
    out.index = pd.to_datetime(out.index)
    return out


def _required_fields(factor: str, params: dict) -> list:
    """OHLCV fields needed to build the selected factor score panel."""
    fields = ["Close"]
    if factor == "Volatility":
        fields += ["High", "Low"]
        if params.get("vol_estimator") == "gk":
            fields.append("Open")
    elif factor == "Liquidity":
        fields.append("Volume")
    return fields


def _build_panels(frames: dict, fields: list):
    """Align multi-asset panels on shared (inner-join) trading days.

    A ticker is valid only if it has every required factor field. Panels for
    each field share the same complete index; Open is attached when present so
    portfolio returns can use Open(t+1) positioning.
    """
    valid = {
        t: df
        for t, df in frames.items()
        if all(f in df.columns for f in fields) and len(df) >= 2
    }
    if not valid:
        return {}, []

    idx = None
    for df in valid.values():
        d = df[fields].dropna(how="any")
        idx = d.index if idx is None else idx.intersection(d.index)
    if idx is None or len(idx) < 2:
        return {}, []

    all_fields = list(dict.fromkeys(fields + ["Open"]))
    panels = {}
    for f in all_fields:
        cols = {
            t: df[f].reindex(idx)
            for t, df in valid.items()
            if f in df.columns
        }
        panels[f] = pd.DataFrame(cols).sort_index() if cols else pd.DataFrame()
    return panels, sorted(valid.keys())


def _factor_scores(factor: str, panels: dict, params: dict):
    """Dispatch to the shared factor API for the selected factor."""
    close = panels["Close"]
    if factor == "Momentum":
        return momentum_factor(
            close, lookback=params["lookback"], skip_months=params["skip"]
        )
    if factor == "Trend":
        return trend_factor(
            close, fast_window=params["fast"], slow_window=params["slow"]
        )
    if factor == "Volatility":
        return vol_factor(
            close,
            panels["High"],
            panels["Low"],
            open_=panels.get("Open"),
            vol_window=params["vol_window"],
            vol_estimator=params["vol_estimator"],
        )
    if factor == "Reversal":
        return reversal_factor(close, lookback=params["lookback"])
    return liquidity_factor(panels["Volume"], volume_window=params["volume_window"])


# ── Factor analytics ─────────────────────────────────────────────────────────
def _rebalance_dates(scores: pd.DataFrame, freq: int) -> pd.Index:
    """Rebalance dates from the first cross-section with enough finite scores.

    Scores are daily (factor/factors.py); sampling every `freq` rows yields the
    documented rebalance schedule without selecting dates the factor cannot
    score yet (warm-up rows are NaN).
    """
    counts = scores.notna().sum(axis=1)
    valid = scores.index[counts >= MIN_CROSS_SECTION]
    if len(valid) < 2:
        return pd.Index([])
    start = scores.index.get_loc(valid[0])
    return scores.index[start::freq]


def _portfolio_returns(
    scores: pd.DataFrame,
    rebalance_dates: pd.Index,
    open_px: pd.DataFrame,
    freq: int,
    n_groups: int,
    method: str,
) -> pd.DataFrame:
    """Long top / short bottom portfolio returns using Open(t+1) positioning.

    For rebalance date t: rank the cross-section, enter both legs at
    Open(t+1) and value at Open(t+1+h). Columns follow FACTOR_RESEARCH.md's
    Factor Returns table (Date, Long, Short, Factor Return, Spread).
    """
    if open_px is None or open_px.empty:
        return pd.DataFrame()
    rows = []
    for t in rebalance_dates:
        if t not in open_px.index:
            continue
        pos = open_px.index.get_loc(t)
        start, end = pos + 1, pos + freq + 1
        if end >= len(open_px):
            continue
        cross = scores.loc[t].dropna()
        if cross.empty:
            continue
        groups = factor_rankings(cross, method)
        fwd = (open_px.iloc[end] / open_px.iloc[start] - 1.0).astype(float)
        long_ret = fwd.reindex(groups.index[groups == n_groups]).dropna()
        short_ret = fwd.reindex(groups.index[groups == 1]).dropna()
        if long_ret.empty or short_ret.empty:
            continue
        lv, sv = float(long_ret.mean()), float(short_ret.mean())
        rows.append({
            "Date": t,
            "Long Return": lv,
            "Short Return": sv,
            "Factor Return": lv - sv,
            "Spread": lv - sv,
        })
    return pd.DataFrame(rows)


def _latest_rankings(scores: pd.DataFrame, method: str):
    """Most recent cross-section: Asset, factor score, rank, rank group."""
    valid = scores.dropna(how="all")
    if valid.empty:
        return pd.DataFrame(), None
    latest = valid.index[-1]
    cross = valid.loc[latest].dropna()
    if cross.empty:
        return pd.DataFrame(), latest
    groups = factor_rankings(cross, method)
    ranks = cross.rank(ascending=False, method="average")
    frame = pd.DataFrame({
        "Asset": [str(a) for a in cross.index],
        "Factor Score": cross.to_numpy(),
        "Rank": ranks.to_numpy(),
        "Group": groups.to_numpy(),
    })
    return frame, latest


# ── Figures ──────────────────────────────────────────────────────────────────
def _style(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _cumulative_figure(port: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if port.empty:
        return _style(fig, 400)
    for col, name, color in (
        ("Long Return", "Long (top group)", "#00E676"),
        ("Short Return", "Short (bottom group)", "#F43F5E"),
        ("Factor Return", "Long - Short", "#38BDF8"),
    ):
        cum = (1.0 + port[col].astype(float)).cumprod()
        fig.add_trace(go.Scatter(
            x=port["Date"], y=cum, mode="lines", name=name,
            line=dict(color=color, width=2),
        ))
    fig.add_hline(y=1.0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
    fig.update_yaxes(title="Growth of 1.0")
    return _style(fig, 400)


def _ic_figure(ic_series: pd.Series) -> go.Figure:
    fig = go.Figure()
    if ic_series is None or ic_series.empty:
        return _style(fig)
    fig.add_trace(go.Bar(
        x=ic_series.index, y=ic_series.to_numpy(dtype=float),
        name="IC", marker_color="rgba(56,189,248,0.7)",
    ))
    fig.add_hline(y=0.0, line_dash="dot", line_color="rgba(255,255,255,0.35)")
    fig.update_yaxes(title="Spearman IC")
    return _style(fig, 360)


def _score_dist_figure(cross: pd.Series) -> go.Figure:
    fig = go.Figure()
    if cross is None or cross.empty:
        return _style(fig)
    fig.add_trace(go.Histogram(
        x=cross.astype(float).to_numpy(),
        name="Factor scores",
        marker_color="rgba(0,230,118,0.55)",
    ))
    fig.update_xaxes(title="Factor score")
    fig.update_yaxes(title="Assets")
    return _style(fig, 360)


# ── Page body ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    st.set_page_config(
        page_title="Factor Research - QuantTerminal",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_theme()

    st.title("🧬 Factor Research")
    st.caption(
        "Construct and evaluate long-short factor portfolios from OHLCV data. "
        "This is a historical measurement, not a forecast or a recommendation."
    )

    # ── Sidebar: universe ────────────────────────────────────────────────────
    st.sidebar.header("📊 Data Source")
    exchange = st.sidebar.selectbox("Exchange", EXCHANGE_OPTIONS)
    universe = _build_universe(exchange)

    if universe.empty:
        st.sidebar.warning("No assets found for the selected exchange.")
    else:
        st.sidebar.markdown(f"📊 **Available**: `{len(universe):,}` assets")

    selected_labels = st.sidebar.multiselect("Tickers", universe["label"].tolist())
    tickers = universe.loc[
        universe["label"].isin(selected_labels), "ticker"
    ].tolist()

    # ── Sidebar: factor settings ─────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.subheader("🧬 Factor Settings")
    factor = st.sidebar.selectbox("Factor", FACTOR_OPTIONS)
    params = {}

    if factor == "Momentum":
        params["lookback"] = st.sidebar.slider(
            "Lookback (trading days)",
            min_value=MIN_MOMENTUM_LOOKBACK,
            max_value=MAX_MOMENTUM_LOOKBACK,
            value=MOMENTUM_LOOKBACK,
        )
        params["skip"] = st.sidebar.slider(
            "Skip Months",
            min_value=MIN_MOMENTUM_SKIP,
            max_value=MAX_MOMENTUM_SKIP,
            value=MOMENTUM_SKIP_MONTHS,
        )
    elif factor == "Trend":
        params["fast"] = st.sidebar.slider(
            "Fast Window",
            min_value=MIN_TREND_FAST,
            max_value=MAX_TREND_FAST,
            value=TREND_FAST_WINDOW,
        )
        params["slow"] = st.sidebar.slider(
            "Slow Window",
            min_value=MIN_TREND_SLOW,
            max_value=MAX_TREND_SLOW,
            value=TREND_SLOW_WINDOW,
        )
    elif factor == "Volatility":
        params["vol_window"] = st.sidebar.slider(
            "Volatility Window",
            min_value=MIN_VOL_WINDOW,
            max_value=MAX_VOL_WINDOW,
            value=VOL_WINDOW,
        )
        params["vol_estimator"] = st.sidebar.selectbox(
            "Volatility Estimator", list(VOL_ESTIMATORS)
        )
    elif factor == "Reversal":
        params["lookback"] = st.sidebar.slider(
            "Lookback (trading days)",
            min_value=MIN_REVERSAL_LOOKBACK,
            max_value=MAX_REVERSAL_LOOKBACK,
            value=REVERSAL_LOOKBACK,
        )
    else:  # Liquidity
        params["volume_window"] = st.sidebar.slider(
            "Volume Window",
            min_value=MIN_LIQUIDITY_WINDOW,
            max_value=MAX_LIQUIDITY_WINDOW,
            value=LIQUIDITY_WINDOW,
        )

    reb_min, reb_max, reb_default = _REBALANCE_SPECS[factor]
    rebalance_freq = st.sidebar.slider(
        "Rebalance Frequency (trading days)",
        min_value=reb_min,
        max_value=reb_max,
        value=reb_default,
    )

    st.sidebar.divider()
    ranking = st.sidebar.selectbox("Ranking", RANKING_OPTIONS)
    method = _RANKING_METHOD[ranking]
    n_groups = RANKING_METHODS[method]

    # ── Guardrails ───────────────────────────────────────────────────────────
    if not tickers:
        st.warning("Select at least one asset universe to run factor research.")
        st.stop()

    if factor == "Trend" and params["fast"] >= params["slow"]:
        st.warning("Fast Window must be smaller than Slow Window.")
        st.stop()

    frames = {t: _load_ohlcv(t) for t in tickers}
    frames = {t: d for t, d in frames.items() if not d.empty}
    if not frames:
        st.warning("No data could be loaded for the selected assets.")
        st.stop()

    fields = _required_fields(factor, params)
    panels, valid_tickers = _build_panels(frames, fields)
    if not panels or panels.get("Close", pd.DataFrame()).empty:
        st.warning(
            "Insufficient aligned OHLCV data. Factor research needs at least "
            f"{MIN_CROSS_SECTION} assets (or 2 for ranking) sharing trading "
            "days and the required columns "
            f"({', '.join(fields)})."
        )
        st.stop()

    dropped = [t for t in frames if t not in valid_tickers]
    if dropped:
        st.sidebar.warning(
            f"{len(dropped)} asset(s) dropped for missing data/fields."
        )
    st.sidebar.success(f"Loaded {len(valid_tickers)} asset(s)")

    close = panels["Close"]

    # ── Factor score panel ───────────────────────────────────────────────────
    try:
        scores = _factor_scores(factor, panels, params)
    except (ValueError, TypeError) as exc:
        st.error(f"Factor computation failed: {exc}")
        st.stop()

    if isinstance(scores, pd.Series):
        scores = scores.to_frame(name=close.columns[0])
    scores = scores.reindex(close.index)
    scores = scores.reindex(columns=close.columns)

    if scores.dropna(how="all").empty:
        st.warning("Factor scores are all NaN for the selected window/parameters.")
        st.stop()

    st.subheader(f"🧬 {factor} Factor — {ranking} Ranking")
    st.caption(
        f"Universe (current-constituents only, backfill/survivorship applies): "
        f"{len(valid_tickers)} assets · window(s) from sidebar · rebalance "
        f"every {int(rebalance_freq)} trading days."
    )

    # ── Factor Rankings (latest cross-section) ───────────────────────────────
    rankings_df, latest_date = _latest_rankings(scores, method)
    st.subheader("Factor Rankings")
    if rankings_df.empty:
        st.info("No finite factor scores to rank.")
    else:
        show = rankings_df.copy()
        show["Factor Score"] = show["Factor Score"].apply(lambda v: _fmt(v, 5))
        show["Rank"] = show["Rank"].apply(lambda v: _fmt(v, 1))
        show["Group"] = show["Group"].apply(
            lambda v: "N/A" if _to_float(v) is None else f"{int(float(v))}"
        )
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.caption(
            f"Cross-section as of {_fmt_date(latest_date)}. Rank 1 = highest "
            f"factor score. Group ascends 1 = bottom, {n_groups} = top "
            "(factor/scores.factor_rankings); the factor is long the top group "
            "and short the bottom group."
        )

    # ── Factor IC ────────────────────────────────────────────────────────────
    st.subheader("Factor IC")
    rebalance_dates = _rebalance_dates(scores, int(rebalance_freq))
    ic_info = None
    ic_error = None
    if len(rebalance_dates) >= 2:
        try:
            ic_info = information_coefficient(
                scores.loc[rebalance_dates], close, int(rebalance_freq)
            )
        except (ValueError, TypeError) as exc:
            ic_error = str(exc)

    if ic_error is not None:
        st.info(f"Factor IC unavailable: {ic_error}")
    elif ic_info is None:
        st.info(
            "Factor IC needs at least two rebalance dates with "
            f"{MIN_CROSS_SECTION}+ scored assets; widen the lookback or reduce "
            "the rebalance frequency."
        )
    else:
        ic_series = ic_info["ic"]
        ic_show = pd.DataFrame({
            "IC": ic_series.apply(lambda v: _fmt(v, 4)),
            "ICIR (rolling)": ic_info["icir"].apply(lambda v: _fmt(v, 4)),
            "Cumulative IC": ic_info["cumulative_ic"].apply(lambda v: _fmt(v, 4)),
        })
        ic_show.index = [_fmt_date(d) for d in ic_show.index]
        st.dataframe(ic_show, use_container_width=True)

        summary = ic_info["summary"]
        try:
            lags = str(newey_west_lags(int(summary["n_periods"])))
        except (TypeError, ValueError):
            lags = "N/A"
        summary_rows = pd.DataFrame([
            ("Evaluable rebalance periods", f'{int(summary["n_periods"])}'),
            ("Mean IC", _fmt(summary["mean_ic"], 4)),
            ("Std IC", _fmt(summary["std_ic"], 4)),
            ("ICIR (mean / std)", _fmt(summary["icir"], 4)),
            ("Newey-West HAC lags (B2)", lags),
            ("t-stat (ICIR * sqrt(n))", _fmt(summary["t_stat_icir"], 3)),
            ("t-stat (Newey-West robust)", _fmt(summary["t_stat_nw"], 3)),
        ], columns=["IC statistic", "Value"])
        st.dataframe(summary_rows, use_container_width=True, hide_index=True)
        st.caption(
            "IC = Spearman rank correlation between scores at t and the forward "
            "rebalance-period return (t+1 .. t+1+freq, via "
            "information_coefficient). B2: Newey-West (HAC) lags = "
            "floor(4·(n/100)^(2/9)) via newey_west_lags. Both the ICIR-implied "
            "and Newey-West robust t-statistics are reported; the docs do not "
            "designate a single headline, so none is claimed (see REVIEW-LATER)."
        )

    # ── Top/Bottom Portfolio Returns ─────────────────────────────────────────
    st.subheader("Top/Bottom Portfolio Returns")
    open_px = panels.get("Open")
    if open_px is None or open_px.empty:
        st.info(
            "Open prices are unavailable, so the documented Open(t+1) "
            "positioning cannot be computed; portfolio returns are omitted "
            "rather than substituting Close."
        )
        port = pd.DataFrame()
    else:
        port = _portfolio_returns(
            scores, rebalance_dates, open_px, int(rebalance_freq), n_groups, method
        )
        if port.empty:
            st.info(
                "No rebalance date has a full forward window and both long/short "
                "legs populated."
            )
        else:
            show = port.copy()
            show["Date"] = show["Date"].apply(_fmt_date)
            for col in ("Long Return", "Short Return", "Factor Return", "Spread"):
                show[col] = show[col].apply(lambda v: _fmt_pct(v, 2))
            st.dataframe(show, use_container_width=True, hide_index=True)
            st.caption(
                "Signals are formed at Close(t); positions are entered at "
                "Open(t+1) and valued at Open(t+1+freq) — equal-weighted legs, "
                "long the top group and short the bottom group. Factor Return "
                "and Spread are both Long - Short per FACTOR_RESEARCH.md. The "
                "IC table above uses the shared API's Close-based forward "
                "return (REVIEW-LATER P9-7)."
            )

    # ── Charts ───────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Cumulative Factor Returns")
    if port.empty:
        st.info("Cumulative factor returns require the portfolio-return series.")
    else:
        st.plotly_chart(_cumulative_figure(port), use_container_width=True)

    st.subheader("IC Time Series")
    if ic_info is None:
        st.info("IC time series requires a computable Factor IC table.")
    else:
        st.plotly_chart(_ic_figure(ic_info["ic"]), use_container_width=True)

    st.subheader("Factor Score Distribution")
    dist_cross = pd.Series(dtype=float)
    if latest_date is not None:
        dist_cross = scores.loc[latest_date].dropna()
    if dist_cross.empty:
        st.info("No finite factor scores to plot.")
    else:
        st.plotly_chart(_score_dist_figure(dist_cross), use_container_width=True)

    # ── Bias controls / REVIEW-LATER ─────────────────────────────────────────
    with st.expander("Bias controls (B2 · B4 · B5 · B10) and REVIEW-LATER"):
        st.markdown(
            "- **B2 Robust SEs** — IC inference reports Newey-West (HAC) lags "
            "`floor(4·(n/100)^(2/9))` via `newey_west_lags`, with both "
            "documented t-statistics; no single headline is claimed.\n"
            "- **B4 Spurious-regression gate** — this page performs no "
            "level/OLS regression, and BIAS_MITIGATION.md Part C maps B4 to "
            "pages 5 and 10, not 9, so no gate input exists here.\n"
            "- **B5 Risk inference** — VaR/CVaR/tail-correlation rules target "
            "pages 7 and 16 (Part C); no Page-9 risk-inference computation is "
            "specified.\n"
            "- **B10 Drift management** — requires `data/loader.py` + "
            "`statistics/tests.py`, which are absent on this branch; not "
            "implemented rather than fabricating a drift metric.\n"
            "- **B1** — provenance files `data/fundamentals.py` / "
            "`data/snapshot.py` are absent; factor research here is OHLCV-only, "
            "so only the documented *current-constituents only* label is "
            "shown.\n"
            "- **B3 / VIF** — BIAS Part C maps B3 to Page 9, but no VIF is "
            "specified by `FACTOR_RESEARCH.md` or `STREAMLIT_PAGES.md §9` and "
            "`machine_learning/features.py` is absent; no undocumented VIF UI "
            "was added.\n"
            "- **Stock Screener / Composite Health-Score** — assigned to "
            "Page 9 by `STREAMLIT_PAGES.md`, but `config.py` / "
            "`data/fundamentals.py` / `data/snapshot.py` / badge helper are "
            "absent; only the Factor Research specification is built.\n"
            "- **IC t-stat headline** — `t_stat_icir` vs `t_stat_nw` is not "
            "resolved by the docs; both are reported.\n"
            "- **Open(t+1) vs Close(t+1)** — the portfolio table uses "
            "Open(t+1) positioning (Open data available); the shared IC API "
            "uses Close-based forward returns and was used unmodified."
        )
