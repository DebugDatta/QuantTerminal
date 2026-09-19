"""Streamlit Page 7 — Risk Analytics.

Contract sources:
    docs/STREAMLIT_PAGES.md §7   - sidebar controls, tables, charts
    docs/RISK_ANALYTICS.md §1-6  - formulas, parameters, drawdown periods table
    docs/BIAS_MITIGATION.md §B5  - VaR triad (CVaR primary), procyclicality,
                                   tail-correlation requirement
    docs/BIAS_MITIGATION.md §B19 - low-frequency microstructure caveat
    docs/BIAS_MITIGATION.md §B8  - state window, historical-not-forecast banner

This page is self-contained: every helper below is local to this file. No shared
API is created or modified. Where the docs conflict or a documented API is
missing (plots/risk.py, data/loader.resolve_ticker, config.BENCHMARKS,
drawdown-period API, compute_confidence_badge, B5 tail correlation), the page
degrades to N/A / an informational note and the gap is recorded in the
implementation report.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils.helper import inject_custom_theme, load_data
from core.returns import compute_returns
from core.metrics import (
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    information_ratio,
    treynor_ratio,
    beta,
    alpha,
)
from core.drawdown import drawdown_series, max_drawdown
from risk.metrics import value_at_risk, conditional_var, tail_risk
from risk.rolling import rolling_sharpe, rolling_beta


# ── Constants (documented values only) ───────────────────────────────────────
EXCHANGE_OPTIONS = ("Auto", "NSE", "BSE", "Global")
BENCHMARK_OPTIONS = ("Nifty 50", "Sensex", "Bank Nifty", "S&P 500", "NASDAQ", "None", "Custom")

# DATA_LAYER.md benchmark tickers. NASDAQ is intentionally absent: no docs/config
# pins its ticker on this branch, so it degrades to N/A (see review note R62).
BENCHMARK_MAP = {
    "Nifty 50": "^NSEI",
    "Sensex": "^BSESN",
    "Bank Nifty": "^NSEBANK",
    "S&P 500": "^GSPC",
}

LOOKBACK_PERIOD = "2y"
LOOKBACK_INTERVAL = "1d"
RISK_FREE_RATE = 0.0
PERIODS_PER_YEAR = 252

# RISK_ANALYTICS.md §5: window default 252, range 20-756 (risk/rolling.py
# MAX_WINDOW = 756). STREAMLIT_PAGES.md §7 states 20-252; the API contract is
# broader, so the slider follows the API/doc range (see review note R67).
WINDOW_MIN = 20
WINDOW_MAX = 756
WINDOW_DEFAULT = 252

CONFIDENCE_MIN = 0.90
CONFIDENCE_MAX = 0.99
CONFIDENCE_DEFAULT = 0.95


# ── Local formatting ─────────────────────────────────────────────────────────
def _fmt_ratio(value):
    return "N/A" if value is None else f"{value:.3f}"


def _fmt_num(value, digits=4):
    return "N/A" if value is None else f"{value:.{digits}f}"


def _fmt_pct(value):
    return "N/A" if value is None else f"{value * 100:.2f}%"


def _fmt_date(value):
    if value is None:
        return "N/A"
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return str(value)


def _duration_days(start, end):
    if start is None or end is None:
        return "N/A"
    try:
        return str(int((pd.Timestamp(end) - pd.Timestamp(start)).days))
    except (TypeError, ValueError):
        return "N/A"


# ── Safe metric wrappers ─────────────────────────────────────────────────────
def _safe_metric(fn, *args, **kwargs):
    """Run a scalar metric; return a finite float or None on any failure."""
    try:
        value = fn(*args, **kwargs)
    except (ValueError, TypeError, ZeroDivisionError, FloatingPointError):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _safe_dict(fn, *args, **kwargs):
    """Run a dict-returning risk function; return the dict or None."""
    try:
        result = fn(*args, **kwargs)
    except (ValueError, TypeError, ZeroDivisionError, FloatingPointError):
        return None
    return result if isinstance(result, dict) else None


# ── Ticker / benchmark resolution (local; resolve_ticker is missing) ─────────
def _is_qualified(ticker):
    return ("." in ticker) or ticker.startswith("^")


def _resolve_ticker(raw, exchange):
    ticker = str(raw or "").strip()
    if not ticker:
        return ""
    exchange = str(exchange or "Auto")
    if exchange == "Global":
        return ticker
    if exchange == "NSE":
        return ticker if _is_qualified(ticker) else f"{ticker}.NS"
    if exchange == "BSE":
        return ticker if _is_qualified(ticker) else f"{ticker}.BO"
    # Auto: preserve an already qualified ticker (index or suffixed), else the
    # repository's India/NSE-first convention (utils/sidebar.py, Page 5).
    return ticker if _is_qualified(ticker) else f"{ticker}.NS"


def _resolve_benchmark(label, custom=""):
    """Return (benchmark_ticker_or_None, warning_message_or_None)."""
    if label in BENCHMARK_MAP:
        return BENCHMARK_MAP[label], None
    if label == "NASDAQ":
        return None, (
            "NASDAQ benchmark ticker is not pinned in the repository docs/config; "
            "benchmark-dependent metrics are shown as N/A. Select Custom to enter "
            "a ticker explicitly."
        )
    if label == "Custom":
        ticker = str(custom or "").strip()
        if not ticker:
            return None, "Enter a custom benchmark ticker, or choose another benchmark."
        return ticker, None
    return None, None


# ── Data preparation ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_close(ticker, period=LOOKBACK_PERIOD, interval=LOOKBACK_INTERVAL):
    """Load a single Close series; empty Series on any failure."""
    if not ticker:
        return pd.Series(dtype=float)
    df = load_data(ticker, period=period, interval=interval)
    if df is None or df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)
    close = df["Close"].dropna()
    if close.empty:
        return pd.Series(dtype=float)
    close.index = pd.to_datetime(close.index)
    return close


def _prepare_returns(close):
    """Clean simple returns from a Close series (drops NaN/inf)."""
    if close is None or len(close.dropna()) < 2:
        return pd.Series(dtype=float)
    returns = compute_returns(close)
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    return returns


def _align_series(primary, secondary):
    """Inner-join two series on common valid dates (DATA_LAYER convention)."""
    if primary is None or secondary is None:
        return None, None
    joined = pd.concat(
        [primary.rename("primary"), secondary.rename("secondary")],
        axis=1,
        join="inner",
    ).dropna()
    if len(joined) < 2:
        return None, None
    return joined["primary"], joined["secondary"]


# ── Drawdown periods (no shared API exists; local helper) ────────────────────
def _drawdown_periods(equity, top_n=10):
    """Identify peak→trough→recovery drawdown periods, worst first.

    A period opens when equity falls below the running peak and closes when
    equity regains that peak. An unrecovered period has recovery=None.
    """
    series = pd.Series(equity).dropna()
    if len(series) < 2:
        return []

    periods = []
    peak_value = float(series.iloc[0])
    peak_date = series.index[0]
    trough_value = peak_value
    trough_date = peak_date
    in_drawdown = False

    for date, value in series.items():
        current = float(value)
        if current >= peak_value:
            if in_drawdown:
                periods.append({
                    "start": peak_date,
                    "end": trough_date,
                    "recovery": date,
                    "depth": trough_value / peak_value - 1.0,
                })
                in_drawdown = False
            peak_value = current
            peak_date = date
            trough_value = current
            trough_date = date
        else:
            if not in_drawdown:
                in_drawdown = True
                trough_value = current
                trough_date = date
            elif current < trough_value:
                trough_value = current
                trough_date = date

    if in_drawdown:
        periods.append({
            "start": peak_date,
            "end": trough_date,
            "recovery": None,
            "depth": trough_value / peak_value - 1.0,
        })

    periods.sort(key=lambda p: p["depth"])
    return periods[:top_n]


def _drawdown_periods_frame(periods):
    columns = ["Start", "End", "Recovery", "Drawdown %", "Duration (days)"]
    rows = [
        {
            "Start": _fmt_date(p["start"]),
            "End": _fmt_date(p["end"]),
            "Recovery": _fmt_date(p["recovery"]),
            "Drawdown %": f"{p['depth'] * 100:.2f}%",
            "Duration (days)": _duration_days(p["start"], p["recovery"]),
        }
        for p in periods
    ]
    return pd.DataFrame(rows, columns=columns)


# ── B5 procyclicality: current window vs stressed (full-sample) ──────────────
def _stressed_var_measure(returns, confidence, window):
    """Current-window vs full-sample Historical VaR / CVaR."""
    series = pd.Series(returns).dropna()
    if len(series) < 2:
        return None

    recent = series.iloc[-window:] if len(series) > window else series

    def _pair(subset):
        var = _safe_dict(value_at_risk, subset, confidence)
        cvar = _safe_dict(conditional_var, subset, confidence)
        return (
            _fmt_pct(var["historical"]) if var else "N/A",
            _fmt_pct(cvar["cvar"]) if cvar else "N/A",
            int(len(subset)),
        )

    current = _pair(recent)
    stressed = _pair(series)
    return {
        "n_current": current[2],
        "n_full": stressed[2],
        "rows": [
            {"Measure": f"Historical VaR ({confidence:.0%})", "current": current[0], "full": stressed[0]},
            {"Measure": f"CVaR ({confidence:.0%})", "current": current[1], "full": stressed[1]},
        ],
    }


# ── Risk metrics table ───────────────────────────────────────────────────────
def _risk_metric_rows(returns, equity, aligned_asset, aligned_benchmark, confidence):
    rows = [
        {"Metric": "Sharpe Ratio (annualized)", "Value": _fmt_ratio(_safe_metric(sharpe_ratio, returns))},
        {"Metric": "Sortino Ratio", "Value": _fmt_ratio(_safe_metric(sortino_ratio, returns))},
        {"Metric": "Calmar Ratio", "Value": _fmt_ratio(_safe_metric(calmar_ratio, equity))},
    ]

    if aligned_benchmark is not None:
        rows.extend([
            {"Metric": "Information Ratio", "Value": _fmt_ratio(_safe_metric(information_ratio, aligned_asset, aligned_benchmark))},
            {"Metric": "Treynor Ratio", "Value": _fmt_ratio(_safe_metric(treynor_ratio, aligned_asset, aligned_benchmark, RISK_FREE_RATE))},
            {"Metric": "Beta", "Value": _fmt_num(_safe_metric(beta, aligned_asset, aligned_benchmark))},
            {"Metric": "Alpha (Jensen)", "Value": _fmt_num(_safe_metric(alpha, aligned_asset, aligned_benchmark, RISK_FREE_RATE))},
        ])
    else:
        rows.extend([
            {"Metric": "Information Ratio", "Value": "N/A"},
            {"Metric": "Treynor Ratio", "Value": "N/A"},
            {"Metric": "Beta", "Value": "N/A"},
            {"Metric": "Alpha (Jensen)", "Value": "N/A"},
        ])

    var = _safe_dict(value_at_risk, returns, confidence)
    cvar = _safe_dict(conditional_var, returns, confidence)
    tail = _safe_dict(tail_risk, returns)

    rows.extend([
        {"Metric": "Historical VaR", "Value": _fmt_pct(var["historical"]) if var else "N/A"},
        {"Metric": "Parametric VaR", "Value": _fmt_pct(var["parametric"]) if var else "N/A"},
        {"Metric": "CVaR (Expected Shortfall)", "Value": _fmt_pct(cvar["cvar"]) if cvar else "N/A"},
        {"Metric": "Max Drawdown", "Value": _fmt_pct(_safe_metric(max_drawdown, equity))},
        {"Metric": "Tail Ratio", "Value": _fmt_ratio(tail["tail_ratio"]) if tail else "N/A"},
    ])
    return rows


# ── Figure style / builders ──────────────────────────────────────────────────
def _style(fig, height=360):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _build_underwater_figure(drawdowns):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdowns.index,
        y=drawdowns.values,
        mode="lines",
        name="Drawdown",
        line=dict(color="#F43F5E", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(244,63,94,0.20)",
    ))
    fig.update_yaxes(title="Drawdown", tickformat=".1%")
    fig.update_xaxes(title="Date")
    return _style(fig, 340)


def _build_rolling_sharpe_figure(series, window):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode="lines",
        name=f"Rolling Sharpe ({window})",
        line=dict(color="#38BDF8", width=1.6),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.35)")
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="Sharpe")
    return _style(fig, 340)


def _build_rolling_beta_figure(series, window):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode="lines",
        name=f"Rolling Beta ({window})",
        line=dict(color="#00E676", width=1.6),
    ))
    fig.add_hline(y=1.0, line_dash="dot", line_color="rgba(255,255,255,0.35)")
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="Beta")
    return _style(fig, 340)


def _build_var_distribution_figure(returns, var, cvar, confidence):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns.values,
        nbinsx=60,
        name="Returns",
        histnorm="probability density",
        marker_color="rgba(56,189,248,0.45)",
    ))
    if var:
        fig.add_vline(
            x=var["historical"], line_color="#F59E0B", line_dash="dash",
            annotation_text=f"Hist VaR {confidence:.0%}", annotation_position="top",
        )
        fig.add_vline(
            x=var["parametric"], line_color="#A78BFA", line_dash="dot",
            annotation_text="Param VaR", annotation_position="bottom",
        )
    if cvar:
        fig.add_vline(
            x=cvar["cvar"], line_color="#F43F5E", line_width=2,
            annotation_text="CVaR", annotation_position="top",
        )
    fig.update_xaxes(title="Return", tickformat=".1%")
    fig.update_yaxes(title="Density")
    return _style(fig, 400)


# ── Page body ────────────────────────────────────────────────────────────────
def render_page():
    st.set_page_config(
        page_title="Risk Analytics - QuantTerminal",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_theme()

    st.title("🛡️ Risk Analytics")
    st.caption("Comprehensive risk measurement: drawdowns, tail risk and rolling exposure.")
    st.info("This is a historical measurement, not a forecast or a recommendation.")

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.header("📊 Data Source")
    ticker_raw = st.sidebar.text_input("Ticker", value="RELIANCE.NS")
    exchange = st.sidebar.selectbox("Exchange", EXCHANGE_OPTIONS)

    benchmark_label = st.sidebar.selectbox("Benchmark", BENCHMARK_OPTIONS)
    custom_benchmark = ""
    if benchmark_label == "Custom":
        custom_benchmark = st.sidebar.text_input("Custom Benchmark Ticker", value="")

    st.sidebar.divider()
    st.sidebar.subheader("🛡️ Risk Settings")
    confidence = st.sidebar.slider(
        "Confidence Level",
        min_value=CONFIDENCE_MIN,
        max_value=CONFIDENCE_MAX,
        value=CONFIDENCE_DEFAULT,
        step=0.01,
        format="%.2f",
    )
    window = st.sidebar.slider(
        "Rolling Window",
        min_value=WINDOW_MIN,
        max_value=WINDOW_MAX,
        value=WINDOW_DEFAULT,
        step=1,
    )

    # ── Resolve inputs ───────────────────────────────────────────────────────
    ticker = _resolve_ticker(ticker_raw, exchange)
    benchmark_ticker, benchmark_warning = _resolve_benchmark(benchmark_label, custom_benchmark)

    if not ticker_raw or not ticker_raw.strip():
        st.warning("Enter a ticker in the sidebar to run the risk analysis.")
        st.stop()

    # ── Load asset data ──────────────────────────────────────────────────────
    close = _load_close(ticker)
    if close.empty:
        st.warning(f"No price data available for **{ticker}**. Check the ticker/exchange.")
        st.stop()

    returns = _prepare_returns(close)
    if len(returns) < 2:
        st.warning(f"Not enough return observations for **{ticker}** to compute risk metrics.")
        st.stop()

    equity = close.dropna()
    if len(equity) < 2:
        st.warning(f"Not enough observations to build an equity curve for **{ticker}**.")
        st.stop()

    st.sidebar.success(f"Loaded {len(close):,} rows for **{ticker}**")

    if benchmark_warning:
        st.sidebar.warning(benchmark_warning)

    # ── Load & align benchmark data ──────────────────────────────────────────
    benchmark_returns = None
    benchmark_aligned = None
    if benchmark_ticker:
        bench_close = _load_close(benchmark_ticker)
        if bench_close.empty:
            st.sidebar.warning(
                f"Benchmark data for **{benchmark_ticker}** could not be loaded; "
                "benchmark-dependent metrics are N/A."
            )
        else:
            bench_returns_raw = _prepare_returns(bench_close)
            benchmark_aligned, benchmark_returns = _align_series(returns, bench_returns_raw)
            if benchmark_aligned is None:
                st.sidebar.warning(
                    "Benchmark and asset returns do not overlap sufficiently; "
                    "benchmark-dependent metrics are N/A."
                )

    # ── Risk Metrics table ───────────────────────────────────────────────────
    st.subheader("Risk Metrics")
    metric_rows = _risk_metric_rows(
        returns,
        equity,
        benchmark_aligned,
        benchmark_returns,
        confidence,
    )
    st.dataframe(
        pd.DataFrame(metric_rows, columns=["Metric", "Value"]),
        use_container_width=True,
        hide_index=True,
    )
    if benchmark_aligned is None:
        st.caption(
            "Benchmark-dependent rows (Information Ratio, Treynor, Beta, Alpha) "
            "are N/A because no usable benchmark is selected."
        )
    st.caption(
        "VaR/CVaR are historical/empirical tail estimates. Per B5, Historical VaR, "
        "Parametric VaR and CVaR are reported together with CVaR as the primary "
        "tail measure."
    )

    # ── B5 procyclicality: current vs stressed ───────────────────────────────
    st.subheader("Tail Risk — Current vs Stressed")
    stressed = _stressed_var_measure(returns, confidence, window)
    if stressed is None:
        st.info("Not enough observations for a current-vs-stressed comparison.")
    else:
        frame = pd.DataFrame(stressed["rows"]).rename(columns={
            "current": f"Current window (n={stressed['n_current']})",
            "full": f"Full sample (n={stressed['n_full']})",
        })
        st.dataframe(frame, use_container_width=True, hide_index=True)
        st.caption(
            "Procyclicality (B5): the current window can understate risk in calm "
            "periods and overstate it just after a shock; the full-sample column is "
            "a stressed/long-horizon reference. B5 also calls for Kendall τ and "
            "conditional crisis correlation, but no shared API or Page-7 output slot "
            "exists on this branch."
        )

    # ── Drawdown Periods table ───────────────────────────────────────────────
    st.subheader("Drawdown Periods — Top 10")
    dd_series = drawdown_series(equity)
    periods = _drawdown_periods(equity, top_n=10)
    if not periods:
        st.info("No drawdown periods detected (equity is monotonic).")
    else:
        st.dataframe(
            _drawdown_periods_frame(periods),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "A period starts at the prior peak, ends at the trough, and recovers when "
            "the peak is regained. 'N/A' recovery marks an ongoing drawdown."
        )

    # ── Charts ───────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Charts")

    try:
        st.plotly_chart(_build_underwater_figure(dd_series), use_container_width=True)
    except (ValueError, TypeError) as exc:
        st.warning(f"Drawdown chart unavailable: {exc}")

    if len(returns) < window:
        st.info(
            f"Rolling window ({window}) exceeds the {len(returns)} available return "
            "observations; rolling charts need more data."
        )
    else:
        try:
            roll_sharpe = rolling_sharpe(returns, window=window, risk_free_rate=RISK_FREE_RATE)
            st.plotly_chart(
                _build_rolling_sharpe_figure(roll_sharpe.dropna(), window),
                use_container_width=True,
            )
        except (ValueError, TypeError) as exc:
            st.warning(f"Rolling Sharpe unavailable: {exc}")

        if benchmark_aligned is None or benchmark_returns is None:
            st.info(
                "Rolling beta requires a benchmark with overlapping dates; "
                "select a benchmark to view this chart."
            )
        else:
            try:
                roll_beta = rolling_beta(benchmark_aligned, benchmark_returns, window=window)
                st.plotly_chart(
                    _build_rolling_beta_figure(roll_beta.dropna(), window),
                    use_container_width=True,
                )
            except (ValueError, TypeError) as exc:
                st.warning(f"Rolling beta unavailable: {exc}")

    var = _safe_dict(value_at_risk, returns, confidence)
    cvar = _safe_dict(conditional_var, returns, confidence)
    try:
        st.plotly_chart(
            _build_var_distribution_figure(returns, var, cvar, confidence),
            use_container_width=True,
        )
    except (ValueError, TypeError) as exc:
        st.warning(f"VaR distribution chart unavailable: {exc}")

    # ── Bias / caveat footer ─────────────────────────────────────────────────
    st.divider()
    st.caption(
        "**Bias controls (B5/B19).** Window disclosure (B8): rolling measures use a "
        f"{window}-day window; risk-free rate is {RISK_FREE_RATE:.1%} (RISK_ANALYTICS §6 "
        "default; not exposed as a control). Low-frequency Close prices (B19): the "
        "bid/ask-vs-close caveat applies, microstructure noise is not modelled and no "
        "explicit transaction costs/friction are applied. Rare extreme events (B5): "
        "the historical sample is not robust to unseen tail events."
    )


if __name__ == "__main__":
    render_page()
