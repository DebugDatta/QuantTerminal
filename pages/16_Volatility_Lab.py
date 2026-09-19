import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils.helper import (
    inject_custom_theme,
    load_data,
    drop_holiday_nans,
    fetch_stocks,
)
from utils.sidebar import render_sidebar
from core.returns import compute_returns
from volatility.estimators import historical_vol, ewma_vol, parkinson, gk, rs, yz
from volatility.garch import fit_garch, fit_egarch, fit_gjr_garch, DEFAULT_HORIZON
from risk.rolling import rolling_vol

EXCHANGE_OPTIONS = ("Auto", "NSE", "BSE", "Global")
GARCH_MODELS = ("GARCH", "EGARCH", "GJR-GARCH")
_WINDOW_MIN, _WINDOW_MAX, _WINDOW_DEFAULT = 5, 252, 20
_LAM_MIN, _LAM_MAX, _LAM_DEFAULT = 0.85, 0.99, 0.94
_PQ_MIN, _PQ_MAX, _PQ_DEFAULT = 1, 5, 1

_ESTIMATOR_SPECS = [
    ("Historical", historical_vol, ["Close"]),
    ("EWMA", ewma_vol, ["Close"]),
    ("Parkinson", parkinson, ["High", "Low"]),
    ("Garman-Klass", gk, ["Open", "High", "Low", "Close"]),
    ("Rogers-Satchell", rs, ["Open", "High", "Low", "Close"]),
    ("Yang-Zhang", yz, ["Open", "High", "Low", "Close"]),
]
_ESTIMATOR_NAMES = tuple(name for name, _, _ in _ESTIMATOR_SPECS)

_GARCH_FNS = {
    "GARCH": fit_garch,
    "EGARCH": fit_egarch,
    "GJR-GARCH": fit_gjr_garch,
}


def _fmt(value) -> str:
    if value is None:
        return "N/A"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(value):
        return "N/A"
    return f"{value:.4f}"


def _fmt_pct(value) -> str:
    if value is None:
        return "N/A"
    try:
        value = float(value) * 100.0
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(value):
        return "N/A"
    return f"{value:.2f}%"


def _resolve_ticker(symbol: str, exchange: str) -> str:
    """Return the ticker string resolved per docs/DATA_LAYER.md.

    Explicit .NS/.BO suffixes are honored; otherwise Auto probes
    `<sym>.NS` -> raw -> `<sym>.BO` and picks the first provider that
    returns data, falling back to the user-provided symbol.
    """
    symbol = str(symbol).strip()
    if not symbol:
        return symbol
    if symbol.lower().endswith((".ns", ".bo")):
        return symbol
    if exchange == "NSE":
        return f"{symbol}.NS"
    if exchange == "BSE":
        return f"{symbol}.BO"
    if exchange == "Global":
        return symbol
    for cand in (f"{symbol}.NS", symbol, f"{symbol}.BO"):
        df = load_data(cand, period="2y", interval="1d")
        if df is not None and not df.empty and "Close" in df.columns:
            return cand
    return symbol


def _load_ohlcv(symbol: str, exchange: str) -> pd.DataFrame:
    """Load and clean an OHLCV frame.

    Returns an empty frame when the ticker cannot be resolved, when the
    provider returns no rows, or when the required capitalised
    Open/High/Low/Close columns are missing. No OHLC values are fabricated.
    """
    ticker = _resolve_ticker(symbol, exchange)
    df = load_data(ticker, period="2y", interval="1d")
    if df is None or df.empty:
        return pd.DataFrame()
    df = drop_holiday_nans(df)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except (TypeError, ValueError, OverflowError):
            pass
    missing = [c for c in ("Open", "High", "Low", "Close") if c not in df.columns]
    if missing:
        return pd.DataFrame()
    return df


def _returns(close: pd.Series) -> pd.Series:
    """Simple percentage returns from Close (core.returns contract)."""
    return compute_returns(close).dropna().astype(float)


def _estimate_stats(series: pd.Series) -> dict:
    """Latest / mean / std / min / max / valid-count of a series."""
    vals = series.dropna()
    if vals.empty:
        return {"latest": None, "mean": None, "std": None, "min": None, "max": None, "count": 0}
    n = int(vals.size)
    latest = float(vals.iloc[-1])
    return {
        "latest": latest,
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)) if n > 1 else 0.0,
        "min": float(vals.min()),
        "max": float(vals.max()),
        "count": n,
    }


def _rolling_estimator(frame: pd.DataFrame, fn, window: int, lam=None) -> pd.Series:
    """Trailing-window rolling series of a point volatility estimator.

    Each step feeds the estimator the trailing `window + 1` rows ending at
    t (the shortest slice every estimator contract requires internally), so
    the rolling surface applies one estimator to the same trailing windows.
    Runs locally through the point-estimator APIs (risk/rolling.rolling_vol
    is unavailable below window 20) — see R57/R61.
    """
    n = len(frame)
    if n < window + 1:
        return pd.Series(dtype=float)
    result = []
    for i in range(window, n):
        chunk = frame.iloc[i - window : i + 1]
        try:
            kwargs = {"window": window}
            if lam is not None:
                kwargs["lam"] = lam
            result.append((frame.index[i], float(fn(chunk, **kwargs))))
        except Exception:
            continue
    if not result:
        return pd.Series(dtype=float)
    series = pd.Series(
        [v for _, v in result], index=pd.Index([t for t, _ in result]), dtype=float
    )
    return series.dropna()


def _rolling_series_for_all(frame: pd.DataFrame, window: int, lam=None) -> dict:
    """Rolling series for every estimator, isolated per estimator."""
    out = {}
    for name, fn, needed in _ESTIMATOR_SPECS:
        if not all(c in frame.columns for c in needed):
            out[name] = pd.Series(dtype=float)
            continue
        try:
            out[name] = _rolling_estimator(
                frame, fn, window, lam=lam if name == "EWMA" else None
            )
        except Exception:
            out[name] = pd.Series(dtype=float)
    return out


def _garch_badge(out: dict, p: int, q: int) -> dict:
    """GARCH confidence badge per docs/MODEL_CONFIDENCE.md lines 19-28.

    Worst-of across observations, convergence, Ljung-Box p and AIC/BIC
    sanity (conservative). MIN_OBSERVATIONS floor value is undocumented
    (R37); the documented band thresholds 250x / 125x are used instead.
    """
    n = int(out["n"])
    lb_p = float(out["ljung_box"]["p_value"])
    aic = float(out["aic"])
    bic = float(out["bic"])
    converged = bool(out.get("converged", True))

    ratio = n / float(p + q)
    obs_level = 2 if ratio >= 250 else (1 if ratio >= 125 else 0)
    lb_level = 2 if lb_p > 0.05 else (1 if lb_p > 0.01 else 0)
    aic_level = (
        2 if aic < bic else (1 if np.isclose(aic, bic, rtol=0.05, atol=1e-6) else 0)
    )
    conv_level = 2 if converged else 0
    worst = min(obs_level, lb_level, aic_level, conv_level)
    label = {2: ("high", "🟢"), 1: ("medium", "🟡"), 0: ("low", "🔴")}[worst]

    if aic < bic:
        aic_desc = "AIC < BIC"
    elif aic_level == 1:
        aic_desc = "AIC ≈ BIC"
    else:
        aic_desc = "AIC > BIC"

    summary = (
        f"n = {n} vs (p+q)={p + q}: ≥250×(p+q) high, ≥125×(p+q) medium; "
        f"Ljung-Box p = {lb_p:.4g}; {aic_desc}; "
        f"{'converged' if converged else 'failed to converge'}"
    )
    return {"level": label[0], "color": label[1], "summary": summary}


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


if __name__ == "__main__":
    st.set_page_config(
        page_title="Volatility Lab - QuantTerminal",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_theme()

    st.title("📊 Volatility Lab")
    st.caption("Annualized volatility estimation and GARCH-family conditional volatility.")

    st.sidebar.header("📊 Data Source")
    ticker_symbol = st.sidebar.text_input("Ticker", value="RELIANCE.NS")
    exchange = st.sidebar.selectbox("Exchange", EXCHANGE_OPTIONS)

    st.sidebar.divider()
    st.sidebar.subheader("🔬 Volatility Settings")
    estimator_name = st.sidebar.selectbox("Estimator", list(_ESTIMATOR_NAMES))
    window = st.sidebar.slider(
        "Window",
        min_value=_WINDOW_MIN,
        max_value=_WINDOW_MAX,
        value=_WINDOW_DEFAULT,
    )
    lambda_ewma = None
    if estimator_name == "EWMA":
        lambda_ewma = st.sidebar.slider(
            "EWMA Lambda (λ)",
            min_value=_LAM_MIN,
            max_value=_LAM_MAX,
            value=_LAM_DEFAULT,
            step=0.01,
        )

    st.sidebar.divider()
    st.sidebar.subheader("📈 GARCH Settings")
    garch_model = st.sidebar.selectbox("GARCH Model", GARCH_MODELS)
    garch_p = st.sidebar.slider(
        "p", min_value=_PQ_MIN, max_value=_PQ_MAX, value=_PQ_DEFAULT
    )
    garch_q = st.sidebar.slider(
        "q", min_value=_PQ_MIN, max_value=_PQ_MAX, value=_PQ_DEFAULT
    )

    frame = _load_ohlcv(ticker_symbol, exchange)
    if frame.empty:
        st.warning("No OHLCV data could be loaded. Check the ticker / exchange and try again.")
        st.stop()
    st.sidebar.success(f"Loaded {len(frame):,} rows")

    close = frame["Close"].astype(float)
    returns = _returns(close)

    rolling = _rolling_series_for_all(frame, window, lambda_ewma)
    stats = {name: _estimate_stats(rolling[name]) for name, _, _ in _ESTIMATOR_SPECS}

    # ── T1: Volatility Estimates ────────────────────────────────────────────
    st.subheader("Volatility Estimates")
    rows = []
    for name, _, needed in _ESTIMATOR_SPECS:
        has_ohlc = all(c in frame.columns for c in needed)
        s = stats[name]
        if s["count"] == 0:
            cause = "N/A (insufficient OHLC)" if not has_ohlc else "N/A"
            rows.append(
                {"Estimator": name, "Latest": cause, "Mean": cause, "Std": cause,
                 "Min": cause, "Max": cause, "Obs": 0}
            )
        else:
            rows.append(
                {"Estimator": name, "Latest": _fmt_pct(s["latest"]),
                 "Mean": _fmt_pct(s["mean"]), "Std": _fmt_pct(s["std"]),
                 "Min": _fmt_pct(s["min"]), "Max": _fmt_pct(s["max"]),
                 "Obs": s["count"]}
            )
    st.dataframe(
        pd.DataFrame(rows, columns=["Estimator", "Latest", "Mean", "Std", "Min", "Max", "Obs"]),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Annualized volatility (×√252) evaluated at the trailing window; values shown as %."
    )

    # ── C1: Rolling volatility overlay ──────────────────────────────────────
    st.subheader("Rolling Volatility — All Estimators")
    fig = go.Figure()
    for name, _, _ in _ESTIMATOR_SPECS:
        s = rolling.get(name)
        if s is not None and len(s) > 0:
            fig.add_trace(
                go.Scatter(x=s.index, y=s.values * 100.0, mode="lines", name=name)
            )
    if not fig.data:
        st.caption("No rolling volatility series available at the selected window size.")
    else:
        fig.update_layout(title="Rolling annualized volatility (%)")
        st.plotly_chart(_style(fig, height=400), use_container_width=True)

    # ── GARCH section ───────────────────────────────────────────────────────
    st.subheader("GARCH — Conditional Volatility")
    garch_out = None
    garch_error = None
    needed_obs = 2 * (garch_p + garch_q + 1)
    if returns.empty or returns.shape[0] < needed_obs:
        garch_error = (
            f"Insufficient observations ({returns.shape[0]}); "
            f"{garch_model}({garch_p},{garch_q}) requires ≥ {needed_obs}."
        )
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                garch_out = _GARCH_FNS[garch_model](
                    returns,
                    p=garch_p,
                    q=garch_q,
                    distribution="normal",
                    horizon=DEFAULT_HORIZON,
                )
        except Exception as exc:
            garch_error = f"{garch_model} fit failed: {exc}"

    if garch_out is None:
        st.info(garch_error)
        st.caption(
            "GARCH output unavailable; coefficients, diagnostics and "
            "conditional-volatility / forecast charts are skipped."
        )
    else:
        badge = _garch_badge(garch_out, garch_p, garch_q)

        # ── T2: GARCH Coefficients + badge ────────────────────────────────
        coefs = garch_out["coefficients"]
        coef_rows = []
        if "omega" in coefs:
            coef_rows.append(["omega (ω)", _fmt(coefs["omega"])])
        if "mu" in coefs:
            coef_rows.append(["mu (μ)", _fmt(coefs["mu"])])
        for i, v in enumerate(coefs.get("alpha") or []):
            coef_rows.append([f"alpha_{i + 1} (α)", _fmt(v)])
        for i, v in enumerate(coefs.get("gamma") or []):
            coef_rows.append([f"gamma_{i + 1} (γ)", _fmt(v)])
        for i, v in enumerate(coefs.get("beta") or []):
            coef_rows.append([f"beta_{i + 1} (β)", _fmt(v)])

        st.markdown(f"**GARCH Coefficients — {garch_out['model']}**")
        st.markdown(f"Confidence badge: {badge['color']} **{badge['level']}**")
        st.caption(badge["summary"])
        st.dataframe(
            pd.DataFrame(coef_rows, columns=["Parameter", "Value"]),
            use_container_width=True,
            hide_index=True,
        )

        # ── T3: GARCH Diagnostics ──────────────────────────────────────────
        lb = garch_out["ljung_box"]
        diag_df = pd.DataFrame(
            [
                {
                    "Model": garch_out["model"],
                    "p": garch_out["p"],
                    "q": garch_out["q"],
                    "AIC": round(garch_out["aic"], 3),
                    "BIC": round(garch_out["bic"], 3),
                    "Ljung-Box stat": round(lb["statistic"], 4),
                    "Ljung-Box p": round(lb["p_value"], 4),
                    "lags": lb["lags"],
                    "Converged": "Yes" if garch_out["converged"] else "No",
                }
            ]
        )
        st.subheader("GARCH Diagnostics")
        st.dataframe(diag_df, use_container_width=True, hide_index=True)
        st.caption(
            "Ljung-Box applied to squared standardized residuals (remaining ARCH "
            "effects; MODEL_CONFIDENCE.md). AIC/BIC and convergence feed the badge."
        )

        # ── C2: Conditional volatility ────────────────────────────────────
        cv = np.asarray(garch_out["conditional_volatility"], dtype=float)
        idx = returns.index[: cv.shape[0]]
        fig = go.Figure(
            go.Scatter(x=idx, y=cv, mode="lines", name=f"{garch_out['model']} σ")
        )
        fig.update_layout(title="GARCH conditional volatility")
        st.plotly_chart(_style(fig, height=360), use_container_width=True)
        st.caption(
            "Conditional volatility is daily (not annualized) per volatility/garch.py contract."
        )

        # ── C3: Volatility forecast ────────────────────────────────────────
        fc = np.asarray(garch_out["forecast"]["volatility"], dtype=float)
        fig = go.Figure(
            go.Scatter(
                x=list(range(1, fc.shape[0] + 1)),
                y=fc,
                mode="lines+markers",
                name="forecast",
            )
        )
        fig.update_layout(title=f"GARCH volatility forecast — next {fc.shape[0]} steps (daily)")
        fig.update_xaxes(title_text="Step")
        st.plotly_chart(_style(fig, height=340), use_container_width=True)

    # ── C4: Estimator comparison ────────────────────────────────────────────
    st.subheader("Estimator Comparison")
    names, values = [], []
    for name, _, _ in _ESTIMATOR_SPECS:
        s = stats[name]
        if s["latest"] is not None:
            names.append(name)
            values.append(s["latest"] * 100.0)
    if not values:
        st.caption("No estimates available for comparison.")
    else:
        fig = go.Figure(
            go.Bar(
                x=names,
                y=values,
                text=[f"{v:.2f}%" for v in values],
                textposition="outside",
                marker_color="#00cc96",
            )
        )
        fig.update_layout(
            title="Latest annualized volatility by estimator (%)",
            yaxis_title="Annualized vol (%)",
        )
        st.plotly_chart(_style(fig, height=380), use_container_width=True)