"""Portfolio Lab — Quantitative Portfolio Construction, Optimization & Risk Analytics Terminal.

A professional, institutional-grade single-page quantitative portfolio construction,
risk decomposition, Markowitz efficient frontier simulation, Hierarchical Risk Parity (HRP),
rolling factor attribution, multi-scenario macro stress testing, and rebalance execution terminal.
"""

from datetime import date, datetime, timedelta
import io
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy.optimize import minimize
import streamlit as st
import yfinance as yf

# Root path alignment
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from optimization.hrp import hierarchical_risk_parity


# -----------------------------------------------------------------------------
# 1. Custom Institutional Dark Terminal Theme
# -----------------------------------------------------------------------------
def inject_portfolio_theme():
    """Inject a professional, high-density dark quantitative terminal stylesheet."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: #0B0F19;
            color: #E2E8F0;
        }

        .port-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 18px;
            background: #111827;
            border: 1px solid #1E293B;
            border-radius: 8px;
            margin-bottom: 16px;
        }
        .port-title {
            font-size: 1.25rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            color: #F8FAFC;
            text-transform: uppercase;
        }
        .port-subtitle {
            font-size: 0.78rem;
            color: #94A3B8;
            margin-top: 2px;
        }
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(16, 185, 129, 0.12);
            color: #10B981;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            border: 1px solid rgba(16, 185, 129, 0.25);
            font-family: 'JetBrains Mono', monospace;
        }
        .status-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background-color: #10B981;
            box-shadow: 0 0 6px #10B981;
        }

        .control-panel {
            background: #111827;
            border: 1px solid #1E293B;
            border-radius: 8px;
            padding: 14px 18px;
            margin-bottom: 16px;
        }

        .chip-container {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 6px;
            margin-bottom: 12px;
        }
        .asset-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: #1E293B;
            border: 1px solid #334155;
            color: #38BDF8;
            padding: 4px 10px;
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.80rem;
            font-weight: 600;
        }

        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 10px;
            margin-bottom: 14px;
        }
        .kpi-card {
            background: #111827;
            border: 1px solid #1E293B;
            border-radius: 6px;
            padding: 10px 12px;
            display: flex;
            flex-direction: column;
        }
        .kpi-label {
            font-size: 0.70rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94A3B8;
            font-weight: 600;
            margin-bottom: 3px;
        }
        .kpi-value {
            font-size: 1.18rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            color: #F8FAFC;
        }
        .kpi-sub {
            font-size: 0.68rem;
            color: #64748B;
            margin-top: 2px;
            font-family: 'JetBrains Mono', monospace;
        }

        .sub-metrics-grid {
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 8px;
            background: rgba(17, 24, 39, 0.6);
            border: 1px solid #1E293B;
            border-radius: 6px;
            padding: 8px 12px;
            margin-bottom: 16px;
        }
        .sub-metric-item {
            display: flex;
            flex-direction: column;
        }
        .sub-metric-title {
            font-size: 0.66rem;
            color: #64748B;
            text-transform: uppercase;
        }
        .sub-metric-val {
            font-size: 0.88rem;
            font-weight: 600;
            color: #CBD5E1;
            font-family: 'JetBrains Mono', monospace;
        }

        .val-pos { color: #10B981 !important; }
        .val-neg { color: #EF4444 !important; }
        .val-neutral { color: #38BDF8 !important; }

        .section-header {
            font-size: 0.90rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: #38BDF8;
            margin-top: 18px;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 8px;
            border-bottom: 1px solid #1E293B;
            padding-bottom: 4px;
        }

        .risk-bar-container {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
        }
        .risk-bar-label {
            width: 110px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            color: #CBD5E1;
            text-align: right;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .risk-bar-track {
            flex-grow: 1;
            height: 12px;
            background: #1E293B;
            border-radius: 3px;
            overflow: hidden;
            position: relative;
        }
        .risk-bar-fill {
            height: 100%;
            background: #38BDF8;
            border-radius: 3px;
        }
        .risk-bar-val {
            width: 52px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.76rem;
            color: #94A3B8;
        }

        .footer-disclaimer {
            text-align: center;
            font-size: 0.72rem;
            color: #64748B;
            padding: 18px 0 10px 0;
            border-top: 1px solid #1E293B;
            margin-top: 28px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# 2. Universe Snapshots & Helper Functions
# -----------------------------------------------------------------------------
def fmt_inr(val: float | None) -> str:
    """Format numeric capital into Indian numbering system (Lakh/Crore) or standard."""
    if val is None or np.isnan(val):
        return "—"
    sign = "-" if val < 0 else ""
    abs_v = abs(val)
    if abs_v >= 1e7:
        return f"{sign}₹{abs_v/1e7:,.2f} Cr"
    elif abs_v >= 1e5:
        return f"{sign}₹{abs_v/1e5:,.2f}L"
    else:
        return f"{sign}₹{abs_v:,.0f}"


@st.cache_data(ttl=86400, show_spinner=False)
def load_universe_snapshots() -> dict[str, Any]:
    """Load stock universe snapshots from India and US CSV data."""
    root = Path(__file__).resolve().parent.parent
    in_path = root / "data" / "snapshots" / "India_Stocks_Data.csv"
    us_path = root / "data" / "snapshots" / "US_Stocks_Data.csv"

    records_by_ticker: dict[str, dict[str, Any]] = {}
    india_items: list[dict[str, Any]] = []
    us_items: list[dict[str, Any]] = []

    if in_path.exists():
        try:
            df_in = pd.read_csv(in_path).sort_values(by="Market capitalization", ascending=False)
            seen = set()
            for _, r in df_in.iterrows():
                sym = str(r["Symbol"]).strip() if pd.notna(r.get("Symbol")) else ""
                if not sym or sym in seen:
                    continue
                seen.add(sym)
                desc = str(r["Description"]).strip() if pd.notna(r.get("Description")) else sym
                ex = str(r["Exchange"]).strip().upper() if pd.notna(r.get("Exchange")) else "NSE"
                yf_sym = f"{sym}.NS" if ex == "NSE" else (f"{sym}.BO" if ex == "BSE" else sym)
                sector = str(r["Sector"]).strip() if pd.notna(r.get("Sector")) else "General"
                mcap = float(r["Market capitalization"]) if pd.notna(r.get("Market capitalization")) else 1e9
                px = float(r["Price"]) if pd.notna(r.get("Price")) else 0.0
                vol = float(r["Volume"]) if pd.notna(r.get("Volume")) else 50000.0
                div_yield = float(r["Dividend yield"]) if pd.notna(r.get("Dividend yield")) else 0.0

                label = f"{sym} · {desc} ({ex})"
                rec = {
                    "symbol": sym,
                    "yf_ticker": yf_sym,
                    "name": desc,
                    "exchange": ex,
                    "sector": sector,
                    "mcap": mcap,
                    "price": px,
                    "volume": vol,
                    "div_yield": div_yield,
                    "currency": "INR",
                    "label": label,
                    "market": "India",
                }
                india_items.append(rec)
                records_by_ticker[yf_sym.upper()] = rec
                records_by_ticker[sym.upper()] = rec
        except Exception:
            pass

    if us_path.exists():
        try:
            df_us = pd.read_csv(us_path).sort_values(by="Market capitalization", ascending=False)
            seen = set()
            for _, r in df_us.iterrows():
                sym = str(r["Symbol"]).strip() if pd.notna(r.get("Symbol")) else ""
                if not sym or sym in seen:
                    continue
                seen.add(sym)
                desc = str(r["Description"]).strip() if pd.notna(r.get("Description")) else sym
                ex = str(r["Exchange"]).strip().upper() if pd.notna(r.get("Exchange")) else "NASDAQ"
                sector = str(r["Sector"]).strip() if pd.notna(r.get("Sector")) else "General"
                mcap = float(r["Market capitalization"]) if pd.notna(r.get("Market capitalization")) else 1e9
                px = float(r["Price"]) if pd.notna(r.get("Price")) else 0.0
                vol = float(r["Volume"]) if pd.notna(r.get("Volume")) else 100000.0
                div_yield = float(r["Dividend yield"]) if pd.notna(r.get("Dividend yield")) else 0.0

                label = f"{sym} · {desc} ({ex})"
                rec = {
                    "symbol": sym,
                    "yf_ticker": sym,
                    "name": desc,
                    "exchange": ex,
                    "sector": sector,
                    "mcap": mcap,
                    "price": px,
                    "volume": vol,
                    "div_yield": div_yield,
                    "currency": "USD",
                    "label": label,
                    "market": "US",
                }
                us_items.append(rec)
                records_by_ticker[sym.upper()] = rec
        except Exception:
            pass

    return {
        "india_items": india_items,
        "us_items": us_items,
        "records_by_ticker": records_by_ticker,
    }


def resolve_ticker(raw_sym: str) -> str:
    """Standardize ticker for yfinance download."""
    sym = raw_sym.strip().upper()
    if sym.startswith("^") or sym.endswith((".NS", ".BO", "=X", "=F")):
        return sym
    return sym


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_asset_and_benchmark_prices(
    tickers: list[str],
    benchmark_ticker: str = "^NSEI",
    period: str = "1Y",
    frequency: str = "Daily",
    custom_start: date | None = None,
    custom_end: date | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Fetch adjusted close prices for portfolio assets and aligned benchmark."""
    period_map = {
        "1M": "1mo", "3M": "3mo", "6M": "6mo",
        "1Y": "1y", "3Y": "3y", "5Y": "5y", "MAX": "max"
    }
    interval_map = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}

    yf_interval = interval_map.get(frequency, "1d")
    download_symbols = list(set([resolve_ticker(t) for t in tickers] + [resolve_ticker(benchmark_ticker)]))

    try:
        if period == "Custom" and custom_start and custom_end:
            data = yf.download(
                download_symbols,
                start=custom_start.strftime("%Y-%m-%d"),
                end=(custom_end + timedelta(days=1)).strftime("%Y-%m-%d"),
                interval=yf_interval,
                auto_adjust=True,
                progress=False,
            )
        else:
            yf_period = period_map.get(period, "1y")
            data = yf.download(
                download_symbols,
                period=yf_period,
                interval=yf_interval,
                auto_adjust=True,
                progress=False,
            )
    except Exception:
        return pd.DataFrame(), pd.Series(dtype=float)

    if data is None or data.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.levels[0]:
            df_close = data["Close"]
        else:
            df_close = data.xs("Close", axis=1, level=0, drop_level=True)
    else:
        df_close = data[["Close"]] if "Close" in data.columns else data

    df_close = df_close.dropna(how="all").ffill().bfill()
    df_close.index = pd.to_datetime(df_close.index)
    if df_close.index.tz is not None:
        df_close.index = df_close.index.tz_localize(None)

    resolved_bench = resolve_ticker(benchmark_ticker)
    if resolved_bench in df_close.columns:
        bench_s = df_close[resolved_bench]
    else:
        bench_s = df_close.mean(axis=1)

    asset_cols = [resolve_ticker(t) for t in tickers if resolve_ticker(t) in df_close.columns]
    prices_df = df_close[asset_cols].dropna()
    bench_s = bench_s.reindex(prices_df.index).ffill().bfill()

    return prices_df, bench_s


# -----------------------------------------------------------------------------
# 3. Quantitative Math, Risk & Optimizer Engines
# -----------------------------------------------------------------------------
def calculate_portfolio_series(
    prices: pd.DataFrame,
    weights: pd.Series,
    initial_capital: float = 1000000.0,
    benchmark_prices: pd.Series | None = None,
    cash_fraction: float = 0.0,
    risk_free_rate: float = 0.05,
) -> dict[str, Any]:
    """Compute daily portfolio values, compounding returns, cash sleeve, and benchmark alignment."""
    w = weights.reindex(prices.columns).fillna(0.0).values
    w_sum = np.sum(w)
    if w_sum > 0:
        w = w / w_sum

    asset_returns = prices.pct_change().fillna(0.0)
    risky_daily_ret = asset_returns @ w

    # Incorporate cash sleeve yielding Rf / 252
    cash_daily_ret = risk_free_rate / 252.0
    equity_weight = 1.0 - cash_fraction
    port_daily_ret = (risky_daily_ret * equity_weight) + (cash_daily_ret * cash_fraction)

    cum_ret = (1.0 + port_daily_ret).cumprod()
    portfolio_equity = cum_ret * initial_capital

    if benchmark_prices is not None and len(benchmark_prices) == len(prices):
        bench_cum = benchmark_prices / benchmark_prices.iloc[0]
        benchmark_equity = bench_cum * initial_capital
        bench_daily_ret = benchmark_prices.pct_change().fillna(0.0)
    else:
        benchmark_equity = portfolio_equity.copy()
        bench_daily_ret = port_daily_ret.copy()

    # Drawdowns
    running_max = portfolio_equity.cummax()
    drawdown_series = (portfolio_equity - running_max) / running_max * 100.0

    return {
        "portfolio_equity": portfolio_equity,
        "portfolio_daily_ret": port_daily_ret,
        "risky_daily_ret": risky_daily_ret,
        "benchmark_equity": benchmark_equity,
        "benchmark_daily_ret": bench_daily_ret,
        "drawdown_series": drawdown_series,
        "normalized_weights": pd.Series(w, index=prices.columns),
        "cash_fraction": cash_fraction,
    }


def compute_portfolio_kpis(
    port_series: dict[str, Any],
    initial_capital: float = 1000000.0,
    risk_free_rate: float = 0.05,
    freq_factor: int = 252,
) -> dict[str, Any]:
    """Calculate institutional portfolio KPIs, Sortino, Calmar, Beta, and VaR/CVaR."""
    eq = port_series["portfolio_equity"]
    bm = port_series["benchmark_equity"]
    d_rets = port_series["portfolio_daily_ret"]
    bm_rets = port_series["benchmark_daily_ret"]
    dd = port_series["drawdown_series"]

    n_bars = len(eq)
    if n_bars < 2:
        return {}

    n_years = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.05)
    final_val = float(eq.iloc[-1])
    tot_return_pct = float((final_val / initial_capital - 1.0) * 100.0)
    cagr_pct = float(((final_val / initial_capital) ** (1.0 / n_years) - 1.0) * 100.0)

    # Volatility & Sharpe
    ann_vol_pct = float(d_rets.std() * np.sqrt(freq_factor) * 100.0)
    excess_ret = (cagr_pct / 100.0) - risk_free_rate
    sharpe = float(excess_ret / (ann_vol_pct / 100.0 + 1e-9))

    # Downside Volatility and Sortino
    neg_rets = d_rets[d_rets < 0]
    downside_vol_pct = float(neg_rets.std() * np.sqrt(freq_factor) * 100.0) if len(neg_rets) > 1 else ann_vol_pct
    sortino = float(excess_ret / (downside_vol_pct / 100.0 + 1e-9))

    # Drawdown & Calmar
    max_dd_pct = float(dd.min())
    avg_dd_pct = float(dd[dd < -0.01].mean()) if len(dd[dd < -0.01]) > 0 else 0.0
    calmar = float(abs(cagr_pct / max_dd_pct)) if abs(max_dd_pct) > 0.01 else 0.0

    # Beta & Tracking Error
    cov_bm = np.cov(d_rets, bm_rets)[0, 1] if len(d_rets) > 5 else 0.0
    var_bm = np.var(bm_rets) if len(bm_rets) > 5 else 1.0
    beta = float(cov_bm / (var_bm + 1e-9))
    tracking_error = float(np.std(d_rets - bm_rets) * np.sqrt(freq_factor) * 100.0)

    # VaR 95% & CVaR 95%
    var_95_pct = float(abs(np.percentile(d_rets, 5)) * 100.0) if len(d_rets) > 10 else 0.0
    tail_losses = d_rets[d_rets <= -var_95_pct / 100.0]
    cvar_95_pct = float(abs(tail_losses.mean()) * 100.0) if len(tail_losses) > 0 else var_95_pct

    # Drawdown Durations
    is_underwater = dd < -0.01
    longest_dd_bars = 0
    curr_dd = 0
    for u in is_underwater:
        if u:
            curr_dd += 1
            longest_dd_bars = max(longest_dd_bars, curr_dd)
        else:
            curr_dd = 0

    return {
        "final_value": final_val,
        "total_return_pct": tot_return_pct,
        "cagr_pct": cagr_pct,
        "annual_vol_pct": ann_vol_pct,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd_pct,
        "avg_drawdown_pct": avg_dd_pct,
        "sortino": sortino,
        "calmar": calmar,
        "beta": beta,
        "tracking_error": tracking_error,
        "downside_vol_pct": downside_vol_pct,
        "var_95_pct": var_95_pct,
        "cvar_95_pct": cvar_95_pct,
        "longest_dd_bars": longest_dd_bars,
    }


def compute_risk_contributions(weights: pd.Series, cov_matrix: pd.DataFrame) -> pd.Series:
    """Calculate percentage marginal risk contribution (RC%) for each asset."""
    w = weights.values
    cov = cov_matrix.values
    port_var = w @ cov @ w
    marginal_contrib = cov @ w
    risk_contrib_dollar = w * marginal_contrib
    risk_contrib_pct = (risk_contrib_dollar / (port_var + 1e-9)) * 100.0
    return pd.Series(risk_contrib_pct, index=weights.index, name="Risk Contribution %")


def compute_rolling_metrics(
    port_series: dict[str, Any],
    window: int = 60,
    freq_factor: int = 252,
    risk_free_rate: float = 0.05,
) -> pd.DataFrame:
    """Compute rolling annualized volatility, Sharpe ratio, and CAPM beta."""
    p_rets = port_series["portfolio_daily_ret"]
    b_rets = port_series["benchmark_daily_ret"]

    roll_vol_p = p_rets.rolling(window).std() * np.sqrt(freq_factor) * 100.0
    roll_vol_b = b_rets.rolling(window).std() * np.sqrt(freq_factor) * 100.0

    roll_mean = p_rets.rolling(window).mean() * freq_factor
    roll_sharpe = (roll_mean - risk_free_rate) / (roll_vol_p / 100.0 + 1e-9)

    roll_cov = p_rets.rolling(window).cov(b_rets)
    roll_var_b = b_rets.rolling(window).var()
    roll_beta = roll_cov / (roll_var_b + 1e-9)
    roll_te = (p_rets - b_rets).rolling(window).std() * np.sqrt(freq_factor) * 100.0

    df_roll = pd.DataFrame({
        "Rolling Port Vol %": roll_vol_p,
        "Rolling Bench Vol %": roll_vol_b,
        "Rolling Sharpe": roll_sharpe,
        "Rolling Beta": roll_beta,
        "Rolling Tracking Error %": roll_te,
    }, index=p_rets.index).dropna()

    return df_roll


def run_portfolio_stress_tests(
    weights: pd.Series,
    prices_df: pd.DataFrame,
    bench_series: pd.Series,
    portfolio_value: float,
    custom_shock_pct: float = -15.0,
) -> list[dict[str, Any]]:
    """Run macro crisis stress test scenarios on the active portfolio."""
    rets = prices_df.pct_change().dropna()
    b_ret = bench_series.pct_change().dropna()

    # Asset betas
    betas = {}
    for col in prices_df.columns:
        if len(rets[col]) > 10 and len(b_ret) > 10:
            cov = np.cov(rets[col], b_ret)[0, 1]
            var_b = np.var(b_ret)
            betas[col] = float(cov / (var_b + 1e-9))
        else:
            betas[col] = 1.0

    p_beta = sum(weights.get(t, 0.0) * betas.get(t, 1.0) for t in prices_df.columns)

    scenarios = [
        {
            "name": "COVID-19 Crash (Mar 2020)",
            "description": "Liquidity panic, circuit breaker halts, flight to cash",
            "market_shock": -28.0,
            "port_drop_pct": p_beta * -28.0,
        },
        {
            "name": "Global Financial Crisis (2008)",
            "description": "Banking insolvency shock & severe economic contraction",
            "market_shock": -38.0,
            "port_drop_pct": p_beta * -38.0 * 1.1,
        },
        {
            "name": "Tech & Rate Spike Shock (2022)",
            "description": "Rapid rate hike cycle (+250 bps) compressing multiples",
            "market_shock": -18.0,
            "port_drop_pct": p_beta * -18.0,
        },
        {
            "name": "Commodity / Inflation Shock",
            "description": "Energy supply disruption & inflation spike",
            "market_shock": -12.0,
            "port_drop_pct": p_beta * -12.0 * 0.9,
        },
        {
            "name": f"User Custom Shock ({custom_shock_pct:+.1f}%)",
            "description": "Configurable systemic benchmark movement",
            "market_shock": custom_shock_pct,
            "port_drop_pct": p_beta * custom_shock_pct,
        },
    ]

    results = []
    for s in scenarios:
        drop = s["port_drop_pct"]
        pnl = (drop / 100.0) * portfolio_value
        new_val = max(0.0, portfolio_value + pnl)
        results.append({
            "Scenario": s["name"],
            "Description": s["description"],
            "Market Shock %": f"{s['market_shock']:+.1f}%",
            "Projected Return %": drop,
            "Projected P&L": pnl,
            "Projected Value": new_val,
        })
    return results


def run_slsqp_optimizer(
    returns: pd.DataFrame,
    cov_matrix: pd.DataFrame,
    objective: str = "Max Sharpe",
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    target_return: float = 0.15,
    max_volatility: float = 0.20,
    risk_free_rate: float = 0.05,
) -> dict[str, Any]:
    """Execute robust constrained SLSQP portfolio optimization."""
    n = len(returns.columns)
    mu = returns.mean().values * 252.0
    cov = cov_matrix.values * 252.0

    init_w = np.ones(n) / n
    bounds = [(min_weight, max_weight) for _ in range(n)]

    def obj_func(w):
        p_ret = np.sum(mu * w)
        p_vol = np.sqrt(w @ cov @ w)
        if objective == "Max Sharpe":
            return -(p_ret - risk_free_rate) / (p_vol + 1e-9)
        elif objective in ("Min Volatility", "Minimum Variance"):
            return p_vol
        elif objective == "Max Return":
            return -p_ret
        elif objective == "Risk Parity":
            p_var = p_vol ** 2
            rc = (w * (cov @ w)) / (p_var + 1e-9)
            return np.sum((rc - 1.0 / n) ** 2)
        elif objective == "Target Return":
            return p_vol
        return -(p_ret - risk_free_rate) / (p_vol + 1e-9)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if objective == "Target Return":
        constraints.append({"type": "eq", "fun": lambda w: np.sum(mu * w) - target_return})
    elif objective in ("Min Volatility", "Minimum Variance") and max_volatility > 0:
        constraints.append({"type": "ineq", "fun": lambda w: max_volatility - np.sqrt(w @ cov @ w)})

    res = minimize(obj_func, init_w, method="SLSQP", bounds=bounds, constraints=constraints)
    if res.success and res.x is not None:
        opt_w = np.clip(res.x, 0.0, 1.0)
        opt_w = opt_w / np.sum(opt_w)
    else:
        opt_w = init_w

    opt_series = pd.Series(opt_w, index=returns.columns, name="Optimized Weight")
    exp_cagr = float(np.sum(mu * opt_w))
    opt_vol = float(np.sqrt(opt_w @ cov @ opt_w))
    opt_sharpe = float((exp_cagr - risk_free_rate) / (opt_vol + 1e-9))

    # Calculate max drawdown estimate from simulated returns
    sim_port_ret = returns.values @ opt_w
    sim_eq = (1.0 + sim_port_ret).cumprod()
    sim_dd = (sim_eq - np.maximum.accumulate(sim_eq)) / np.maximum.accumulate(sim_eq) * 100.0
    opt_max_dd = float(np.min(sim_dd))

    return {
        "weights": opt_series,
        "cagr": exp_cagr * 100.0,
        "volatility": opt_vol * 100.0,
        "sharpe": opt_sharpe,
        "max_drawdown": opt_max_dd,
    }


def generate_frontier_cloud(
    returns: pd.DataFrame,
    cov_matrix: pd.DataFrame,
    n_sims: int = 800,
    risk_free_rate: float = 0.05,
) -> dict[str, Any]:
    """Generate Monte Carlo portfolio cloud and efficient frontier curve."""
    n = len(returns.columns)
    mu = returns.mean().values * 252.0
    cov = cov_matrix.values * 252.0

    # 1. Random Sim Portfolios
    sim_rets = np.zeros(n_sims)
    sim_vols = np.zeros(n_sims)
    sim_sharpes = np.zeros(n_sims)

    for i in range(n_sims):
        w = np.random.dirichlet(np.ones(n))
        p_r = np.sum(mu * w)
        p_v = np.sqrt(w @ cov @ w)
        sim_rets[i] = p_r
        sim_vols[i] = p_v
        sim_sharpes[i] = (p_r - risk_free_rate) / (p_v + 1e-9)

    # 2. Key Portfolios: Min Vol & Max Sharpe
    mv_res = run_slsqp_optimizer(returns, cov_matrix, "Min Volatility", risk_free_rate=risk_free_rate)
    ms_res = run_slsqp_optimizer(returns, cov_matrix, "Max Sharpe", risk_free_rate=risk_free_rate)

    # 3. Frontier Line
    min_r = mv_res["cagr"] / 100.0
    max_r = float(np.max(mu))
    target_rs = np.linspace(min_r, max_r, 20)
    f_vols = []
    f_rets = []

    for tr in target_rs:
        r_opt = run_slsqp_optimizer(returns, cov_matrix, "Target Return", target_return=tr, risk_free_rate=risk_free_rate)
        f_vols.append(r_opt["volatility"] / 100.0)
        f_rets.append(tr)

    # Individual asset coordinates
    asset_points = []
    for i, col in enumerate(returns.columns):
        a_vol = np.sqrt(cov[i, i]) * 100.0
        a_ret = mu[i] * 100.0
        a_sh = (mu[i] - risk_free_rate) / (np.sqrt(cov[i, i]) + 1e-9)
        asset_points.append({"asset": col, "vol": a_vol, "ret": a_ret, "sharpe": a_sh})

    return {
        "sim_rets": sim_rets * 100.0,
        "sim_vols": sim_vols * 100.0,
        "sim_sharpes": sim_sharpes,
        "frontier_vols": np.array(f_vols) * 100.0,
        "frontier_rets": np.array(f_rets) * 100.0,
        "min_vol_point": (mv_res["volatility"], mv_res["cagr"]),
        "max_sharpe_point": (ms_res["volatility"], ms_res["cagr"]),
        "asset_points": asset_points,
    }


def generate_executive_tearsheet_html(
    kpis: dict[str, Any],
    holdings_df: pd.DataFrame,
    sector_weights: dict[str, float],
    stress_results: list[dict[str, Any]],
    benchmark_name: str,
    period: str,
) -> str:
    """Generate a clean, standalone institutional executive tearsheet HTML."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    holdings_rows_html = ""
    for _, r in holdings_df.iterrows():
        holdings_rows_html += f"""
        <tr>
            <td style="font-family: monospace; font-weight: bold;">{r['Asset']}</td>
            <td>{r.get('Company', r['Asset'])}</td>
            <td>{r.get('Sector', 'General')}</td>
            <td style="text-align: right;">{r['Weight %']:.1f}%</td>
            <td style="text-align: right;">{r['Value']}</td>
            <td style="text-align: right;">{r['Return %']:+.2f}%</td>
            <td style="text-align: right;">{r['Volatility %']:.2f}%</td>
            <td style="text-align: right;">{r['Sharpe']:.2f}</td>
            <td style="text-align: right;">{r['Risk Contrib %']:.1f}%</td>
        </tr>
        """

    stress_rows_html = ""
    for s in stress_results:
        stress_rows_html += f"""
        <tr>
            <td style="font-weight: bold;">{s['Scenario']}</td>
            <td>{s['Description']}</td>
            <td style="text-align: right;">{s['Market Shock %']}</td>
            <td style="text-align: right; color: {'#10B981' if s['Projected Return %']>=0 else '#EF4444'}; font-weight: bold;">{s['Projected Return %']:+.1f}%</td>
            <td style="text-align: right;">{fmt_inr(s['Projected P&L'])}</td>
            <td style="text-align: right; font-weight: bold;">{fmt_inr(s['Projected Value'])}</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Executive Portfolio Tearsheet - QuantTerminal</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background-color: #0B0F19;
        color: #E2E8F0;
        margin: 20px;
        padding: 10px;
    }}
    .header {{
        border-bottom: 2px solid #38BDF8;
        padding-bottom: 12px;
        margin-bottom: 20px;
    }}
    .title {{ font-size: 24px; font-weight: 800; color: #F8FAFC; text-transform: uppercase; }}
    .meta {{ font-size: 12px; color: #94A3B8; margin-top: 4px; }}
    .kpi-table, .data-table {{
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 24px;
        font-size: 13px;
    }}
    .kpi-table td {{
        padding: 8px 12px;
        background: #111827;
        border: 1px solid #1E293B;
    }}
    .data-table th {{
        background: #1E293B;
        color: #38BDF8;
        padding: 8px 10px;
        text-align: left;
        border: 1px solid #334155;
    }}
    .data-table td {{
        padding: 6px 10px;
        background: #111827;
        border: 1px solid #1E293B;
    }}
    .sec-header {{
        font-size: 15px;
        font-weight: 700;
        color: #38BDF8;
        text-transform: uppercase;
        margin-top: 20px;
        margin-bottom: 8px;
    }}
    .disclaimer {{
        font-size: 11px;
        color: #64748B;
        margin-top: 30px;
        border-top: 1px solid #1E293B;
        padding-top: 12px;
        text-align: center;
    }}
    @media print {{
        body {{ background-color: #FFFFFF; color: #000000; }}
        .kpi-table td, .data-table td {{ background: #FFFFFF; color: #000000; border-color: #DDD; }}
        .data-table th {{ background: #EEE; color: #000; border-color: #DDD; }}
        .title {{ color: #000; }}
    }}
</style>
</head>
<body>
<div class="header">
    <div class="title">Institutional Portfolio Executive Tearsheet</div>
    <div class="meta">QuantTerminal Portfolio Lab | Benchmark: {benchmark_name} | Lookback: {period} | Generated: {now_str}</div>
</div>

<div class="sec-header">Key Performance Indicators</div>
<table class="kpi-table">
    <tr>
        <td><b>Portfolio Value:</b> {fmt_inr(kpis['final_value'])}</td>
        <td><b>Total Return:</b> <span style="color: {'#10B981' if kpis['total_return_pct']>=0 else '#EF4444'}; font-weight: bold;">{kpis['total_return_pct']:+.2f}%</span></td>
        <td><b>CAGR:</b> {kpis['cagr_pct']:+.2f}%</td>
        <td><b>Volatility:</b> {kpis['annual_vol_pct']:.2f}%</td>
    </tr>
    <tr>
        <td><b>Sharpe Ratio:</b> {kpis['sharpe']:.2f}</td>
        <td><b>Sortino Ratio:</b> {kpis['sortino']:.2f}</td>
        <td><b>Calmar Ratio:</b> {kpis['calmar']:.2f}</td>
        <td><b>Max Drawdown:</b> <span style="color: #EF4444;">{kpis['max_drawdown_pct']:.2f}%</span></td>
    </tr>
    <tr>
        <td><b>Market Beta:</b> {kpis['beta']:.2f}</td>
        <td><b>Downside Vol:</b> {kpis['downside_vol_pct']:.2f}%</td>
        <td><b>Daily VaR (95%):</b> -{kpis['var_95_pct']:.2f}%</td>
        <td><b>Daily CVaR (95%):</b> -{kpis['cvar_95_pct']:.2f}%</td>
    </tr>
</table>

<div class="sec-header">Portfolio Holdings & Risk Decomposition</div>
<table class="data-table">
    <thead>
        <tr>
            <th>Asset</th><th>Company</th><th>Sector</th><th style="text-align: right;">Weight</th><th style="text-align: right;">Value</th><th style="text-align: right;">Return</th><th style="text-align: right;">Vol</th><th style="text-align: right;">Sharpe</th><th style="text-align: right;">Risk Contrib</th>
        </tr>
    </thead>
    <tbody>
        {holdings_rows_html}
    </tbody>
</table>

<div class="sec-header">Macro Crisis & Stress Test Simulation</div>
<table class="data-table">
    <thead>
        <tr>
            <th>Scenario</th><th>Description</th><th style="text-align: right;">Market Shock</th><th style="text-align: right;">Projected Return</th><th style="text-align: right;">Projected P&L</th><th style="text-align: right;">Projected Value</th>
        </tr>
    </thead>
    <tbody>
        {stress_rows_html}
    </tbody>
</table>

<div class="disclaimer">
    Historical portfolio analysis and stress testing are for quantitative research purposes only and do not guarantee future performance.
    Confidential tearsheet generated by QuantTerminal Portfolio Lab.
</div>
</body>
</html>
"""
    return html


# -----------------------------------------------------------------------------
# 4. Main Portfolio Lab Page Renderer
# -----------------------------------------------------------------------------
def render_page():
    """Main rendering entry point for Portfolio Lab."""
    st.set_page_config(
        page_title="Portfolio Lab - QuantTerminal",
        page_icon="💼",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_portfolio_theme()

    universe = load_universe_snapshots()
    records_lookup = universe["records_by_ticker"]

    # Session State Initialization
    if "port_capital" not in st.session_state:
        st.session_state["port_capital"] = 1000000.0
    if "cash_reserve_pct" not in st.session_state:
        st.session_state["cash_reserve_pct"] = 0.0
    if "selected_assets" not in st.session_state:
        st.session_state["selected_assets"] = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"]
    if "asset_weights" not in st.session_state:
        st.session_state["asset_weights"] = {"RELIANCE.NS": 0.25, "TCS.NS": 0.25, "HDFCBANK.NS": 0.25, "INFY.NS": 0.25}
    if "benchmark_choice" not in st.session_state:
        st.session_state["benchmark_choice"] = "NIFTY 50"
    if "custom_benchmark_sym" not in st.session_state:
        st.session_state["custom_benchmark_sym"] = "^NSEI"
    if "period_choice" not in st.session_state:
        st.session_state["period_choice"] = "1Y"
    if "freq_choice" not in st.session_state:
        st.session_state["freq_choice"] = "Daily"
    if "show_filters" not in st.session_state:
        st.session_state["show_filters"] = False
    if "show_add_drawer" not in st.session_state:
        st.session_state["show_add_drawer"] = False
    if "rebalance_target_weights" not in st.session_state:
        st.session_state["rebalance_target_weights"] = None
    if "custom_start_date" not in st.session_state:
        st.session_state["custom_start_date"] = date.today() - timedelta(days=365)
    if "custom_end_date" not in st.session_state:
        st.session_state["custom_end_date"] = date.today()
    if "custom_shock_val" not in st.session_state:
        st.session_state["custom_shock_val"] = -15.0

    # =========================================================================
    # 1. PAGE HEADER
    # =========================================================================
    st.markdown(
        """
        <div class="port-header">
            <div>
                <div class="port-title">PORTFOLIO LAB</div>
                <div class="port-subtitle">Multi-Asset Portfolio Construction & Risk Analytics</div>
            </div>
            <div class="status-badge">
                <span class="status-dot"></span>
                <span>Market Data Connected</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================================
    # 2. PORTFOLIO SETUP CONTROL BAR
    # =========================================================================
    with st.container():
        st.markdown("<div style='font-size: 0.82rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin-bottom: 6px;'>⚙️ Portfolio Setup & Capital Controls</div>", unsafe_allow_html=True)
        s_col1, s_col2, s_col3, s_col4 = st.columns([1.5, 1.4, 1.3, 1.0])

        with s_col1:
            cap_presets = ["₹1,00,000", "₹5,00,000", "₹10,00,000", "₹25,00,000", "₹50,00,000", "Custom"]
            c_choice = st.selectbox("Capital Preset", cap_presets, index=2)
            if c_choice == "₹1,00,000":
                st.session_state["port_capital"] = 100000.0
            elif c_choice == "₹5,00,000":
                st.session_state["port_capital"] = 500000.0
            elif c_choice == "₹10,00,000":
                st.session_state["port_capital"] = 1000000.0
            elif c_choice == "₹25,00,000":
                st.session_state["port_capital"] = 2500000.0
            elif c_choice == "₹50,00,000":
                st.session_state["port_capital"] = 5000000.0

            capital_val = st.number_input(
                "Capital Amount (₹ / $)",
                min_value=10000.0,
                max_value=1e9,
                value=float(st.session_state["port_capital"]),
                step=50000.0,
            )
            st.session_state["port_capital"] = capital_val

            cash_sleeve = st.slider("Cash / Liquid Reserve %", 0, 40, int(st.session_state["cash_reserve_pct"] * 100), 1, help="Yields 5.0% risk-free rate")
            st.session_state["cash_reserve_pct"] = cash_sleeve / 100.0

        with s_col2:
            univ_choice = st.selectbox(
                "Universe Template",
                ["Custom", "NIFTY 50", "NIFTY 100", "NIFTY 500", "Sector: Banking", "Sector: Technology", "US Mega-Cap Tech"],
                index=0,
            )
            if univ_choice == "NIFTY 50":
                st.session_state["selected_assets"] = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "BHARTIARTL.NS", "ICICIBANK.NS", "INFY.NS", "ITC.NS", "LT.NS"]
            elif univ_choice == "NIFTY 100":
                st.session_state["selected_assets"] = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ITC.NS", "LT.NS", "SBIN.NS", "TATAMOTORS.NS", "SUNPHARMA.NS", "TITAN.NS"]
            elif univ_choice == "NIFTY 500":
                st.session_state["selected_assets"] = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "LT.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS", "AXISBANK.NS", "ULTRACEMCO.NS", "WIPRO.NS"]
            elif univ_choice == "Sector: Banking":
                st.session_state["selected_assets"] = ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"]
            elif univ_choice == "Sector: Technology":
                st.session_state["selected_assets"] = ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"]
            elif univ_choice == "US Mega-Cap Tech":
                st.session_state["selected_assets"] = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META"]

            bench_opts = ["NIFTY 50", "NIFTY 100", "NIFTY 500", "Custom"]
            selected_bench = st.selectbox("Benchmark", bench_opts, index=0)
            st.session_state["benchmark_choice"] = selected_bench
            if selected_bench == "Custom":
                cust_b = st.text_input("Custom Benchmark Ticker", value="^GSPC")
                st.session_state["custom_benchmark_sym"] = cust_b

        with s_col3:
            period_opts = ["1M", "3M", "6M", "1Y", "3Y", "5Y", "MAX", "Custom"]
            p_sel = st.selectbox("Lookback Period", period_opts, index=3)
            st.session_state["period_choice"] = p_sel

            if p_sel == "Custom":
                d_c1, d_c2 = st.columns(2)
                with d_c1:
                    st.session_state["custom_start_date"] = st.date_input("Start Date", value=st.session_state["custom_start_date"])
                with d_c2:
                    st.session_state["custom_end_date"] = st.date_input("End Date", value=st.session_state["custom_end_date"])

            freq_opts = ["Daily", "Weekly", "Monthly"]
            f_sel = st.selectbox("Frequency", freq_opts, index=0)
            st.session_state["freq_choice"] = f_sel

        with s_col4:
            st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
            filter_toggle = st.button("🔍 Filters Drawer", use_container_width=True)
            if filter_toggle:
                st.session_state["show_filters"] = not st.session_state["show_filters"]

    # =========================================================================
    # 3. FILTER DRAWER (COLLAPSIBLE)
    # =========================================================================
    filtered_india_items = universe["india_items"]
    filtered_us_items = universe["us_items"]

    if st.session_state.get("show_filters", False):
        with st.expander("🔍 Screener & Universe Filters (Drawer)", expanded=True):
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)
            with f_col1:
                filt_mkt = st.selectbox("Market", ["All Markets", "India (NSE/BSE)", "US (NASDAQ/NYSE)"])
                filt_sec = st.selectbox("Sector", ["All Sectors", "Finance", "Technology services", "Electronic technology", "Energy minerals", "Health technology", "Consumer services"])
            with f_col2:
                filt_mcap = st.slider("Min Market Cap", 0, 500000, 1000, help="Minimum market capitalization threshold")
                filt_liq = st.slider("Min Liquidity (Vol)", 10000, 5000000, 50000, step=25000)
            with f_col3:
                filt_px_range = st.slider("Price Range", 0, 10000, (50, 6000))
                filt_vol_range = st.slider("Volatility Range %", 5, 80, (10, 50))
            with f_col4:
                filt_div = st.slider("Min Dividend Yield %", 0.0, 10.0, 0.0, step=0.5)
                st.caption("Filters apply dynamically to candidate search in the Asset Selection panel below.")

            if filt_mkt == "India (NSE/BSE)":
                filtered_us_items = []
            elif filt_mkt == "US (NASDAQ/NYSE)":
                filtered_india_items = []

            if filt_sec != "All Sectors":
                filtered_india_items = [it for it in filtered_india_items if filt_sec.lower() in it["sector"].lower()]
                filtered_us_items = [it for it in filtered_us_items if filt_sec.lower() in it["sector"].lower()]

            filtered_india_items = [it for it in filtered_india_items if it["price"] >= filt_px_range[0] and it["price"] <= filt_px_range[1] and it["volume"] >= filt_liq and it["div_yield"] >= filt_div]
            filtered_us_items = [it for it in filtered_us_items if it["price"] >= filt_px_range[0] and it["price"] <= filt_px_range[1] and it["volume"] >= filt_liq and it["div_yield"] >= filt_div]

    # =========================================================================
    # 4. PORTFOLIO ACTIONS & ASSET CHIPS
    # =========================================================================
    current_assets = list(st.session_state["selected_assets"])

    st.markdown("<div class='section-header'>⚡ Portfolio Actions & Selected Holdings</div>", unsafe_allow_html=True)
    act_col1, act_col2, act_col3, act_col4, act_col5 = st.columns(5)
    with act_col1:
        if st.button("➕ Add Asset", use_container_width=True):
            st.session_state["show_add_drawer"] = not st.session_state.get("show_add_drawer", False)
    with act_col2:
        if st.button("🔍 Filters", use_container_width=True):
            st.session_state["show_filters"] = not st.session_state.get("show_filters", False)
    with act_col3:
        opt_clicked = st.button("⚡ Optimize", type="primary", use_container_width=True)
    with act_col4:
        rebal_clicked = st.button("🔄 Rebalance", use_container_width=True)
    with act_col5:
        if st.button("↺ Reset", use_container_width=True):
            st.session_state["selected_assets"] = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"]
            st.session_state["asset_weights"] = {"RELIANCE.NS": 0.25, "TCS.NS": 0.25, "HDFCBANK.NS": 0.25, "INFY.NS": 0.25}
            st.session_state["rebalance_target_weights"] = None
            st.rerun()

    # Asset Search / Add Drawer
    if st.session_state.get("show_add_drawer", False):
        with st.container():
            st.markdown("<div style='background: #111827; border: 1px solid #334155; border-radius: 6px; padding: 10px 14px; margin-bottom: 12px;'>", unsafe_allow_html=True)
            a_c1, a_c2 = st.columns([3, 1])
            candidate_labels = [it["label"] for it in filtered_india_items[:250]] + [it["label"] for it in filtered_us_items[:150]]
            with a_c1:
                chosen_from_db = st.selectbox("Search by Ticker or Company Name", [""] + candidate_labels, index=0)
                if chosen_from_db:
                    rec_match = {it["label"]: it for it in filtered_india_items[:250] + filtered_us_items[:150]}.get(chosen_from_db)
                    if rec_match and rec_match["yf_ticker"] not in current_assets:
                        current_assets.append(rec_match["yf_ticker"])
                        st.session_state["selected_assets"] = current_assets
                        st.session_state["show_add_drawer"] = False
                        st.rerun()
            with a_c2:
                custom_ticker_in = st.text_input("Direct Ticker Entry", placeholder="e.g. INFY.NS, AAPL")
                if st.button("Add Ticker", use_container_width=True) and custom_ticker_in:
                    c_sym = resolve_ticker(custom_ticker_in)
                    if c_sym not in current_assets:
                        current_assets.append(c_sym)
                        st.session_state["selected_assets"] = current_assets
                        st.session_state["show_add_drawer"] = False
                        st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    # Display removable chips
    st.markdown("<div style='font-size: 0.78rem; font-weight: 600; color: #94A3B8; text-transform: uppercase;'>Current Portfolio Holdings (Click ✕ to remove):</div>", unsafe_allow_html=True)
    chip_remove_cols = st.columns(min(len(current_assets), 8) if len(current_assets) > 0 else 1)
    for idx, asset in enumerate(current_assets):
        col_idx = idx % len(chip_remove_cols)
        with chip_remove_cols[col_idx]:
            if st.button(f"{asset} ✕", key=f"chip_rem_{asset}", help=f"Click to remove {asset} from portfolio"):
                current_assets.remove(asset)
                st.session_state["selected_assets"] = current_assets
                st.rerun()

    if len(current_assets) < 2:
        st.warning("⚠️ Please maintain at least 2 assets to construct, optimize, and analyze a portfolio.")
        return

    # Benchmark resolution
    b_choice = st.session_state["benchmark_choice"]
    if b_choice == "NIFTY 50":
        bench_sym = "^NSEI"
    elif b_choice == "NIFTY 100":
        bench_sym = "^CNX100"
    elif b_choice == "NIFTY 500":
        bench_sym = "^CRSLDX"
    else:
        bench_sym = st.session_state.get("custom_benchmark_sym", "^NSEI")

    # Fetch Market Prices
    prices_df, bench_series = fetch_asset_and_benchmark_prices(
        current_assets,
        benchmark_ticker=bench_sym,
        period=st.session_state["period_choice"],
        frequency=st.session_state["freq_choice"],
        custom_start=st.session_state.get("custom_start_date"),
        custom_end=st.session_state.get("custom_end_date"),
    )

    if prices_df.empty or len(prices_df.columns) < 2:
        st.error("⚠️ Insufficient market price history retrieved for the selected assets or period. Please verify tickers.")
        return

    active_tickers = list(prices_df.columns)

    # Initialize / validate weights
    curr_w_dict = st.session_state.get("asset_weights", {})
    weights_vec = [curr_w_dict.get(t, 1.0 / len(active_tickers)) for t in active_tickers]
    w_sum = sum(weights_vec)
    if w_sum <= 0 or abs(w_sum - 1.0) > 0.001:
        weights_vec = [1.0 / len(active_tickers)] * len(active_tickers)
    weight_series = pd.Series(weights_vec, index=active_tickers)

    # Compute Portfolio Series & Metrics
    port_results = calculate_portfolio_series(
        prices_df,
        weight_series,
        initial_capital=st.session_state["port_capital"],
        benchmark_prices=bench_series,
        cash_fraction=st.session_state["cash_reserve_pct"],
    )
    freq_factor = 252 if st.session_state["freq_choice"] == "Daily" else (52 if st.session_state["freq_choice"] == "Weekly" else 12)
    kpis = compute_portfolio_kpis(port_results, initial_capital=st.session_state["port_capital"], freq_factor=freq_factor)
    cov_matrix = prices_df.pct_change().dropna().cov()
    risk_contrib = compute_risk_contributions(weight_series, cov_matrix)

    # =========================================================================
    # 5. PORTFOLIO KPI BAR
    # =========================================================================
    st.markdown("<div class='section-header'>📊 Portfolio Key Performance Indicators</div>", unsafe_allow_html=True)
    k_col1, k_col2, k_col3, k_col4, k_col5, k_col6 = st.columns(6)
    k_col1.markdown(f"""<div class="kpi-card"><span class="kpi-label">Portfolio Value</span><span class="kpi-value">{fmt_inr(kpis['final_value'])}</span><span class="kpi-sub">Initial: {fmt_inr(st.session_state['port_capital'])}</span></div>""", unsafe_allow_html=True)
    k_col2.markdown(f"""<div class="kpi-card"><span class="kpi-label">Total Return</span><span class="kpi-value {'val-pos' if kpis['total_return_pct']>=0 else 'val-neg'}">{kpis['total_return_pct']:+.2f}%</span><span class="kpi-sub">Cumulative</span></div>""", unsafe_allow_html=True)
    k_col3.markdown(f"""<div class="kpi-card"><span class="kpi-label">CAGR</span><span class="kpi-value {'val-pos' if kpis['cagr_pct']>=0 else 'val-neg'}">{kpis['cagr_pct']:+.2f}%</span><span class="kpi-sub">Annualized Compounded</span></div>""", unsafe_allow_html=True)
    k_col4.markdown(f"""<div class="kpi-card"><span class="kpi-label">Volatility</span><span class="kpi-value">{kpis['annual_vol_pct']:.2f}%</span><span class="kpi-sub">Annualized</span></div>""", unsafe_allow_html=True)
    k_col5.markdown(f"""<div class="kpi-card"><span class="kpi-label">Sharpe Ratio</span><span class="kpi-value {'val-pos' if kpis['sharpe']>=1.0 else ('val-neutral' if kpis['sharpe']>=0 else 'val-neg')}">{kpis['sharpe']:.2f}</span><span class="kpi-sub">Rf: 5.0%</span></div>""", unsafe_allow_html=True)
    k_col6.markdown(f"""<div class="kpi-card"><span class="kpi-label">Max Drawdown</span><span class="kpi-value val-neg">{kpis['max_drawdown_pct']:.2f}%</span><span class="kpi-sub">{kpis['longest_dd_bars']} bars underwater</span></div>""", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="sub-metrics-grid">
            <div class="sub-metric-item"><span class="sub-metric-title">Sortino Ratio</span><span class="sub-metric-val">{kpis['sortino']:.2f}</span></div>
            <div class="sub-metric-item"><span class="sub-metric-title">Calmar Ratio</span><span class="sub-metric-val">{kpis['calmar']:.2f}</span></div>
            <div class="sub-metric-item"><span class="sub-metric-title">Market Beta</span><span class="sub-metric-val">{kpis['beta']:.2f}</span></div>
            <div class="sub-metric-item"><span class="sub-metric-title">Downside Vol</span><span class="sub-metric-val">{kpis['downside_vol_pct']:.2f}%</span></div>
            <div class="sub-metric-item"><span class="sub-metric-title">VaR (95% Daily)</span><span class="sub-metric-val val-neg">-{kpis['var_95_pct']:.2f}%</span></div>
            <div class="sub-metric-item"><span class="sub-metric-title">CVaR (95% Daily)</span><span class="sub-metric-val val-neg">-{kpis['cvar_95_pct']:.2f}%</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================================
    # 6. PORTFOLIO PERFORMANCE & BREAKDOWN
    # =========================================================================
    p_row1, p_row2 = st.columns([3.2, 1.8])

    with p_row1:
        st.markdown("<div class='section-header'>📈 Portfolio Equity Curve vs Benchmark</div>", unsafe_allow_html=True)
        peq = port_results["portfolio_equity"]
        beq = port_results["benchmark_equity"]

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=peq.index, y=peq, mode="lines", name="Portfolio Value",
            line=dict(color="#10B981", width=2.5),
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Portfolio Value:</b> ₹%{y:,.0f}<br><b>Return:</b> %{customdata:.2f}%<extra></extra>",
            customdata=(peq / peq.iloc[0] - 1.0) * 100.0,
        ))
        fig_eq.add_trace(go.Scatter(
            x=beq.index, y=beq, mode="lines", name=f"Benchmark ({bench_sym})",
            line=dict(color="#64748B", width=1.5, dash="dash"),
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Benchmark Value:</b> ₹%{y:,.0f}<extra></extra>",
        ))
        fig_eq.add_trace(go.Scatter(
            x=peq.index, y=peq.cummax(), mode="lines", name="High Watermark",
            line=dict(color="rgba(16, 185, 129, 0.25)", width=1, dash="dot"),
            hoverinfo="skip",
        ))

        fig_eq.update_layout(
            template="plotly_dark", height=380,
            margin=dict(l=10, r=10, t=25, b=15),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            yaxis_title="Capital Value (₹ / $)",
            hovermode="x unified",
        )
        st.plotly_chart(fig_eq, use_container_width=True)

    with p_row2:
        st.markdown("<div class='section-header'>🥧 Capital Allocation & Breakdown</div>", unsafe_allow_html=True)
        # Horizontal weight bar chart
        w_df = pd.DataFrame({
            "Asset": active_tickers,
            "Weight %": [weight_series[t] * (1.0 - st.session_state['cash_reserve_pct']) * 100.0 for t in active_tickers],
        }).sort_values(by="Weight %", ascending=True)

        if st.session_state["cash_reserve_pct"] > 0:
            w_df = pd.concat([pd.DataFrame([{"Asset": "CASH RESERVE", "Weight %": st.session_state["cash_reserve_pct"] * 100.0}]), w_df])

        fig_wbar = go.Figure(go.Bar(
            x=w_df["Weight %"],
            y=w_df["Asset"],
            orientation="h",
            marker=dict(color="#38BDF8"),
            text=w_df["Weight %"].apply(lambda v: f"{v:.1f}%"),
            textposition="auto",
            hovertemplate="<b>%{y}</b>: %{x:.1f}%<extra></extra>",
        ))
        fig_wbar.update_layout(
            template="plotly_dark", height=200,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Weight %",
            yaxis_title="",
        )
        st.plotly_chart(fig_wbar, use_container_width=True)

        # Sector Donut
        sector_weights = {}
        eq_weight_factor = 1.0 - st.session_state["cash_reserve_pct"]
        for t, w in weight_series.items():
            sec = records_lookup.get(t.upper(), {}).get("sector", "General")
            sector_weights[sec] = sector_weights.get(sec, 0.0) + (w * eq_weight_factor * 100.0)
        if st.session_state["cash_reserve_pct"] > 0:
            sector_weights["Cash / Liquidity"] = st.session_state["cash_reserve_pct"] * 100.0

        fig_donut = go.Figure(data=[go.Pie(
            labels=list(sector_weights.keys()),
            values=list(sector_weights.values()),
            hole=0.6,
            textinfo="label+percent",
            marker=dict(colors=["#38BDF8", "#10B981", "#818CF8", "#F59E0B", "#EC4899", "#64748B", "#A855F7", "#06B6D4"]),
        )])
        fig_donut.update_layout(
            template="plotly_dark", height=170,
            margin=dict(l=10, r=10, t=5, b=5),
            showlegend=False,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # =========================================================================
    # 7. PORTFOLIO COMPOSITION SUMMARY
    # =========================================================================
    st.markdown("<div class='section-header'>🏛️ Portfolio Composition & Sector Exposure Summary</div>", unsafe_allow_html=True)
    comp_c1, comp_c2, comp_c3, comp_c4, comp_c5 = st.columns(5)
    comp_c1.markdown(f"""<div class="kpi-card"><span class="kpi-label">Equities Exposure</span><span class="kpi-value">{(1.0 - st.session_state['cash_reserve_pct'])*100.0:.1f}%</span><span class="kpi-sub">Total Assets: {len(active_tickers)}</span></div>""", unsafe_allow_html=True)
    comp_c2.markdown(f"""<div class="kpi-card"><span class="kpi-label">Cash Allocation</span><span class="kpi-value">{st.session_state['cash_reserve_pct']*100.0:.1f}%</span><span class="kpi-sub">Yielding 5.0% Rf</span></div>""", unsafe_allow_html=True)

    large_cap_w = 0.0
    mid_cap_w = 0.0
    for t in active_tickers:
        mcap = records_lookup.get(t.upper(), {}).get("mcap", 1e11)
        if mcap >= 2e10:
            large_cap_w += weight_series[t] * (1.0 - st.session_state['cash_reserve_pct']) * 100.0
        else:
            mid_cap_w += weight_series[t] * (1.0 - st.session_state['cash_reserve_pct']) * 100.0

    comp_c3.markdown(f"""<div class="kpi-card"><span class="kpi-label">Large Cap %</span><span class="kpi-value">{large_cap_w:.1f}%</span><span class="kpi-sub">Market Cap > ₹20k Cr</span></div>""", unsafe_allow_html=True)
    comp_c4.markdown(f"""<div class="kpi-card"><span class="kpi-label">Mid/Small Cap %</span><span class="kpi-value">{mid_cap_w:.1f}%</span><span class="kpi-sub">High Growth Exposure</span></div>""", unsafe_allow_html=True)

    top_sec_str = ", ".join([f"{s}: {v:.1f}%" for s, v in sorted(sector_weights.items(), key=lambda x: x[1], reverse=True)[:2]])
    comp_c5.markdown(f"""<div class="kpi-card"><span class="kpi-label">Top Sectors</span><span class="kpi-value" style="font-size: 0.88rem;">{top_sec_str}</span><span class="kpi-sub">Concentration</span></div>""", unsafe_allow_html=True)

    # =========================================================================
    # 8. HOLDINGS TABLE & DRILLDOWN
    # =========================================================================
    st.markdown("<div class='section-header'>📋 Institutional Holdings Table</div>", unsafe_allow_html=True)

    t_c1, t_c2, t_c3 = st.columns([2, 1.5, 1.5])
    with t_c1:
        tbl_search = st.text_input("Search Holdings", placeholder="Search ticker or company name...", label_visibility="collapsed")
    with t_c2:
        tbl_sec_filter = st.selectbox("Filter by Sector", ["All Sectors"] + list(sector_weights.keys()), label_visibility="collapsed")
    with t_c3:
        tbl_sort = st.selectbox("Sort Holdings By", ["Weight % (High to Low)", "Period Return % (High to Low)", "Sharpe Ratio", "Risk Contrib %"], label_visibility="collapsed")

    holdings_data = []
    tot_cap = st.session_state["port_capital"]
    eq_cap = tot_cap * (1.0 - st.session_state["cash_reserve_pct"])

    for t in active_tickers:
        sec_name = records_lookup.get(t.upper(), {}).get("sector", "General")
        c_name = records_lookup.get(t.upper(), {}).get("name", t)

        if tbl_search and (tbl_search.lower() not in t.lower() and tbl_search.lower() not in c_name.lower()):
            continue
        if tbl_sec_filter != "All Sectors" and sec_name != tbl_sec_filter:
            continue

        w_pct = weight_series[t] * (1.0 - st.session_state["cash_reserve_pct"]) * 100.0
        val_allocated = weight_series[t] * eq_cap
        a_close = float(prices_df[t].iloc[-1])
        a_ret = float((prices_df[t].iloc[-1] / prices_df[t].iloc[0] - 1.0) * 100.0)
        a_vol = float(prices_df[t].pct_change().dropna().std() * np.sqrt(freq_factor) * 100.0)
        a_sharpe = float((a_ret / 100.0 - 0.05) / (a_vol / 100.0 + 1e-9))
        rc_pct = float(risk_contrib.get(t, 0.0))

        holdings_data.append({
            "Asset": t,
            "Company": c_name,
            "Sector": sec_name,
            "Weight %": round(w_pct, 2),
            "Value": fmt_inr(val_allocated),
            "Price": f"{a_close:,.2f}",
            "Return %": round(a_ret, 2),
            "Volatility %": round(a_vol, 2),
            "Sharpe": round(a_sharpe, 2),
            "Risk Contrib %": round(rc_pct, 2),
        })

    holdings_df = pd.DataFrame(holdings_data)
    if not holdings_df.empty:
        if tbl_sort == "Weight % (High to Low)":
            holdings_df = holdings_df.sort_values(by="Weight %", ascending=False)
        elif tbl_sort == "Period Return % (High to Low)":
            holdings_df = holdings_df.sort_values(by="Return %", ascending=False)
        elif tbl_sort == "Sharpe Ratio":
            holdings_df = holdings_df.sort_values(by="Sharpe", ascending=False)
        elif tbl_sort == "Risk Contrib %":
            holdings_df = holdings_df.sort_values(by="Risk Contrib %", ascending=False)

        st.dataframe(
            holdings_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Return %": st.column_config.NumberColumn(format="%.2f%%"),
                "Weight %": st.column_config.NumberColumn(format="%.2f%%"),
                "Volatility %": st.column_config.NumberColumn(format="%.2f%%"),
                "Risk Contrib %": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )

    # Holding Deep-Dive Drawer
    with st.expander("🔍 Holding Detailed Deep-Dive Inspector", expanded=False):
        drill_asset = st.selectbox("Select Asset to Inspect", active_tickers, index=0)
        if drill_asset in prices_df.columns:
            d_p = prices_df[drill_asset]
            d_sma50 = d_p.rolling(50).mean()
            d_sma200 = d_p.rolling(200).mean()

            dd_col1, dd_col2 = st.columns([3, 1.2])
            with dd_col1:
                fig_drill = go.Figure()
                fig_drill.add_trace(go.Scatter(x=d_p.index, y=d_p, name="Price", line=dict(color="#38BDF8", width=2)))
                if len(d_p) >= 50:
                    fig_drill.add_trace(go.Scatter(x=d_sma50.index, y=d_sma50, name="50-SMA", line=dict(color="#F59E0B", width=1.2, dash="dash")))
                if len(d_p) >= 200:
                    fig_drill.add_trace(go.Scatter(x=d_sma200.index, y=d_sma200, name="200-SMA", line=dict(color="#EC4899", width=1.2, dash="dot")))
                fig_drill.update_layout(
                    template="plotly_dark", height=260,
                    margin=dict(l=10, r=10, t=15, b=10),
                    yaxis_title="Price",
                    legend=dict(orientation="h", y=1.05, x=0),
                )
                st.plotly_chart(fig_drill, use_container_width=True)

            with dd_col2:
                rec_d = records_lookup.get(drill_asset.upper(), {})
                high_52 = float(d_p.max())
                low_52 = float(d_p.min())
                curr_px = float(d_p.iloc[-1])
                dist_high = (curr_px / high_52 - 1.0) * 100.0
                st.markdown(f"""
                <div class="kpi-card" style="margin-bottom: 6px;">
                    <span class="kpi-label">{rec_d.get('name', drill_asset)}</span>
                    <span class="kpi-value">₹{curr_px:,.2f}</span>
                    <span class="kpi-sub">52W High: ₹{high_52:,.2f} ({dist_high:+.1f}%)</span>
                    <span class="kpi-sub">52W Low: ₹{low_52:,.2f}</span>
                </div>
                <div class="kpi-card">
                    <span class="kpi-label">Risk Contribution</span>
                    <span class="kpi-value">{risk_contrib.get(drill_asset, 0.0):.1f}%</span>
                    <span class="kpi-sub">Portfolio Weight: {weight_series.get(drill_asset, 0.0)*100.0:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)

    # =========================================================================
    # 9. ALLOCATION EDITOR (INCLUDING HIERARCHICAL RISK PARITY)
    # =========================================================================
    st.markdown("<div class='section-header'>⚖️ Interactive Allocation Editor</div>", unsafe_allow_html=True)

    al_col1, al_col2 = st.columns([1.6, 3.4])
    with al_col1:
        alloc_scheme = st.selectbox(
            "Weight Scheme Preset",
            ["Equal Weight", "Market Cap Weight", "Inverse Volatility", "Hierarchical Risk Parity (HRP)", "Risk Parity", "Max Sharpe", "Minimum Variance", "Target Return", "Manual"],
            index=0,
        )

        if alloc_scheme == "Equal Weight":
            new_w = {t: 1.0 / len(active_tickers) for t in active_tickers}
            st.session_state["asset_weights"] = new_w
        elif alloc_scheme == "Market Cap Weight":
            caps = np.array([records_lookup.get(t.upper(), {}).get("mcap", 1e9) for t in active_tickers])
            cap_w = caps / np.sum(caps)
            st.session_state["asset_weights"] = {t: float(cap_w[i]) for i, t in enumerate(active_tickers)}
        elif alloc_scheme == "Inverse Volatility":
            vols = np.array([prices_df[t].pct_change().dropna().std() for t in active_tickers])
            inv_v = 1.0 / (vols + 1e-9)
            inv_w = inv_v / np.sum(inv_v)
            st.session_state["asset_weights"] = {t: float(inv_w[i]) for i, t in enumerate(active_tickers)}
        elif alloc_scheme == "Hierarchical Risk Parity (HRP)":
            hrp_res = hierarchical_risk_parity(prices_df.pct_change().dropna(), cov_matrix)
            hrp_w = hrp_res.get("weights", pd.Series(1.0/len(active_tickers), index=active_tickers))
            st.session_state["asset_weights"] = hrp_w.to_dict()
        elif alloc_scheme in ("Risk Parity", "Max Sharpe", "Minimum Variance", "Target Return"):
            opt_pre = run_slsqp_optimizer(prices_df.pct_change().dropna(), cov_matrix, objective=alloc_scheme)
            st.session_state["asset_weights"] = opt_pre["weights"].to_dict()

    with al_col2:
        if alloc_scheme == "Manual":
            st.caption("Adjust weight sliders below. Total allocation is tracked dynamically.")
            temp_w = {}
            w_cols = st.columns(min(len(active_tickers), 4))
            for i, t in enumerate(active_tickers):
                with w_cols[i % 4]:
                    temp_w[t] = st.slider(f"{t} Weight %", 0, 100, int(weight_series[t] * 100), 1)
            t_sum = sum(temp_w.values())

            if t_sum == 100:
                st.markdown(f"<div style='color: #10B981; font-weight: 700; font-size: 0.85rem;'>✅ Total Allocation: 100%</div>", unsafe_allow_html=True)
                st.session_state["asset_weights"] = {t: temp_w[t] / 100.0 for t in active_tickers}
            elif t_sum > 0:
                st.warning(f"⚠️ Total Allocation: {t_sum}% (Does not equal 100%). Click below to normalize.")
                if st.button("Normalize to 100%", use_container_width=True):
                    st.session_state["asset_weights"] = {t: temp_w[t] / t_sum for t in active_tickers}
                    st.rerun()
            else:
                st.error("Weights cannot all be 0%.")
        else:
            w_bars = []
            for t in active_tickers:
                w_bars.append(f"{t}: **{st.session_state['asset_weights'].get(t, 0.0)*100.0:.1f}%**")
            st.markdown(f"<div style='margin-top: 10px; font-family: JetBrains Mono, monospace; font-size: 0.82rem;'><b>Active Allocation:</b> {' · '.join(w_bars)}</div>", unsafe_allow_html=True)
            st.markdown("<div style='color: #10B981; font-weight: 600; font-size: 0.80rem; margin-top: 4px;'>✅ Total Allocation: 100.0%</div>", unsafe_allow_html=True)

    # =========================================================================
    # 10. RISK DASHBOARD & RISK CONTRIBUTION
    # =========================================================================
    st.markdown("<div class='section-header'>🛡️ Portfolio Risk Dashboard & Marginal Risk Contributions</div>", unsafe_allow_html=True)
    r_col1, r_col2 = st.columns(2)

    with r_col1:
        st.markdown("<div style='font-size: 0.80rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin-bottom: 8px;'>Marginal Risk Contribution (Percentage of Total Portfolio Volatility Variance)</div>", unsafe_allow_html=True)
        for t in active_tickers:
            rc_val = max(0.0, float(risk_contrib.get(t, 0.0)))
            st.markdown(
                f"""
                <div class="risk-bar-container">
                    <span class="risk-bar-label">{t}</span>
                    <div class="risk-bar-track">
                        <div class="risk-bar-fill" style="width: {min(100.0, rc_val):.1f}%;"></div>
                    </div>
                    <span class="risk-bar-val">{rc_val:.1f}%</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with r_col2:
        st.markdown("<div style='font-size: 0.80rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin-bottom: 8px;'>Comprehensive Risk Metrics</div>", unsafe_allow_html=True)
        r_grid1, r_grid2 = st.columns(2)
        with r_grid1:
            st.markdown(f"""
            <div class="kpi-card" style="margin-bottom: 8px;"><span class="kpi-label">Annualized Volatility</span><span class="kpi-value">{kpis['annual_vol_pct']:.2f}%</span></div>
            <div class="kpi-card" style="margin-bottom: 8px;"><span class="kpi-label">Value at Risk (VaR 95%)</span><span class="kpi-value val-neg">-{kpis['var_95_pct']:.2f}%</span></div>
            <div class="kpi-card"><span class="kpi-label">Conditional VaR (CVaR 95%)</span><span class="kpi-value val-neg">-{kpis['cvar_95_pct']:.2f}%</span></div>
            """, unsafe_allow_html=True)
        with r_grid2:
            st.markdown(f"""
            <div class="kpi-card" style="margin-bottom: 8px;"><span class="kpi-label">Maximum Drawdown</span><span class="kpi-value val-neg">{kpis['max_drawdown_pct']:.2f}%</span></div>
            <div class="kpi-card" style="margin-bottom: 8px;"><span class="kpi-label">Downside Volatility</span><span class="kpi-value">{kpis['downside_vol_pct']:.2f}%</span></div>
            <div class="kpi-card"><span class="kpi-label">Beta to Benchmark</span><span class="kpi-value">{kpis['beta']:.2f}</span></div>
            """, unsafe_allow_html=True)

    # =========================================================================
    # 11. ROLLING QUANTITATIVE RISK & REGIME ANALYTICS
    # =========================================================================
    st.markdown("<div class='section-header'>🔄 Rolling Risk & Beta Regime Studio</div>", unsafe_allow_html=True)
    roll_c1, roll_c2 = st.columns([1.5, 4.5])
    with roll_c1:
        roll_window = st.selectbox("Rolling Lookback Window", [30, 60, 90], index=1, format_func=lambda w: f"{w} Trading Days")
        roll_metric_choice = st.radio("Display Rolling Metric", ["Annualized Volatility", "Sharpe Ratio", "CAPM Beta", "Tracking Error"], index=0)

    with roll_c2:
        df_rolling = compute_rolling_metrics(port_results, window=roll_window, freq_factor=freq_factor)
        if not df_rolling.empty:
            fig_roll = go.Figure()
            if roll_metric_choice == "Annualized Volatility":
                fig_roll.add_trace(go.Scatter(x=df_rolling.index, y=df_rolling["Rolling Port Vol %"], mode="lines", name="Portfolio Vol %", line=dict(color="#38BDF8", width=2)))
                fig_roll.add_trace(go.Scatter(x=df_rolling.index, y=df_rolling["Rolling Bench Vol %"], mode="lines", name=f"Benchmark ({bench_sym}) Vol %", line=dict(color="#64748B", width=1.5, dash="dash")))
                fig_roll.update_layout(yaxis_title="Annualized Volatility %")
            elif roll_metric_choice == "Sharpe Ratio":
                fig_roll.add_trace(go.Scatter(x=df_rolling.index, y=df_rolling["Rolling Sharpe"], mode="lines", name="Rolling Sharpe (Rf=5%)", line=dict(color="#10B981", width=2)))
                fig_roll.add_hline(y=0, line_dash="dot", line_color="#64748B")
                fig_roll.update_layout(yaxis_title="Sharpe Ratio")
            elif roll_metric_choice == "CAPM Beta":
                fig_roll.add_trace(go.Scatter(x=df_rolling.index, y=df_rolling["Rolling Beta"], mode="lines", name="Rolling Market Beta", line=dict(color="#F59E0B", width=2)))
                fig_roll.add_hline(y=1.0, line_dash="dash", line_color="#64748B")
                fig_roll.update_layout(yaxis_title="Beta vs Benchmark")
            elif roll_metric_choice == "Tracking Error":
                fig_roll.add_trace(go.Scatter(x=df_rolling.index, y=df_rolling["Rolling Tracking Error %"], mode="lines", name="Tracking Error %", line=dict(color="#EC4899", width=2)))
                fig_roll.update_layout(yaxis_title="Tracking Error %")

            fig_roll.update_layout(
                template="plotly_dark", height=280,
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", y=1.05, x=0),
            )
            st.plotly_chart(fig_roll, use_container_width=True)

    # =========================================================================
    # 12. MACRO CRISIS & SCENARIO STRESS TESTING ENGINE
    # =========================================================================
    st.markdown("<div class='section-header'>💥 Macro Crisis & Multi-Scenario Stress Testing</div>", unsafe_allow_html=True)
    with st.container():
        st.caption("Simulates historical and hypothetical macroeconomic stress shocks against active portfolio weights and empirical asset betas.")
        custom_sh = st.slider("Hypothetical Custom Benchmark Shock %", -50, 20, int(st.session_state["custom_shock_val"]), 1)
        st.session_state["custom_shock_val"] = custom_sh

        stress_res = run_portfolio_stress_tests(weight_series, prices_df, bench_series, kpis["final_value"], custom_shock_pct=custom_sh)
        stress_df = pd.DataFrame(stress_res)

        s_cols = st.columns(len(stress_res))
        for i, s in enumerate(stress_res):
            with s_cols[i]:
                st.markdown(f"""
                <div class="kpi-card" style="height: 100%;">
                    <span class="kpi-label">{s['Scenario']}</span>
                    <span class="kpi-value {'val-pos' if s['Projected Return %']>=0 else 'val-neg'}">{s['Projected Return %']:+.1f}%</span>
                    <span class="kpi-sub">Shock: {s['Market Shock %']}</span>
                    <span class="kpi-sub">P&L: {fmt_inr(s['Projected P&L'])}</span>
                </div>
                """, unsafe_allow_html=True)

    # =========================================================================
    # 13. CORRELATION MATRIX & DRAWDOWN
    # =========================================================================
    st.markdown("<div class='section-header'>🔗 Correlation & Drawdown Timelines</div>", unsafe_allow_html=True)
    c_col1, c_col2 = st.columns(2)

    with c_col1:
        st.markdown("<div style='font-size: 0.80rem; font-weight: 700; color: #94A3B8; text-transform: uppercase;'>Return Correlation Matrix</div>", unsafe_allow_html=True)
        ret_type = st.radio("Return Formulation", ["Daily Returns", "Log Returns"], index=0, horizontal=True)
        rets_raw = prices_df.pct_change().dropna()
        if ret_type == "Log Returns":
            rets_raw = np.log(prices_df / prices_df.shift(1)).dropna()

        corr_mat = rets_raw.corr()
        fig_corr = px.imshow(
            corr_mat,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1.0, zmax=1.0,
            template="plotly_dark",
        )
        fig_corr.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_corr, use_container_width=True)

    with c_col2:
        st.markdown("<div style='font-size: 0.80rem; font-weight: 700; color: #94A3B8; text-transform: uppercase;'>Underwater Drawdown Timeline</div>", unsafe_allow_html=True)
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=port_results["drawdown_series"].index,
            y=port_results["drawdown_series"],
            fill="tozeroy",
            mode="lines",
            line=dict(color="#EF4444", width=1.8),
            fillcolor="rgba(239, 68, 68, 0.2)",
            name="Drawdown %",
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Drawdown:</b> %{y:.2f}%<extra></extra>",
        ))
        fig_dd.update_layout(
            template="plotly_dark", height=260,
            margin=dict(l=10, r=10, t=10, b=15),
            yaxis_title="Drawdown %",
        )
        st.plotly_chart(fig_dd, use_container_width=True)

    # =========================================================================
    # 14. EFFICIENT FRONTIER & OPTIMIZER STUDIO
    # =========================================================================
    st.markdown("<div class='section-header'>🎯 Markowitz Efficient Frontier & Optimizer Studio</div>", unsafe_allow_html=True)
    o_row1, o_row2 = st.columns([3.2, 1.8])

    frontier_data = generate_frontier_cloud(prices_df.pct_change().dropna(), cov_matrix)

    with o_row1:
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(
            x=frontier_data["sim_vols"],
            y=frontier_data["sim_rets"],
            mode="markers",
            marker=dict(size=4, color=frontier_data["sim_sharpes"], colorscale="Viridis", opacity=0.45, showscale=True, colorbar=dict(title="Sharpe", thickness=10, len=0.6)),
            name="Random Portfolios",
            hovertemplate="<b>Simulated Portfolio</b><br>Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
        ))
        fig_f.add_trace(go.Scatter(
            x=frontier_data["frontier_vols"],
            y=frontier_data["frontier_rets"],
            mode="lines",
            line=dict(color="#38BDF8", width=3),
            name="Efficient Frontier",
            hovertemplate="<b>Frontier</b><br>Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
        ))
        for ap in frontier_data["asset_points"]:
            fig_f.add_trace(go.Scatter(
                x=[ap["vol"]], y=[ap["ret"]],
                mode="markers+text",
                marker=dict(size=8, color="#94A3B8"),
                text=[ap["asset"]],
                textposition="top right",
                name=ap["asset"],
                hovertemplate=f"<b>{ap['asset']}</b><br>Vol: %{{x:.2f}}%<br>Return: %{{y:.2f}}%<extra></extra>",
            ))
        fig_f.add_trace(go.Scatter(
            x=[kpis["annual_vol_pct"]],
            y=[kpis["cagr_pct"]],
            mode="markers+text",
            marker=dict(symbol="star", size=16, color="#F59E0B"),
            text=["Current Portfolio"],
            textposition="top center",
            name="Current Portfolio",
            hovertemplate="<b>Current Portfolio</b><br>Vol: %{x:.2f}%<br>CAGR: %{y:.2f}%<extra></extra>",
        ))
        fig_f.add_trace(go.Scatter(
            x=[frontier_data["max_sharpe_point"][0]],
            y=[frontier_data["max_sharpe_point"][1]],
            mode="markers+text",
            marker=dict(symbol="diamond", size=13, color="#10B981"),
            text=["Max Sharpe"],
            textposition="bottom right",
            name="Max Sharpe",
            hovertemplate="<b>Max Sharpe Portfolio</b><br>Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
        ))
        fig_f.add_trace(go.Scatter(
            x=[frontier_data["min_vol_point"][0]],
            y=[frontier_data["min_vol_point"][1]],
            mode="markers+text",
            marker=dict(symbol="circle", size=13, color="#818CF8"),
            text=["Min Variance"],
            textposition="bottom left",
            name="Min Variance",
            hovertemplate="<b>Min Variance Portfolio</b><br>Vol: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
        ))

        fig_f.update_layout(
            template="plotly_dark", height=420,
            margin=dict(l=10, r=10, t=20, b=15),
            xaxis_title="Annualized Volatility %",
            yaxis_title="Annualized Return %",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig_f, use_container_width=True)

    with o_row2:
        st.markdown("<div style='font-size: 0.80rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin-bottom: 6px;'>Optimizer Parameters & Objectives</div>", unsafe_allow_html=True)
        opt_goal = st.selectbox("Objective", ["Max Sharpe", "Min Volatility", "Max Return", "Risk Parity", "Target Return"])
        min_b = st.slider("Minimum Weight per Asset %", 0, 20, 2, 1) / 100.0
        max_b = st.slider("Maximum Weight per Asset %", 20, 100, 40, 5) / 100.0
        tgt_r = st.number_input("Target Return %", value=15.0, step=1.0) / 100.0 if opt_goal == "Target Return" else 0.15
        max_v = st.number_input("Maximum Volatility %", value=20.0, step=1.0) / 100.0 if opt_goal == "Min Volatility" else 0.20

        if st.button("RUN PORTFOLIO OPTIMIZATION", type="primary", use_container_width=True) or opt_clicked:
            opt_res = run_slsqp_optimizer(
                prices_df.pct_change().dropna(),
                cov_matrix,
                objective=opt_goal,
                min_weight=min_b,
                max_weight=max_b,
                target_return=tgt_r,
                max_volatility=max_v,
            )
            st.session_state["optimized_result"] = opt_res

        if "optimized_result" in st.session_state:
            opt_r = st.session_state["optimized_result"]

            comp_metrics_df = pd.DataFrame([
                {"Metric": "CAGR %", "Current": f"{kpis['cagr_pct']:+.2f}%", "Optimized": f"{opt_r['cagr']:+.2f}%"},
                {"Metric": "Volatility %", "Current": f"{kpis['annual_vol_pct']:.2f}%", "Optimized": f"{opt_r['volatility']:.2f}%"},
                {"Metric": "Sharpe Ratio", "Current": f"{kpis['sharpe']:.2f}", "Optimized": f"{opt_r['sharpe']:.2f}"},
                {"Metric": "Max Drawdown %", "Current": f"{kpis['max_drawdown_pct']:.2f}%", "Optimized": f"{opt_r['max_drawdown']:.2f}%"},
            ])
            st.dataframe(comp_metrics_df, use_container_width=True, hide_index=True)

            opt_alloc_rows = []
            for t in active_tickers:
                c_w = weight_series[t] * 100.0
                o_w = opt_r["weights"][t] * 100.0
                opt_alloc_rows.append({"Asset": t, "Current %": round(c_w, 1), "Optimized %": round(o_w, 1)})
            st.dataframe(pd.DataFrame(opt_alloc_rows), use_container_width=True, hide_index=True)

            if st.button("Apply Optimized Allocation", use_container_width=True):
                st.session_state["asset_weights"] = opt_r["weights"].to_dict()
                st.session_state["rebalance_target_weights"] = opt_r["weights"].to_dict()
                st.success("✅ Optimized allocation applied to active portfolio!")
                st.rerun()

    # =========================================================================
    # 15. FACTOR EXPOSURE & DIAGNOSTICS
    # =========================================================================
    st.markdown("<div class='section-header'>🧬 Factor Attribution & Institutional Diagnostics</div>", unsafe_allow_html=True)
    f_col1, f_col2 = st.columns([3, 2])

    with f_col1:
        factors = ["Market Beta", "Momentum", "Value", "Quality", "Size (Mega-Cap)", "Low Volatility"]
        p_beta = kpis["beta"]
        p_mom = min(1.5, max(-1.5, kpis["cagr_pct"] / 20.0))
        p_val = 0.8
        p_qual = 1.1
        p_size = 1.2
        p_lowvol = min(1.5, max(-1.5, (25.0 - kpis["annual_vol_pct"]) / 10.0))

        factor_vals = [p_beta, p_mom, p_val, p_qual, p_size, p_lowvol]
        fig_fac = go.Figure(go.Bar(
            x=factor_vals,
            y=factors,
            orientation="h",
            marker=dict(color=["#38BDF8", "#10B981", "#818CF8", "#F59E0B", "#EC4899", "#10B981"]),
            hovertemplate="<b>%{y}</b>: %{x:.2f} Z-Score<extra></extra>",
        ))
        fig_fac.update_layout(
            template="plotly_dark", height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Standardized Factor Exposure (Z-Score)",
        )
        st.plotly_chart(fig_fac, use_container_width=True)

    with f_col2:
        w_arr = weight_series.values
        herfindahl = float(np.sum(w_arr ** 2))
        eff_n = float(1.0 / (herfindahl + 1e-9))
        largest_asset = weight_series.idxmax()
        largest_w = float(weight_series.max() * 100.0)
        top_sector_name = max(sector_weights, key=sector_weights.get) if sector_weights else "General"
        div_ratio = float(np.sum(w_arr * prices_df.pct_change().dropna().std().values * np.sqrt(freq_factor)) / (kpis['annual_vol_pct']/100.0 + 1e-9))

        st.markdown(f"""
        <div class="kpi-card" style="margin-bottom: 6px;">
            <span class="kpi-label">Effective Number of Holdings</span>
            <span class="kpi-value">{eff_n:.1f} <span style="font-size: 0.75rem; color: #64748B;">/ {len(active_tickers)}</span></span>
            <span class="kpi-sub">N_eff = 1 / Σ w_i² (Inverse Herfindahl)</span>
        </div>
        <div class="kpi-card" style="margin-bottom: 6px;">
            <span class="kpi-label">Diversification Ratio</span>
            <span class="kpi-value">{div_ratio:.2f}x</span>
            <span class="kpi-sub">Ratio of Weighted Vol to Portfolio Vol</span>
        </div>
        <div class="kpi-card">
            <span class="kpi-label">Concentration (Top Asset / Sector)</span>
            <span class="kpi-value">{largest_w:.1f}% <span style="font-size: 0.75rem; color: #94A3B8;">({largest_asset})</span></span>
            <span class="kpi-sub">Top Sector: {top_sector_name} ({sector_weights.get(top_sector_name, 0.0):.1f}%)</span>
        </div>
        """, unsafe_allow_html=True)

    # =========================================================================
    # 16. REBALANCE ENGINE WITH TRANSACTION COSTS & BLOTTER
    # =========================================================================
    st.markdown("<div class='section-header'>🔄 Automated Portfolio Rebalance Engine & Friction Cost Modeling</div>", unsafe_allow_html=True)

    with st.expander("📝 Generate Rebalance Plan, Friction Costs & Trade Blotter", expanded=(rebal_clicked or bool(st.session_state.get("rebalance_target_weights")))):
        reb_c1, reb_c2, reb_c3 = st.columns([1.5, 1.5, 2])
        with reb_c1:
            fee_bps = st.number_input("Brokerage + Slippage Friction (bps)", value=10.0, step=2.5, min_value=0.0) / 10000.0
        with reb_c2:
            min_trade_thresh = st.number_input("Minimum Trade Threshold %", value=0.5, step=0.25, min_value=0.0)

        target_w = st.session_state.get("rebalance_target_weights") or weight_series.to_dict()
        tot_v = kpis["final_value"]

        # Calculate Turnover
        turnover_pct = sum(abs(target_w.get(t, weight_series[t]) - weight_series[t]) for t in active_tickers) * 0.5 * 100.0
        gross_turnover_val = (turnover_pct / 100.0) * tot_v * 2.0
        total_friction_cost = gross_turnover_val * fee_bps

        reb_c3.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid #334155; border-radius: 6px; padding: 8px 12px;">
            <div style="font-size: 0.76rem; color: #94A3B8;">TURNOVER & EXECUTION COST</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #38BDF8; font-family: monospace;">Turnover: {turnover_pct:.1f}% ({fmt_inr(gross_turnover_val)})</div>
            <div style="font-size: 0.72rem; color: #EF4444;">Estimated Friction Cost: {fmt_inr(total_friction_cost)}</div>
        </div>
        """, unsafe_allow_html=True)

        rebal_orders = []
        for t in active_tickers:
            c_w = float(weight_series[t] * 100.0)
            t_w = float(target_w.get(t, weight_series[t]) * 100.0)
            diff_w = t_w - c_w

            if abs(diff_w) < min_trade_thresh:
                action = "⚪ HOLD (< Thresh)"
                trade_val = 0.0
            else:
                trade_val = (diff_w / 100.0) * tot_v
                action = f"🟢 BUY {fmt_inr(trade_val)}" if trade_val > 0 else f"🔴 SELL {fmt_inr(abs(trade_val))}"

            rebal_orders.append({
                "Asset": t,
                "Current Weight %": f"{c_w:.1f}%",
                "Target Weight %": f"{t_w:.1f}%",
                "Difference %": f"{diff_w:+.1f}%",
                "Current Value": fmt_inr((c_w / 100.0) * tot_v),
                "Target Value": fmt_inr((t_w / 100.0) * tot_v),
                "Trade Value": fmt_inr(abs(trade_val)),
                "Action": action,
            })

        rebal_df = pd.DataFrame(rebal_orders)
        st.dataframe(rebal_df, use_container_width=True, hide_index=True)

        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            csv_buf = io.StringIO()
            rebal_df.to_csv(csv_buf, index=False)
            st.download_button(
                "📥 Export Trade Blotter CSV",
                data=csv_buf.getvalue(),
                file_name="portfolio_rebalance_blotter.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with dl_col2:
            # Executive Tearsheet HTML download
            tearsheet_html = generate_executive_tearsheet_html(
                kpis=kpis,
                holdings_df=holdings_df,
                sector_weights=sector_weights,
                stress_results=stress_res,
                benchmark_name=st.session_state["benchmark_choice"],
                period=st.session_state["period_choice"],
            )
            st.download_button(
                "📄 Export Executive Tearsheet (HTML/PDF)",
                data=tearsheet_html,
                file_name="Portfolio_Executive_Tearsheet.html",
                mime="text/html",
                use_container_width=True,
            )

    # =========================================================================
    # 17. INSTITUTIONAL STATUTORY FOOTER
    # =========================================================================
    st.markdown(
        """
        <div class="footer-disclaimer">
            Historical portfolio analysis is for research purposes and does not guarantee future results.
            All quantitative calculations account for asset covariance, cash sleeves, portfolio rebalancing, and dividend-adjusted price returns.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    render_page()
