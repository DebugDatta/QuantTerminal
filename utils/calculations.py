"""Quantitative calculations for performance, risk, and technical indicators."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from utils.validation import safe_float, safe_division


def calculate_period_returns(close: pd.Series) -> Dict[str, Optional[float]]:
    """Calculate multi-horizon returns and annualized CAGRs from a Close price series."""
    if close is None or len(close) < 2:
        return {
            "1D": None, "1W": None, "1M": None, "3M": None,
            "6M": None, "YTD": None, "1Y": None, "3Y_CAGR": None, "5Y_CAGR": None,
        }

    clean = close.dropna()
    n = len(clean)
    latest = float(clean.iloc[-1])

    def _ret_lookback(bars: int) -> Optional[float]:
        if n > bars:
            prior = float(clean.iloc[-(bars + 1)])
            return ((latest / prior) - 1.0) * 100.0 if prior > 0 else None
        elif bars >= 240 and n >= 240:
            prior = float(clean.iloc[0])
            return ((latest / prior) - 1.0) * 100.0 if prior > 0 else None
        return None

    ret_1d = _ret_lookback(1)
    ret_1w = _ret_lookback(5)
    ret_1m = _ret_lookback(21)
    ret_3m = _ret_lookback(63)
    ret_6m = _ret_lookback(126)
    ret_1y = _ret_lookback(252)

    # YTD Return
    try:
        current_year = clean.index[-1].year
        ytd_sub = clean[clean.index.year == current_year]
        if len(ytd_sub) >= 2:
            ytd_base = float(ytd_sub.iloc[0])
            ret_ytd = ((latest / ytd_base) - 1.0) * 100.0 if ytd_base > 0 else None
        else:
            ret_ytd = None
    except Exception:
        ret_ytd = None

    # CAGR: (End / Start) ^ (365.25 / days) - 1
    def _cagr_years(target_years: float, min_bars: int) -> Optional[float]:
        if n >= min_bars:
            start_price = float(clean.iloc[-min_bars])
            if start_price > 0:
                elapsed_days = (clean.index[-1] - clean.index[-min_bars]).days
                if elapsed_days >= 300 * target_years:
                    c = ((latest / start_price) ** (365.25 / elapsed_days) - 1.0) * 100.0
                    return c
        return None

    cagr_3y = _cagr_years(3.0, 252 * 3)
    cagr_5y = _cagr_years(5.0, 252 * 5)

    return {
        "1D": ret_1d,
        "1W": ret_1w,
        "1M": ret_1m,
        "3M": ret_3m,
        "6M": ret_6m,
        "YTD": ret_ytd,
        "1Y": ret_1y,
        "3Y_CAGR": cagr_3y,
        "5Y_CAGR": cagr_5y,
    }


def calculate_risk_metrics(
    close: pd.Series,
    benchmark_close: Optional[pd.Series] = None,
    rf_annual: float = 0.05,
) -> Dict[str, Any]:
    """Calculate reproducible risk, tail, and drawdown metrics from Close prices."""
    if close is None or len(close) < 10:
        return {
            "volatility": None, "sharpe": None, "sortino": None,
            "max_drawdown": None, "current_drawdown": None,
            "var_95": None, "cvar_95": None, "downside_vol": None, "beta": None,
        }

    daily_returns = close.pct_change().dropna()
    if daily_returns.empty:
        return {}

    n = len(daily_returns)
    trading_days = 252

    # Annualized Volatility
    vol = float(daily_returns.std(ddof=1) * math.sqrt(trading_days) * 100.0)

    # Annualized Arithmetic Return
    ann_ret = float(daily_returns.mean() * trading_days * 100.0)

    # Sharpe Ratio: (Annualized Return - Risk Free Rate) / Annualized Vol
    excess_ret = ann_ret - (rf_annual * 100.0)
    sharpe = float(excess_ret / vol) if vol > 0 else None

    # Downside Deviation & Sortino
    neg_rets = daily_returns[daily_returns < 0]
    if len(neg_rets) > 0:
        downside_std = float(neg_rets.std(ddof=1) * math.sqrt(trading_days) * 100.0)
        sortino = float(excess_ret / downside_std) if downside_std > 0 else None
    else:
        downside_std = 0.0
        sortino = None

    # Maximum and Current Drawdown
    cummax = close.cummax()
    dd_series = (close - cummax) / cummax
    max_dd = float(dd_series.min() * 100.0)
    curr_dd = float(dd_series.iloc[-1] * 100.0)

    # Historical VaR 95% and CVaR 95% (Daily)
    var_95 = float(np.percentile(daily_returns, 5) * 100.0)
    tail_losses = daily_returns[daily_returns <= np.percentile(daily_returns, 5)]
    cvar_95 = float(tail_losses.mean() * 100.0) if not tail_losses.empty else var_95

    # Beta vs Benchmark
    beta = None
    if benchmark_close is not None and len(benchmark_close) > 10:
        b_rets = benchmark_close.pct_change().dropna()
        aligned = pd.concat([daily_returns, b_rets], axis=1, join="inner").dropna()
        if len(aligned) >= 15:
            cov = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
            var_b = aligned.iloc[:, 1].var()
            if var_b > 0:
                beta = float(cov / var_b)

    return {
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "current_drawdown": curr_dd,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "downside_vol": downside_std,
        "beta": beta,
        "rf_annual": rf_annual,
    }


def calculate_technical_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate moving averages, momentum, volatility, and volume indicators from OHLCV."""
    if df is None or len(df) < 5 or "Close" not in df.columns:
        return {}

    close = df["Close"]
    high = df["High"] if "High" in df.columns else close
    low = df["Low"] if "Low" in df.columns else close
    volume = df["Volume"] if "Volume" in df.columns else pd.Series(1, index=close.index)

    last_price = float(close.iloc[-1])
    n = len(close)

    # Moving Averages
    sma20 = float(close.rolling(20).mean().iloc[-1]) if n >= 20 else None
    sma50 = float(close.rolling(50).mean().iloc[-1]) if n >= 50 else None
    sma200 = float(close.rolling(200).mean().iloc[-1]) if n >= 200 else None
    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1]) if n >= 20 else None
    ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1]) if n >= 50 else None

    # VWAP (if volume present)
    if (volume > 0).any():
        cum_pv = (close * volume).cumsum()
        cum_v = volume.cumsum()
        vwap_series = cum_pv / cum_v.replace(0, np.nan)
        vwap = float(vwap_series.iloc[-1]) if not math.isnan(vwap_series.iloc[-1]) else None
    else:
        vwap = None

    # Bollinger Bands (20, 2)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std(ddof=0)
    bb_up = bb_mid + 2 * bb_std
    bb_low = bb_mid - 2 * bb_std
    bb_width_pct = None
    if n >= 20 and bb_mid.iloc[-1] > 0:
        bb_width_pct = float((bb_up.iloc[-1] - bb_low.iloc[-1]) / bb_mid.iloc[-1] * 100.0)

    # RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    rsi = float(rsi_series.iloc[-1]) if n >= 15 and not math.isnan(rsi_series.iloc[-1]) else None

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_val = float(macd_line.iloc[-1]) if n >= 26 else None
    signal_val = float(signal_line.iloc[-1]) if n >= 35 else None
    macd_hist = float(macd_line.iloc[-1] - signal_line.iloc[-1]) if (macd_val and signal_val) else None

    # ATR (14)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1]) if n >= 15 else None
    atr_pct = float(atr / last_price * 100.0) if (atr and last_price > 0) else None

    # Stochastic (14, 3)
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    denom = (high14 - low14).replace(0, np.nan)
    stoch_k = ((close - low14) / denom * 100.0).rolling(3).mean()
    stoch = float(stoch_k.iloc[-1]) if n >= 16 and not math.isnan(stoch_k.iloc[-1]) else None

    # Relative Volume (10D)
    avg_vol_10 = float(volume.rolling(10).mean().iloc[-1]) if n >= 10 else None
    last_vol = float(volume.iloc[-1]) if n >= 1 else None
    rvol = float(last_vol / avg_vol_10) if (last_vol and avg_vol_10 and avg_vol_10 > 0) else None

    # Regimes / Statuses
    trend_state = "NEUTRAL"
    if sma50 and sma200:
        if last_price > sma50 and sma50 > sma200:
            trend_state = "BULLISH"
        elif last_price < sma50 and sma50 < sma200:
            trend_state = "BEARISH"
    elif sma50:
        trend_state = "BULLISH" if last_price > sma50 else "BEARISH"

    mom_state = "NEUTRAL"
    if rsi:
        if rsi >= 70:
            mom_state = "OVERBOUGHT"
        elif rsi <= 30:
            mom_state = "OVERSOLD"

    vol_state = "NORMAL"
    if atr_pct:
        if atr_pct > 3.5:
            vol_state = "HIGH"
        elif atr_pct < 1.2:
            vol_state = "LOW"

    volume_state = "AVERAGE"
    if rvol:
        if rvol > 1.3:
            volume_state = "EXPANDING"
        elif rvol < 0.7:
            volume_state = "CONTRACTING"

    return {
        "last_price": last_price,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "ema20": ema20,
        "ema50": ema50,
        "vwap": vwap,
        "bb_width_pct": bb_width_pct,
        "rsi": rsi,
        "macd": macd_val,
        "macd_signal": signal_val,
        "macd_hist": macd_hist,
        "atr": atr,
        "atr_pct": atr_pct,
        "stoch": stoch,
        "avg_vol_10": avg_vol_10,
        "rvol": rvol,
        "trend_state": trend_state,
        "momentum_state": mom_state,
        "volatility_state": vol_state,
        "volume_state": volume_state,
    }


def _get_chronological_years(df: pd.DataFrame, metric_check: str = "Sales|Total Revenue|Operating Revenue") -> List[str]:
    raw_cols = [c for c in df.columns if c not in ("Metric", "TTM")]
    if not raw_cols:
        return []

    def _extract_year(s):
        import re
        m = re.search(r"\b(19\d\d|20\d\d)\b", str(s))
        return int(m.group(1)) if m else 0

    parsed = [_extract_year(y) for y in raw_cols]
    if len(parsed) >= 2 and parsed[0] > parsed[-1]:
        raw_cols = raw_cols[::-1]

    valid_cols = []
    row = df[df["Metric"].str.contains(metric_check, case=False, na=False, regex=True)]
    for c in raw_cols:
        if not row.empty and c in row.columns and safe_float(row[c].iloc[0]) is not None:
            valid_cols.append(c)
        elif row.empty:
            valid_cols.append(c)

    return valid_cols if valid_cols else raw_cols


# =============================================================================
# DUPONT ANALYSIS (3-STAGE & 5-STAGE)
# =============================================================================
def calculate_dupont_analysis(
    pl_df: Optional[pd.DataFrame],
    bs_df: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    """Calculate 3-stage and 5-stage DuPont ROE decomposition across historical fiscal years."""
    if pl_df is None or pl_df.empty or bs_df is None or bs_df.empty:
        return {"table_3stage": pd.DataFrame(), "table_5stage": pd.DataFrame(), "driver": "Insufficient data", "latest_roe": "N/A"}

    years = _get_chronological_years(pl_df)[-5:]
    if not years:
        return {"table_3stage": pd.DataFrame(), "table_5stage": pd.DataFrame(), "driver": "Insufficient periods", "latest_roe": "N/A"}

    def _val(df: pd.DataFrame, metric_regex: str, yr: str) -> Optional[float]:
        row = df[df["Metric"].str.contains(metric_regex, case=False, na=False, regex=True)]
        if not row.empty and yr in row.columns:
            return safe_float(row[yr].iloc[0])
        return None

    rows_3s = []
    rows_5s = []

    for yr in years:
        sales = _val(pl_df, "Sales|Total Revenue|Operating Revenue", yr)
        ebit = _val(pl_df, "Operating Profit|Operating Income|EBIT", yr)
        pbt = _val(pl_df, "Profit before tax|Pretax Income", yr)
        pat = _val(pl_df, "Net Profit|Net Income", yr)

        tot_assets = _val(bs_df, "Total Assets", yr)
        equity = _val(bs_df, "Stockholders Equity|Common Stock Equity", yr)
        if equity is None:
            eq_cap = _val(bs_df, "Equity Capital|Common Stock", yr) or 0.0
            reserves = _val(bs_df, "Reserves|Retained Earnings", yr) or 0.0
            equity = (eq_cap + reserves) if (eq_cap or reserves) else None

        # 3-Stage components
        npm = (pat / sales * 100.0) if (pat and sales and sales > 0) else None
        at = (sales / tot_assets) if (sales and tot_assets and tot_assets > 0) else None
        lev = (tot_assets / equity) if (tot_assets and equity and equity > 0) else None
        roe_3s = (npm * at * lev) if (npm and at and lev) else None

        rows_3s.append({
            "Fiscal Year": yr,
            "Net Margin (%)": f"{npm:.2f}%" if npm is not None else "N/A",
            "Asset Turnover (x)": f"{at:.2f}x" if at is not None else "N/A",
            "Leverage (x)": f"{lev:.2f}x" if lev is not None else "N/A",
            "DuPont ROE (%)": f"{roe_3s:.2f}%" if roe_3s is not None else "N/A",
            "_npm": npm,
            "_at": at,
            "_lev": lev,
            "_roe": roe_3s,
        })

        # 5-Stage components
        tax_burden = (pat / pbt) if (pat and pbt and pbt > 0) else None
        int_burden = (pbt / ebit) if (pbt and ebit and ebit > 0) else None
        ebit_margin = (ebit / sales * 100.0) if (ebit and sales and sales > 0) else None
        roe_5s = (tax_burden * int_burden * (ebit_margin / 100.0) * at * lev * 100.0) if (tax_burden and int_burden and ebit_margin and at and lev) else None

        rows_5s.append({
            "Fiscal Year": yr,
            "Tax Burden (x)": f"{tax_burden:.2f}x" if tax_burden is not None else "N/A",
            "Interest Burden (x)": f"{int_burden:.2f}x" if int_burden is not None else "N/A",
            "EBIT Margin (%)": f"{ebit_margin:.2f}%" if ebit_margin is not None else "N/A",
            "Asset Turnover (x)": f"{at:.2f}x" if at is not None else "N/A",
            "Leverage (x)": f"{lev:.2f}x" if lev is not None else "N/A",
            "DuPont ROE (%)": f"{roe_5s:.2f}%" if roe_5s is not None else "N/A",
        })

    # Determine key driver between last 2 available periods
    driver = "Balanced Operational & Capital Efficiency"
    if len(rows_3s) >= 2:
        prev, curr = rows_3s[-2], rows_3s[-1]
        npm_diff = (curr["_npm"] - prev["_npm"]) if (curr["_npm"] is not None and prev["_npm"] is not None) else 0.0
        at_diff = (curr["_at"] - prev["_at"]) if (curr["_at"] is not None and prev["_at"] is not None) else 0.0
        lev_diff = (curr["_lev"] - prev["_lev"]) if (curr["_lev"] is not None and prev["_lev"] is not None) else 0.0

        if npm_diff > 0.5 and lev_diff <= 0.1:
            driver = "High Quality: Operating Margin Expansion"
        elif lev_diff > 0.3 and npm_diff <= 0.0:
            driver = "Leverage-Driven: Rising Debt Relative to Equity"
        elif at_diff > 0.1:
            driver = "Asset Efficiency: Improved Capacity Utilization"
        elif npm_diff < -0.5 and lev_diff > 0.2:
            driver = "Deteriorating Margin Masked by Higher Financial Leverage"

    clean_3s = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows_3s]

    return {
        "table_3stage": pd.DataFrame(clean_3s),
        "table_5stage": pd.DataFrame(rows_5s),
        "driver": driver,
        "latest_roe": rows_3s[-1]["DuPont ROE (%)"] if rows_3s else "N/A",
    }


# =============================================================================
# FORENSIC QUALITY & FINANCIAL HEALTH (PIOTROSKI & ALTMAN Z)
# =============================================================================
def calculate_piotroski_f_score(
    pl_df: Optional[pd.DataFrame],
    bs_df: Optional[pd.DataFrame],
    cf_df: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    """Calculate 9-point Piotroski F-Score for fundamental health & forensic accounting strength."""
    if pl_df is None or pl_df.empty or bs_df is None or bs_df.empty:
        return {"score": None, "verdict": "Data Unavailable", "checklist": pd.DataFrame()}

    years = _get_chronological_years(pl_df)[-2:]
    if len(years) < 2:
        return {"score": None, "verdict": "Multi-year statements needed", "checklist": pd.DataFrame()}

    y_curr, y_prev = years[1], years[0]

    def _v(df: Optional[pd.DataFrame], metric_regex: str, yr: str) -> Optional[float]:
        if df is None or df.empty:
            return None
        row = df[df["Metric"].str.contains(metric_regex, case=False, na=False, regex=True)]
        if not row.empty and yr in row.columns:
            return safe_float(row[yr].iloc[0])
        return None

    net_inc = _v(pl_df, "Net Profit|Net Income", y_curr)
    net_inc_prev = _v(pl_df, "Net Profit|Net Income", y_prev)
    sales = _v(pl_df, "Sales|Total Revenue|Operating Revenue", y_curr)
    sales_prev = _v(pl_df, "Sales|Total Revenue|Operating Revenue", y_prev)
    assets = _v(bs_df, "Total Assets", y_curr)
    assets_prev = _v(bs_df, "Total Assets", y_prev)
    cfo = _v(cf_df, "Operating Activity|Operating Cash Flow|Cash Flow From Continuing Operating Activities", y_curr) if cf_df is not None else None
    borrowings = _v(bs_df, "Borrowings|Total Debt|Long Term Debt", y_curr)
    borrowings_prev = _v(bs_df, "Borrowings|Total Debt|Long Term Debt", y_prev)
    shares = _v(bs_df, "Equity Capital|Common Stock|Ordinary Shares Number|Share Issued", y_curr)
    shares_prev = _v(bs_df, "Equity Capital|Common Stock|Ordinary Shares Number|Share Issued", y_prev)
    opm = _v(pl_df, "OPM|Operating Margin", y_curr)
    opm_prev = _v(pl_df, "OPM|Operating Margin", y_prev)

    if opm is None and sales and sales > 0:
        ebit_c = _v(pl_df, "Operating Profit|Operating Income|EBIT", y_curr)
        if ebit_c is not None:
            opm = (ebit_c / sales) * 100.0
    if opm_prev is None and sales_prev and sales_prev > 0:
        ebit_p = _v(pl_df, "Operating Profit|Operating Income|EBIT", y_prev)
        if ebit_p is not None:
            opm_prev = (ebit_p / sales_prev) * 100.0

    roa = (net_inc / assets) if (net_inc and assets and assets > 0) else None
    roa_prev = (net_inc_prev / assets_prev) if (net_inc_prev and assets_prev and assets_prev > 0) else None
    at = (sales / assets) if (sales and assets and assets > 0) else None
    at_prev = (sales_prev / assets_prev) if (sales_prev and assets_prev and assets_prev > 0) else None

    items = [
        {"Pillar": "Profitability", "Criteria": "Positive Net Income", "Detail": f"PAT = {net_inc:,.0f}" if net_inc else "Profitable year", "Score": 1 if (net_inc and net_inc > 0) else 0},
        {"Pillar": "Profitability", "Criteria": "Positive Operating Cash Flow", "Detail": f"CFO = {cfo:,.0f}" if cfo else "Cash generated", "Score": 1 if (cfo and cfo > 0) else (1 if (net_inc and net_inc > 0) else 0)},
        {"Pillar": "Profitability", "Criteria": "Higher ROA YoY", "Detail": f"{roa*100:.1f}% vs {roa_prev*100:.1f}%" if (roa and roa_prev) else "ROA Expansion", "Score": 1 if (roa and roa_prev and roa > roa_prev) else 0},
        {"Pillar": "Profitability", "Criteria": "Quality of Earnings (CFO > Net Income)", "Detail": "Cash flow exceeds net accounting profit" if (cfo and net_inc and cfo > net_inc) else "Accruals vs CFO", "Score": 1 if (cfo and net_inc and cfo > net_inc) else (1 if cfo is None else 0)},
        {"Pillar": "Leverage", "Criteria": "Lower Long-Term Borrowings YoY", "Detail": f"{borrowings:,.0f} vs {borrowings_prev:,.0f}" if (borrowings is not None and borrowings_prev is not None) else "Debt trajectory", "Score": 1 if (borrowings is not None and borrowings_prev is not None and borrowings <= borrowings_prev) else 0},
        {"Pillar": "Leverage", "Criteria": "Zero Share Dilution", "Detail": f"{shares:,.0f} vs {shares_prev:,.0f}" if (shares is not None and shares_prev is not None) else "Equity base stable", "Score": 1 if (shares is not None and shares_prev is not None and shares <= shares_prev) else 0},
        {"Pillar": "Efficiency", "Criteria": "Operating Margin Expansion YoY", "Detail": f"{opm:.1f}% vs {opm_prev:.1f}%" if (opm and opm_prev) else "Margin trend", "Score": 1 if (opm and opm_prev and opm >= opm_prev) else 0},
        {"Pillar": "Efficiency", "Criteria": "Higher Asset Turnover YoY", "Detail": f"{at:.2f}x vs {at_prev:.2f}x" if (at and at_prev) else "Asset efficiency", "Score": 1 if (at and at_prev and at >= at_prev) else 0},
    ]

    score = sum(it["Score"] for it in items)
    total_criteria = len(items)

    verdict = "🟢 High Quality" if score >= 6 else ("🟡 Moderate / Stable" if score >= 4 else "🔴 Forensic Distress Warning")
    return {
        "score": score,
        "max_score": total_criteria,
        "verdict": verdict,
        "checklist": pd.DataFrame(items),
    }


def calculate_altman_z_score(
    pl_df: Optional[pd.DataFrame],
    bs_df: Optional[pd.DataFrame],
    market_cap: Optional[float],
) -> Dict[str, Any]:
    """Calculate Altman Z-Score for distress and credit risk assessment."""
    if pl_df is None or pl_df.empty or bs_df is None or bs_df.empty:
        return {"z_score": None, "zone": "Data Unavailable", "zone_color": "#94A3B8"}

    years = _get_chronological_years(pl_df)
    if not years:
        return {"z_score": None, "zone": "Data Unavailable", "zone_color": "#94A3B8"}
    yr = years[-1]

    def _v(df: pd.DataFrame, metric_regex: str) -> Optional[float]:
        row = df[df["Metric"].str.contains(metric_regex, case=False, na=False, regex=True)]
        return safe_float(row[yr].iloc[0]) if not row.empty else None

    sales = _v(pl_df, "Sales|Total Revenue|Operating Revenue")
    ebit = _v(pl_df, "Operating Profit|Operating Income|EBIT")
    reserves = _v(bs_df, "Reserves|Retained Earnings")
    assets = _v(bs_df, "Total Assets")
    borrowings = _v(bs_df, "Borrowings|Total Debt|Long Term Debt")
    other_assets = _v(bs_df, "Other Assets|Working Capital|Current Assets") or ((assets * 0.2) if assets else 0.0)

    if not assets or assets <= 0:
        return {"z_score": None, "zone": "Data Unavailable", "zone_color": "#94A3B8"}

    if market_cap:
        if market_cap > 1e10 and assets < 1e7:
            scale = 1e7 if (market_cap / assets > 5e6) else 1e6
            mcap_cr = market_cap / scale
        elif market_cap > 1e7 and assets < 1e6:
            mcap_cr = market_cap / 1e6
        else:
            mcap_cr = market_cap
    else:
        mcap_cr = 1000.0

    x1 = other_assets / assets
    x2 = (reserves / assets) if reserves else 0.0
    x3 = (ebit / assets) if ebit else 0.0
    x4 = (mcap_cr / borrowings) if (borrowings and borrowings > 0) else 2.0
    x5 = (sales / assets) if sales else 0.0

    z = (1.2 * x1) + (1.4 * x2) + (3.3 * x3) + (0.6 * x4) + (0.999 * x5)

    if z > 2.99:
        zone = "Safe Zone"
        zone_color = "#10B981"
        desc = "Negligible bankruptcy risk / High balance sheet solvency"
    elif z >= 1.81:
        zone = "Grey Zone"
        zone_color = "#F59E0B"
        desc = "Moderate risk / Stable solvency"
    else:
        zone = "Distress Zone"
        zone_color = "#F43F5E"
        desc = "Elevated financial distress risk"

    return {
        "z_score": z,
        "zone": zone,
        "zone_color": zone_color,
        "description": desc,
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "x4": x4,
        "x5": x5,
    }


# =============================================================================
# INTRINSIC VALUATION & REVERSE DCF ENGINE
# =============================================================================
def calculate_dcf_valuation(
    fcf0: float,
    growth_5y: float,
    terminal_growth: float,
    discount_rate: float,
    shares_out: float,
    net_debt: float = 0.0,
) -> Dict[str, Any]:
    """Calculate 2-stage Discounted Cash Flow valuation."""
    if shares_out <= 0 or discount_rate <= terminal_growth or fcf0 <= 0:
        return {
            "fair_value": None,
            "equity_val": None,
            "pv_fcf": None,
            "pv_tv": None,
            "tv": None,
            "projections": pd.DataFrame(),
        }

    curr_fcf = fcf0
    pv_fcf_sum = 0.0
    projections = []

    for t in range(1, 6):
        curr_fcf = curr_fcf * (1.0 + growth_5y)
        pv = curr_fcf / ((1.0 + discount_rate) ** t)
        pv_fcf_sum += pv
        projections.append({
            "Period": f"Year {t}",
            "Projected FCF": curr_fcf,
            "Discount Factor": 1.0 / ((1.0 + discount_rate) ** t),
            "Present Value": pv,
        })

    tv = (curr_fcf * (1.0 + terminal_growth)) / (discount_rate - terminal_growth)
    pv_tv = tv / ((1.0 + discount_rate) ** 5)
    ev = pv_fcf_sum + pv_tv
    equity_val = max(0.0, ev - net_debt)
    fair_value = equity_val / shares_out

    return {
        "fair_value": fair_value,
        "equity_val": equity_val,
        "pv_fcf": pv_fcf_sum,
        "pv_tv": pv_tv,
        "tv": tv,
        "projections": pd.DataFrame(projections),
    }


def solve_reverse_dcf(
    target_price: float,
    fcf0: float,
    terminal_growth: float,
    discount_rate: float,
    shares_out: float,
    net_debt: float = 0.0,
) -> Optional[float]:
    """Solve for market-implied 5-year FCF growth rate discounted by current stock price."""
    if target_price <= 0 or fcf0 <= 0 or shares_out <= 0 or discount_rate <= terminal_growth:
        return None

    low, high = -0.60, 1.20
    for _ in range(35):
        mid = (low + high) / 2.0
        res = calculate_dcf_valuation(fcf0, mid, terminal_growth, discount_rate, shares_out, net_debt)
        fv = res.get("fair_value")
        if fv is None or fv < target_price:
            low = mid
        else:
            high = mid

    return (low + high) / 2.0


def generate_dcf_sensitivity_matrix(
    fcf0: float,
    growth_5y: float,
    shares_out: float,
    discount_rates: List[float],
    terminal_rates: List[float],
    net_debt: float = 0.0,
) -> pd.DataFrame:
    """Generate 2D matrix of Fair Values across Discount Rates and Terminal Growth Rates."""
    matrix = {}
    for g_t in terminal_rates:
        col_name = f"g = {g_t*100:.1f}%"
        col_vals = []
        for r in discount_rates:
            if r > g_t:
                res = calculate_dcf_valuation(fcf0, growth_5y, g_t, r, shares_out, net_debt)
                fv = res.get("fair_value")
                col_vals.append(f"{fv:.1f}" if fv is not None else "N/A")
            else:
                col_vals.append("—")
        matrix[col_name] = col_vals

    df = pd.DataFrame(matrix, index=[f"WACC = {r*100:.1f}%" for r in discount_rates])
    df.index.name = "Discount Rate"
    return df


# =============================================================================
# MONTHLY RETURNS HEATMAP MATRIX
# =============================================================================
def calculate_monthly_returns_matrix(history_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Calculate multi-year monthly returns heatmap matrix and YTD statistics."""
    if history_df is None or history_df.empty or "Close" not in history_df.columns:
        return {"matrix": pd.DataFrame(), "avg_row": pd.Series(dtype=float), "years": []}

    close = history_df["Close"].dropna()
    if len(close) < 5:
        return {"matrix": pd.DataFrame(), "avg_row": pd.Series(dtype=float), "years": []}

    try:
        monthly = close.resample("ME").last()
    except Exception:
        monthly = close.resample("M").last()

    m_ret = monthly.pct_change() * 100.0
    m_df = pd.DataFrame({"Return": m_ret, "Year": m_ret.index.year, "Month": m_ret.index.month})
    pivot = m_df.pivot(index="Year", columns="Month", values="Return")

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for m in range(1, 13):
        if m not in pivot.columns:
            pivot[m] = np.nan
    pivot = pivot[[m for m in range(1, 13)]]
    pivot.columns = month_names

    def _calc_ytd(row):
        v = row.dropna()
        if v.empty:
            return np.nan
        return ((1.0 + v / 100.0).prod() - 1.0) * 100.0

    pivot["YTD"] = pivot.apply(_calc_ytd, axis=1)
    pivot = pivot.dropna(how="all")
    avg_row = pivot.mean(axis=0)
    years_list = sorted(list(pivot.index), reverse=True)

    return {
        "matrix": pivot,
        "avg_row": avg_row,
        "years": years_list,
    }


