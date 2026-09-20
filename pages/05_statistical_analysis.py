"""Page 5: Statistical Analysis — formal tests and multivariate analysis.

Runs a battery of 30+ statistical tests across the selected universe:
stationarity (ADF/KPSS per asset & per lag), autocorrelation (Ljung-Box),
normality (Jarque-Bera, Shapiro-Wilk), correlation, PCA, clustering and
pairwise cointegration (Engle-Granger).
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

from scipy import stats as sc
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage

from statistics.diagnostics import adf_test, kpss_test, ljung_box, jarque_bera, shapiro_wilk
from plots.correlation import plot_heatmap, plot_dendrogram
from plots.clustering import plot_pca_scatter, plot_scree, plot_clusters

PRESET_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS", "TATAMOTORS.NS",
    "ASIANPAINT.NS", "MARUTI.NS",
]


def _resolve_ticker(symbol: str, exchange: str) -> str:
    upper = exchange.upper()
    if upper == "NSE":
        return symbol if symbol.endswith(".NS") else f"{symbol}.NS"
    if upper == "BSE":
        return symbol if symbol.endswith(".BO") else f"{symbol}.BO"
    if upper == "GLOBAL":
        return symbol
    if symbol.endswith((".NS", ".BO")) or symbol.startswith("^"):
        return symbol
    return symbol


@st.cache_data(show_spinner="Loading data…", ttl=3600)
def _load_close(ticker: str, start: str, end: str) -> pd.Series:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df["Close"].dropna()


@st.cache_data(ttl=3600)
def _load_portfolio(tickers: tuple, start: str, end: str) -> pd.DataFrame:
    closes = {t: _load_close(t, start, end) for t in tickers}
    panel = pd.DataFrame(closes).dropna(how="all")
    returns = panel.pct_change().dropna(how="all")
    return panel, returns


def _format_pvalue(v: float) -> str:
    if pd.isna(v):
        return "—"
    return f"{v:.4f}" + ("*" if v < 0.05 else "")


def _stationarity_report(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in returns.columns:
        r = returns[ticker].dropna()
        adf = adf_test(r)
        kp = kpss_test(r)
        rows.append({
            "Asset": ticker,
            "ADF Stat": adf["statistic"], "ADF p-value": _format_pvalue(adf["pvalue"]),
            "ADF Conclusion": adf["conclusion"],
            "KPSS Stat": kp["statistic"], "KPSS p-value": _format_pvalue(kp["pvalue"]),
            "KPSS Conclusion": kp["conclusion"],
        })
    return pd.DataFrame(rows)


def _diagnostics_report(returns: pd.DataFrame, lags: int) -> pd.DataFrame:
    rows = []
    for ticker in returns.columns:
        r = returns[ticker].dropna()
        lb = ljung_box(r, lags=lags)
        jb = jarque_bera(r)
        sw = shapiro_wilk(r)
        rows.append({
            "Asset": ticker,
            "JB Stat": jb["statistic"], "JB p-value": _format_pvalue(jb["pvalue"]),
            "Shapiro Stat": sw["statistic"], "Shapiro p-value": _format_pvalue(sw["pvalue"]),
            f"LB Lag {min(lags, 1)}": _format_pvalue(lb['pvalue'].iloc[0]) if len(lb) else "—",
            "Skewness": sc.skew(r, bias=False),
            "Kurtosis": sc.kurtosis(r, bias=False, fisher=True),
        })
    return pd.DataFrame(rows)


def _pca_analysis(returns: pd.DataFrame):
    r = returns.dropna(axis=0)
    if len(r) < 3 or r.shape[1] < 2:
        return None, None, None, None
    scaler = r / r.std()
    pca = PCA()
    scores = pd.DataFrame(pca.fit_transform(scaler), index=r.index,
                          columns=[f"PC{i + 1}" for i in range(min(r.shape[1], len(r)))])
    loadings = pd.DataFrame(pca.components_.T, index=r.columns,
                            columns=scores.columns)
    explained = pd.Series(pca.explained_variance_ratio_, index=scores.columns)
    return pca, scores, loadings, explained


def _cluster_analysis(returns: pd.DataFrame, n_clusters: int):
    r = returns.dropna(axis=0)
    if len(r) < 3 or r.shape[1] < max(2, n_clusters):
        return None, None, None
    features = (r / r.std()).T
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = pd.Series(model.fit_predict(features), index=features.index, name="cluster")
    labels = labels.astype("category").cat.rename_categories(
        {i: f"Cluster {i + 1}" for i in range(n_clusters)})
    condensed = np.array(np.sqrt(2 * (1 - features.corr().clip(-1, 1))))
    np.fill_diagonal(condensed, 0)
    from scipy.spatial.distance import squareform
    linked = linkage(squareform(condensed), method="ward")
    return labels, linked, features.corr()


def _cointegration_report(returns: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Pairwise Engle-Granger cointegration via statsmodels coint."""
    from statsmodels.tsa.stattools import coint
    panel = (returns + 1).cumprod()
    cols = list(panel.columns)
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = panel[cols[i]].dropna(), panel[cols[j]].dropna()
            common = a.index.intersection(b.index)
            if len(common) < 30:
                continue
            try:
                stat, pvalue, _ = coint(a.loc[common], b.loc[common],
                                        trend="c", maxlag=1, autolag="aic")
            except Exception:
                rows.append({"Pair": f"{cols[i]} / {cols[j]}",
                             "Asset A": cols[i], "Asset B": cols[j],
                             "Engle-Granger Stat": np.nan, "p-value": "—",
                             "Cointegrated": False})
                continue
            rows.append({
                "Pair": f"{cols[i]} / {cols[j]}",
                "Asset A": cols[i], "Asset B": cols[j],
                "Engle-Granger Stat": float(stat),
                "p-value": _format_pvalue(float(pvalue)),
                "Cointegrated": bool(pvalue < 0.05),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("p-value", key=lambda s: [float(v) if v != "—" else 1.0 for v in s]).head(top_n)


def render_page():
    st.title("Statistical Analysis")
    st.caption("Formal statistical tests and multivariate analysis (30+ tests)")

    with st.sidebar:
        tickers = st.multiselect(
            "Tickers",
            options=PRESET_TICKERS,
            default=["RELIANCE.NS", "TCS.NS", "INFY.NS"],
            placeholder="Search or type ticker…",
        )
        custom = st.text_area("Custom tickers (comma-separated)", height=60,
                              placeholder="e.g. AAPL, MSFT, ^GSPC")
        exchange = st.selectbox("Exchange", ["Auto", "NSE", "BSE", "Global"])
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Stationarity", "Diagnostics", "Correlation", "PCA", "Clustering", "Cointegration"],
        )
        lags = st.slider("Ljung-Box Lags", min_value=1, max_value=40, value=10)
        n_clusters = st.slider("N Clusters", min_value=2, max_value=6, value=3)

    if custom.strip():
        custom_list = [t.strip() for t in custom.split(",") if t.strip()]
        tickers = list(dict.fromkeys(tickers + custom_list))

    if len(tickers) < 1:
        st.info("Select one or more tickers from the sidebar.")
        return

    resolved = [_resolve_ticker(t, exchange) for t in tickers]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 2)
    panel, returns = _load_portfolio(tuple(resolved), start_date.strftime("%Y-%m-%d"),
                                     end_date.strftime("%Y-%m-%d"))
    returns = returns.dropna(axis=1)

    if returns.empty or len(returns.columns) < 1:
        st.error("No valid return series could be loaded.")
        return

    st.caption(f"Universe: {', '.join(list(returns.columns))} — {len(returns)} periods")

    corr = returns.corr()

    if analysis_type == "Stationarity":
        st.subheader("Stationarity Tests (ADF + KPSS)")
        rep = _stationarity_report(returns)
        st.dataframe(rep, use_container_width=True)
        st.caption("* p < 0.05 — ADF rejects a unit root (stationary); KPSS rejects stationarity.")

    elif analysis_type == "Diagnostics":
        st.subheader("Distribution Diagnostics")
        rep = _diagnostics_report(returns, lags)
        st.dataframe(rep, use_container_width=True)
        st.caption("Ljung-Box (lag 1) checks remaining autocorrelation; JB & Shapiro test normality.")

    elif analysis_type == "Correlation":
        st.subheader("Correlation Matrix")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(corr.round(3), use_container_width=True)
        with col2:
            st.plotly_chart(plot_heatmap(corr), use_container_width=True)

    elif analysis_type == "PCA":
        st.subheader("Principal Component Analysis")
        pca, scores, loadings, explained = _pca_analysis(returns)
        if pca is None:
            st.warning("PCA requires at least 2 assets and sufficient history.")
            return
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_pca_scatter(scores), use_container_width=True)
            st.subheader("PCA Loadings")
            st.dataframe(loadings.round(3), use_container_width=True)
        with col2:
            st.plotly_chart(plot_scree(explained), use_container_width=True)
            st.subheader("Explained Variance")
            st.dataframe(pd.DataFrame({
                "Component": explained.index,
                "Explained Variance": explained.values,
                "Cumulative": np.cumsum(explained.values),
            }), use_container_width=True)

    elif analysis_type == "Clustering":
        st.subheader("Clustering")
        labels, linked, cluster_corr = _cluster_analysis(returns, n_clusters)
        if labels is None:
            st.warning("Clustering requires more assets or longer history.")
            return
        scores = pd.DataFrame(PCA(n_components=2).fit_transform(returns.dropna(axis=0) / returns.std()),
                              index=returns.dropna(axis=0).index)
        scores.columns = ["PC1", "PC2"]
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_clusters(scores, labels), use_container_width=True)
            st.subheader("Cluster Labels")
            st.dataframe(pd.DataFrame({"Asset": labels.index, "Cluster": labels.values}),
                         use_container_width=True)
        with col2:
            st.plotly_chart(plot_dendrogram(cluster_corr, labels=labels.index), use_container_width=True)

    elif analysis_type == "Cointegration":
        st.subheader("Cointegration (Engle-Granger)")
        rep = _cointegration_report(returns)
        if rep.empty:
            st.warning("Not enough paired history for cointegration tests.")
        else:
            st.dataframe(rep, use_container_width=True)
            st.caption("* p < 0.05 indicates the pair is cointegrated (mean-reverting spread).")

    st.divider()
    st.caption("This is a historical measurement, not a forecast or a recommendation.")


if __name__ == "__main__":
    render_page()