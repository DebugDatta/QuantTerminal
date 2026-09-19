import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "utils")):
    if _p not in sys.path:
        sys.path.append(_p)

try:
    from utils.helper import inject_custom_theme, load_data, fetch_stocks
except ImportError:
    from helper import inject_custom_theme, load_data, fetch_stocks

try:
    from core.returns import compute_returns
    from statistics.stationarity import adf_test, kpss_test, pp_test, zivot_andrews
    from statistics.diagnostics import ljung_box, jarque_bera, shapiro_wilk
    from statistics.correlation import correlation_matrix
    from statistics.pca import pca_decomposition, scree_data
    from statistics.clustering import kmeans_clustering, hierarchical_data
    from statistics.timeseries import acf, pacf
except ImportError:
    from returns import compute_returns
    from stationarity import adf_test, kpss_test, pp_test, zivot_andrews
    from diagnostics import ljung_box, jarque_bera, shapiro_wilk
    from correlation import correlation_matrix
    from pca import pca_decomposition, scree_data
    from clustering import kmeans_clustering, hierarchical_data
    from timeseries import acf, pacf

EXCHANGE_OPTIONS = ("Auto", "NSE", "BSE", "Global")
ANALYSIS_TYPES = ("Stationarity", "Diagnostics", "Correlation", "PCA", "Clustering")


# ── Universe & returns ───────────────────────────────────────────────────────
def _build_universe(exchange: str) -> pd.DataFrame:
    """Build (ticker, label) asset universe for the exchange selector.

    Auto -> India NSE + Global symbols (browser-style auto resolution);
    NSE/BSE filter the India snapshot; Global keeps only the international set.
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
def _get_close(ticker: str) -> pd.Series:
    df = load_data(ticker, period="2y", interval="1d")
    if df is None or df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)
    clean = df["Close"].dropna()
    clean.index = pd.to_datetime(clean.index)
    return clean


def _asset_returns(closes: dict) -> pd.DataFrame:
    series = {}
    for ticker, close in closes.items():
        r = compute_returns(close).dropna()
        if len(r) >= 2:
            series[ticker] = r
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).dropna(how="any")


# ── Confidence badge (local PCA rule, documented) ────────────────────────────
def _pca_badge(pca: dict) -> dict:
    """Local PCA confidence badge for Page 5.

    compute_confidence_badge (MODEL_CONFIDENCE.md:156) is not implemented on
    this branch (R37). This page reproduces ONLY the documented PCA badge table
    (MODEL_CONFIDENCE.md lines 93-96) locally; no other model types are covered:
        Observations / n_features : >=10x high, 5-10x medium, <5x low
        Cumulative var (PC1+PC2)  : >0.6 high, 0.3-0.6 medium, <0.3 low
    The worst of the two checks sets the level (conservative).
    """
    ratio = float(pca["n"]) / float(pca["n_features"])
    cum = np.asarray(pca["cumulative_variance"], dtype=float)
    cumvar = float(cum[1]) if cum.size > 1 else float(cum[-1])

    def _level(x, hi, lo):
        return 2 if x >= hi else (1 if x >= lo else 0)

    obs_level = _level(ratio, 10.0, 5.0)
    var_level = _level(cumvar, 0.6, 0.3)
    worst = min(obs_level, var_level)
    label = {2: ("high", "🟢"), 1: ("medium", "🟡"), 0: ("low", "🔴")}[worst]
    return {
        "level": label[0],
        "color": label[1],
        "summary": (
            f"Observations/n_features = {ratio:.1f}x; "
            f"PC1+PC2 cumulative variance = {cumvar:.0%}"
        ),
    }


def _lower_triangle(mat: pd.DataFrame) -> pd.DataFrame:
    mask = np.triu(np.ones(mat.shape), k=0).astype(bool)
    return mat.where(~mask)


def _fmt_p(value) -> str:
    if value is None:
        return "N/A"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(value):
        return "N/A"
    return f"{value:.2e}" if abs(value) < 1e-4 else f"{value:.4f}"


def _fmt_critical(critical_values) -> str:
    if not critical_values:
        return "N/A"
    parts = []
    for key in ("1%", "5%", "10%"):
        v = critical_values.get(key)
        parts.append(f"{key}: {v:.4f}" if isinstance(v, (int, float)) else "N/A")
    return " | ".join(parts)


def _stationarity_rows(returns: pd.Series) -> list:
    rows = []
    for name, fn in (
        ("ADF", adf_test),
        ("KPSS", kpss_test),
        ("Phillips-Perron", pp_test),
        ("Zivot-Andrews", zivot_andrews),
    ):
        try:
            res = fn(returns)
            rows.append({
                "Statistic": f"{res['test_statistic']:.4f}",
                "p-value": _fmt_p(res.get("p_value")),
                "Critical values": _fmt_critical(res.get("critical_values")),
                "Conclusion": (
                    "Stationary" if res["is_stationary"] else "Non-stationary"
                ),
            })
        except (ValueError, RuntimeError) as exc:
            rows.append({
                "Statistic": "N/A",
                "p-value": "N/A",
                "Critical values": "N/A",
                "Conclusion": f"N/A ({exc})",
            })
    return rows


def _diagnostics_rows(returns: pd.Series, lags: int) -> list:
    rows = []
    for name, fn in (
        ("Ljung-Box", lambda s: ljung_box(s, lags=lags)),
        ("Jarque-Bera", jarque_bera),
        ("Shapiro-Wilk", shapiro_wilk),
    ):
        try:
            res = fn(returns)
            rows.append({
                "Statistic": f"{res['test_statistic']:.4f}",
                "p-value": _fmt_p(res.get("p_value")),
                "Conclusion": res.get("conclusion", "—"),
            })
        except (ValueError, RuntimeError) as exc:
            rows.append({
                "Statistic": "N/A",
                "p-value": "N/A",
                "Conclusion": f"N/A ({exc})",
            })
    return rows


# ── Figure builders ──────────────────────────────────────────────────────────
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


def _correlation_figure(mat: pd.DataFrame) -> go.Figure:
    mask = np.triu(np.ones(mat.shape), k=1).astype(bool)
    z = mat.where(~mask)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=[str(c) for c in mat.columns],
        y=[str(r) for r in mat.index],
        z=z.values,
        text=pd.DataFrame(np.round(z.to_numpy(float), 2), index=mat.index, columns=mat.columns).values,
        texttemplate="%{text}",
        colorscale="RdBu_r",
        zmin=-1.0,
        zmax=1.0,
        colorbar=dict(title="Corr"),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=460,
        margin=dict(l=40, r=20, t=30, b=40),
    )
    return fig


def _pca_scatter_figure(scores: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    text = [
        str(d.date()) if hasattr(d, "date") else str(d) for d in scores.index
    ]
    fig.add_trace(go.Scatter(
        x=scores["PC1"],
        y=scores["PC2"],
        mode="markers",
        text=text,
        marker=dict(color="#38BDF8", size=6, opacity=0.8),
        name="Observations",
    ))
    fig.add_shape(
        type="line",
        x0=float(scores["PC1"].min()) * 1.1,
        x1=float(scores["PC1"].max()) * 1.1,
        y0=0, y1=0, line=dict(color="rgba(255,255,255,0.3)", width=1),
    )
    fig.add_shape(
        type="line",
        x0=0, x1=0,
        y0=float(scores["PC2"].min()) * 1.1,
        y1=float(scores["PC2"].max()) * 1.1,
        line=dict(color="rgba(255,255,255,0.3)", width=1),
    )
    fig.update_xaxes(title="PC1")
    fig.update_yaxes(title="PC2")
    return _style(fig, height=380)


def _scree_figure(sc: dict) -> go.Figure:
    fig = go.Figure()
    x = [f"PC{i + 1}" for i in range(int(sc["eigenvalues"].size))]
    fig.add_trace(go.Bar(
        x=x, y=sc["eigenvalues"],
        name="Eigenvalue", marker_color="rgba(56,189,248,0.6)",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=sc["cumulative_variance"],
        name="Cumulative variance",
        yaxis="y2",
        mode="lines+markers",
        line=dict(color="#00E676", width=2),
    ))
    fig.update_layout(
        yaxis=dict(title="Eigenvalue"),
        yaxis2=dict(title="Cumulative variance", overlaying="y", side="right"),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin=dict(l=40, r=40, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _cluster_scatter_figure(km: dict) -> go.Figure:
    fig = go.Figure()
    labels = km["labels"]
    colors = ["#38BDF8", "#00E676", "#F59E0B", "#F43F5E", "#A78BFA"]
    for i, cluster in enumerate(sorted(int(v) for v in pd.unique(labels.values))):
        idx = labels.index[labels.values == cluster]
        pos = [labels.index.get_loc(a) for a in idx]
        fig.add_trace(go.Scatter(
            x=[km["pc1"][p] for p in pos],
            y=[km["pc2"][p] for p in pos],
            mode="markers+text",
            text=[str(a) for a in idx],
            textposition="top center",
            name=f"Cluster {cluster}",
            marker=dict(size=9, color=colors[i % len(colors)], opacity=0.85),
        ))
    centroids = np.asarray(km["centroids"], dtype=float)
    fig.add_trace(go.Scatter(
        x=centroids[:, 0], y=centroids[:, 1],
        mode="markers",
        name="Centroid",
        marker=dict(symbol="x", size=12, color="#FFFFFF", line=dict(width=2)),
    ))
    fig.update_xaxes(title="PC1")
    fig.update_yaxes(title="PC2")
    return _style(fig, height=380)


def _dendrogram_figure(linkage, labels) -> go.Figure:
    dn = dendrogram(
        np.asarray(linkage),
        labels=[str(a) for a in labels],
        no_plot=True,
    )
    icoords = np.asarray(dn["icoord"], dtype=float)
    dcoords = np.asarray(dn["dcoord"], dtype=float)
    fig = go.Figure()
    for i in range(icoords.shape[0]):
        xs, ys = icoords[i], dcoords[i]
        fig.add_trace(go.Scatter(
            x=[xs[0], xs[1]], y=[ys[0], ys[1]],
            mode="lines", line=dict(color="#38BDF8", width=1.2),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=[xs[1], xs[2]], y=[ys[1], ys[2]],
            mode="lines", line=dict(color="#38BDF8", width=1.2),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=[xs[2], xs[3]], y=[ys[2], ys[3]],
            mode="lines", line=dict(color="#38BDF8", width=1.2),
            showlegend=False, hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=list(range(len(dn["ivl"]))),
        y=[0.0] * len(dn["ivl"]),
        mode="text",
        text=dn["ivl"],
        textposition="bottom center",
        name="Assets",
        textfont=dict(color="#E2E8F0", size=11),
    ))
    fig.update_layout(
        xaxis=dict(showticklabels=False),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420,
        margin=dict(l=40, r=20, t=30, b=40),
    )
    return fig


def _acf_figure(kind: str, res: dict) -> go.Figure:
    y = res["acf"] if kind == "ACF" else res["pacf"]
    band = float(res["band"])
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=res["lags"], y=y,
        name=kind, marker_color="rgba(56,189,248,0.7)",
    ))
    fig.add_hline(y=band, line_dash="dot", line_color="#F43F5E")
    fig.add_hline(y=-band, line_dash="dot", line_color="#F43F5E")
    fig.update_xaxes(title="Lag")
    fig.update_yaxes(title=kind)
    return _style(fig, height=320)


# ── Page body ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    st.set_page_config(
        page_title="Statistical Analysis - QuantTerminal",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_theme()

    st.title("🔬 Statistical Analysis")
    st.caption("Formal statistical tests and multivariate analysis.")

    st.sidebar.header("📊 Data Source")
    exchange = st.sidebar.selectbox("Exchange", EXCHANGE_OPTIONS)
    universe = _build_universe(exchange)

    if universe.empty:
        st.sidebar.warning("No assets found for the selected exchange.")
    else:
        labels = universe["label"].tolist()
        st.sidebar.markdown(f"📊 **Available**: `{len(labels):,}` assets")

    selected_labels = st.sidebar.multiselect("Tickers", universe["label"].tolist())
    tickers = universe.loc[universe["label"].isin(selected_labels), "ticker"].tolist()

    st.sidebar.divider()
    st.sidebar.subheader("🔬 Analysis Settings")

    analysis_type = st.sidebar.selectbox("Analysis Type", list(ANALYSIS_TYPES))
    if analysis_type == "Diagnostics":
        lags = st.sidebar.slider("Lags", min_value=1, max_value=40, value=10)
    if analysis_type == "Clustering":
        n_available = max(2, min(10, len(tickers))) if len(tickers) >= 2 else 2
        n_clusters = st.sidebar.slider(
            "Number of Clusters", min_value=2, max_value=n_available,
            value=min(3, n_available),
        )

    if not tickers:
        st.warning("Select at least one asset to run analysis.")
        st.stop()

    closes = {t: _get_close(t) for t in tickers}
    closes = {t: s for t, s in closes.items() if not s.empty}
    if not closes:
        st.warning("No data could be loaded for the selected assets.")
        st.stop()

    st.sidebar.success(f"Loaded data for {len(closes)} asset(s)")

    returns_frame = _asset_returns(closes)
    if returns_frame.empty:
        st.warning("No usable return observations after cleaning.")
        st.stop()

    if analysis_type == "Stationarity":
        st.subheader("Stationarity — Test Results")
        rows = []
        for ticker, close in closes.items():
            r = compute_returns(close).dropna()
            if len(r) < 3:
                rows.append({
                    "Asset": ticker, "Statistic": "N/A", "p-value": "N/A",
                    "Critical values": "N/A", "Conclusion": "Insufficient observations",
                })
                continue
            for row in _stationarity_rows(r):
                row["Asset"] = ticker
                rows.append(row)
        st.dataframe(
            pd.DataFrame(rows, columns=[
                "Asset", "Statistic", "p-value", "Critical values", "Conclusion",
            ]),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "ADF / PP / Zivot-Andrews: H0 = unit root (low p → stationary). "
            "KPSS: H0 = stationary (high p → stationary). "
            "Zivot-Andrews requires ≥100 observations."
        )

    elif analysis_type == "Diagnostics":
        st.subheader("Diagnostics — Test Results")
        rows = []
        for ticker, close in closes.items():
            r = compute_returns(close).dropna()
            if len(r) < 3:
                rows.append({
                    "Asset": ticker, "Statistic": "N/A", "p-value": "N/A",
                    "Conclusion": "Insufficient observations",
                })
                continue
            for row in _diagnostics_rows(r, lags):
                row["Asset"] = ticker
                rows.append(row)
        st.dataframe(
            pd.DataFrame(rows, columns=["Asset", "Statistic", "p-value", "Conclusion"]),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Ljung-Box: H0 = no autocorrelation (10% decision). "
            "Jarque-Bera / Shapiro-Wilk: H0 = normality."
        )

        st.subheader("ACF / PACF")
        if len(closes) > 1:
            acf_ticker = st.selectbox("Series", list(closes.keys()))
        else:
            acf_ticker = list(closes.keys())[0]
        acf_series = compute_returns(closes[acf_ticker]).dropna()
        if len(acf_series) < 2 or lags >= len(acf_series):
            st.info(f"Not enough observations for {lags} lags on {acf_ticker}.")
        else:
            fig_acf = _acf_figure("ACF", acf(acf_series, lags=lags))
            fig_pacf = _acf_figure("PACF", pacf(acf_series, lags=lags))
            st.plotly_chart(fig_acf, use_container_width=True)
            st.plotly_chart(fig_pacf, use_container_width=True)

    elif analysis_type == "Correlation":
        if returns_frame.shape[1] < 2:
            st.info("Correlation requires at least 2 assets.")
        else:
            corr = correlation_matrix(returns_frame)
            lower = _lower_triangle(corr["pearson"])
            shown = lower.copy()
            for col in shown.columns:
                shown[col] = shown[col].apply(
                    lambda v: f"{v:.2f}" if pd.notna(v) else "—"
                )
            st.subheader("Correlation Matrix (lower triangle)")
            st.dataframe(shown, use_container_width=True)
            st.caption(f"Based on {int(corr['n'])} complete observations.")
            st.subheader("Correlation Heatmap")
            st.plotly_chart(
                _correlation_figure(corr["pearson"]), use_container_width=True
            )

    elif analysis_type == "PCA":
        if returns_frame.shape[1] < 2 or returns_frame.shape[0] < 3:
            st.info("PCA requires ≥2 assets and ≥3 complete observations.")
        else:
            try:
                n_comp = min(2, returns_frame.shape[1])
                pca = pca_decomposition(returns_frame, n_components=n_comp)
                badge = _pca_badge(pca)
                st.subheader(f"PCA Loadings {badge['color']}")
                st.dataframe(pca["loadings"], use_container_width=True)
                st.caption(f"Badge ({badge['level']}): {badge['summary']}")

                scores = pca["principal_components"]
                if {"PC1", "PC2"} <= set(scores.columns):
                    st.subheader("PCA Scatter — PC1 vs PC2")
                    st.plotly_chart(
                        _pca_scatter_figure(scores), use_container_width=True
                    )

                st.subheader("Scree Plot")
                st.plotly_chart(
                    _scree_figure(scree_data(returns_frame)),
                    use_container_width=True,
                )
            except (ValueError, TypeError) as exc:
                st.info(f"PCA unavailable: {exc}")

    else:  # Clustering
        if returns_frame.shape[1] < 2 or returns_frame.shape[0] < 3:
            st.info("Clustering requires ≥2 assets and ≥3 complete observations.")
        else:
            try:
                km = kmeans_clustering(returns_frame, n_clusters=n_clusters)
                st.subheader("Cluster Labels")
                st.dataframe(km["labels"].rename("Cluster").to_frame(), use_container_width=True)
                st.subheader("Cluster Scatter (PC1 vs PC2)")
                st.plotly_chart(_cluster_scatter_figure(km), use_container_width=True)

                hier = hierarchical_data(returns_frame, n_clusters=n_clusters)
                st.subheader("Hierarchical Dendrogram")
                st.plotly_chart(
                    _dendrogram_figure(hier["linkage"], hier["labels"].index),
                    use_container_width=True,
                )
            except (ValueError, TypeError) as exc:
                st.info(f"Clustering unavailable: {exc}")