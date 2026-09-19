import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from io import BytesIO
from technical.trend import sma
from technical.momentum import rsi, macd
from technical.volatility import bollinger_bands

st.set_page_config(layout="wide", page_icon="📄", page_title="Reports")


@st.cache_data(ttl=3600)
def load_data(ticker: str, exchange: str) -> pd.DataFrame:
    """Fetch OHLCV data from yfinance, handling MultiIndex columns."""
    symbol = ticker if exchange == "Global" else f"{ticker}"
    raw = yf.download(symbol, period="1y", progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw


def compute_executive_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute single-row executive summary metrics."""
    close = df["Close"].dropna()
    total_return = (close.iloc[-1] / close.iloc[0]) - 1
    days = (close.index[-1] - close.index[0]).days
    cagr = (1 + total_return) ** (365 / max(days, 1)) - 1
    daily_ret = close.pct_change().dropna()
    volatility = float(daily_ret.std() * np.sqrt(252))
    sharpe = float((daily_ret.mean() * 252) / (daily_ret.std() * np.sqrt(252))) if daily_ret.std() > 0 else 0
    downside = daily_ret[daily_ret < 0]
    sortino = float((daily_ret.mean() * 252) / (downside.std() * np.sqrt(252))) if len(downside) > 0 and downside.std() > 0 else 0
    running_max = close.cummax()
    drawdown = (close - running_max) / running_max
    max_dd = float(drawdown.min())
    var_95 = float(np.percentile(daily_ret, 5))
    return pd.DataFrame({
        "Total Return": [f"{total_return:.2%}"],
        "CAGR": [f"{cagr:.2%}"],
        "Volatility": [f"{volatility:.2%}"],
        "Sharpe": [f"{sharpe:.2f}"],
        "Sortino": [f"{sortino:.2f}"],
        "Max Drawdown": [f"{max_dd:.2%}"],
        "VaR (95%)": [f"{var_95:.2%}"],
    })


def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics plus skew and kurtosis."""
    close = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    desc = close.describe()
    daily_ret = df["Close"].pct_change().dropna()
    skew = float(daily_ret.skew())
    kurt = float(daily_ret.kurtosis())
    extra = pd.DataFrame({"Skewness": [skew], "Kurtosis": [kurt]}, index=["pct_return"])
    return pd.concat([desc, extra])


def compute_technical(df: pd.DataFrame) -> pd.DataFrame:
    """Compute last 50 rows of RSI(14), MACD, and Bollinger Bands."""
    close = df["Close"].dropna()
    rsi_vals = rsi(close, period=14)
    macd_line, signal_line, hist = macd(close)
    upper, mid, lower = bollinger_bands(close, period=20, std_dev=2)
    result = pd.DataFrame({
        "Close": close,
        "RSI(14)": rsi_vals,
        "MACD": macd_line,
        "MACD Signal": signal_line,
        "MACD Hist": hist,
        "BB Upper": upper,
        "BB Mid": mid,
        "BB Lower": lower,
    })
    return result.tail(50)


def compute_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Compute risk metrics: VaR, CVaR, volatility, max drawdown."""
    close = df["Close"].dropna()
    daily_ret = close.pct_change().dropna()
    var_hist = float(np.percentile(daily_ret, 5))
    mu, sigma = float(daily_ret.mean()), float(daily_ret.std())
    from scipy.stats import norm
    var_param = float(norm.ppf(0.05, mu, sigma))
    cvar = float(daily_ret[daily_ret <= var_hist].mean())
    vol = float(daily_ret.std() * np.sqrt(252))
    running_max = close.cummax()
    max_dd = float(((close - running_max) / running_max).min())
    return pd.DataFrame({
        "Metric": ["VaR (Historical, 95%)", "VaR (Parametric, 95%)", "CVaR (95%)", "Annualized Volatility", "Max Drawdown"],
        "Value": [f"{var_hist:.4f}", f"{var_param:.4f}", f"{cvar:.4f}", f"{vol:.4f}", f"{max_dd:.4f}"],
    })


def compute_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute SMA cross signal summary."""
    close = df["Close"].dropna()
    sma_short = sma(close, period=10)
    sma_long = sma(close, period=50)
    combined = pd.DataFrame({"sma_short": sma_short, "sma_long": sma_long}).dropna()
    signal = (combined["sma_short"] > combined["sma_long"]).astype(int).diff().dropna()
    buy_count = int((signal == 1).sum())
    sell_count = int((signal == -1).sum())
    return pd.DataFrame({
        "Metric": ["Buy Signals", "Sell Signals", "Total Signals"],
        "Count": [buy_count, sell_count, buy_count + sell_count],
    })


def export_csv(sections: dict[str, pd.DataFrame]) -> bytes:
    """Export all sections to CSV."""
    frames = []
    for name, df in sections.items():
        header = pd.DataFrame({f"=== {name} ===": [""]})
        frames.append(header)
        frames.append(df)
        frames.append(pd.DataFrame({"": [""]}))
    combined = pd.concat(frames)
    return combined.to_csv(index=True).encode("utf-8")


def export_excel(sections: dict[str, pd.DataFrame]) -> bytes:
    """Export all sections to Excel with one sheet per section."""
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, df in sections.items():
            safe_name = name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=True)
    return buffer.getvalue()


def export_pdf(sections: dict[str, pd.DataFrame]) -> bytes:
    """Export all sections to PDF using reportlab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Table, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Report", styles["Title"]))
    story.append(Spacer(1, 20))

    for name, df in sections.items():
        story.append(Paragraph(name, styles["Heading2"]))
        story.append(Spacer(1, 8))

        table_data = [list(df.columns)]
        for _, row in df.iterrows():
            table_data.append([str(v) for v in row])

        if len(table_data) > 1:
            t = Table(table_data, repeatRows=1)
            t.setStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ])
            story.append(t)
        story.append(Spacer(1, 16))

    doc.build(story)
    return buffer.getvalue()


def render_page() -> None:
    """Render the Reports & Export page."""
    st.title("📄 Reports & Export")

    with st.sidebar:
        ticker = st.text_input("Ticker", value="RELIANCE.NS")
        exchange = st.selectbox("Exchange", ["Auto", "NSE", "BSE", "Global"])
        sections = st.multiselect(
            "Include Sections",
            ["Executive Summary", "Statistics", "Technical Analysis", "Risk", "Portfolio", "Strategy"],
            default=["Executive Summary", "Statistics", "Risk"],
        )
        export_format = st.selectbox("Export Format", ["CSV", "Excel", "PDF"])
        generate = st.button("Generate Report")

    if not generate:
        return

    progress = st.progress(0, text="Loading data...")
    try:
        df = load_data(ticker, exchange)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    if df.empty:
        st.warning("No data returned for this ticker.")
        return

    total_steps = len(sections)
    section_data: dict[str, pd.DataFrame] = {}

    for i, section in enumerate(sections):
        progress.progress((i + 1) / (total_steps + 1), text=f"Generating {section}...")

        if section == "Executive Summary":
            section_data["Executive Summary"] = compute_executive_summary(df)
        elif section == "Statistics":
            section_data["Statistics"] = compute_statistics(df)
        elif section == "Technical Analysis":
            section_data["Technical Analysis"] = compute_technical(df)
        elif section == "Risk":
            section_data["Risk"] = compute_risk(df)
        elif section == "Portfolio":
            section_data["Portfolio"] = pd.DataFrame({
                "Note": ["Select multiple tickers in Portfolio Lab for portfolio reports"],
            })
        elif section == "Strategy":
            section_data["Strategy"] = compute_strategy(df)

    progress.progress(1.0, text="Generation complete!")

    for name, data in section_data.items():
        st.subheader(name)
        st.dataframe(data, use_container_width=True)

    st.divider()

    if export_format == "CSV":
        raw = export_csv(section_data)
        st.download_button("Download CSV", data=raw, file_name=f"{ticker}_report.csv", mime="text/csv")
    elif export_format == "Excel":
        raw = export_excel(section_data)
        st.download_button("Download Excel", data=raw, file_name=f"{ticker}_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    elif export_format == "PDF":
        raw = export_pdf(section_data)
        st.download_button("Download PDF", data=raw, file_name=f"{ticker}_report.pdf", mime="application/pdf")


if __name__ == "__main__":
    render_page()
