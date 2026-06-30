"""
ForeTicker interactive analytics dashboard.

Run with:  streamlit run dashboard/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import PROCESSED_FEATURES_DIR, DEFAULT_TICKERS
from features.fundamentals import fetch_fundamentals

st.set_page_config(page_title="ForeTicker", layout="wide", page_icon="📈")


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_available_tickers() -> list[str]:
    available = []
    for ticker in DEFAULT_TICKERS:
        path = PROCESSED_FEATURES_DIR / f"{ticker.replace('.', '_')}_features.parquet"
        if path.exists():
            available.append(ticker)
    return available


@st.cache_data(ttl=600)
def load_features(ticker: str) -> pd.DataFrame:
    path = PROCESSED_FEATURES_DIR / f"{ticker.replace('.', '_')}_features.parquet"
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


@st.cache_data(ttl=3600)
def load_fundamentals_cached(ticker: str) -> dict:
    try:
        return fetch_fundamentals(ticker)
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=600)
def load_backtest_metrics(ticker: str) -> dict | None:
    try:
        import mlflow
        from config import MLFLOW_TRACKING_URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name("foreticker_walkforward")
        if exp is None:
            return None
        runs = client.search_runs(
            exp.experiment_id,
            filter_string=f"params.ticker = '{ticker}'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            return None
        return runs[0].data.metrics
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_large(n) -> str:
    if n is None:
        return "—"
    n = float(n)
    for unit, div in [("T", 1e12), ("B", 1e9), ("M", 1e6)]:
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:,.0f}"


def fmt_pct(n) -> str:
    return "—" if n is None else f"{n * 100:.2f}%"


def fmt_num(n, decimals: int = 2) -> str:
    return "—" if n is None else f"{n:.{decimals}f}"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

tickers = get_available_tickers()
if not tickers:
    st.error("No feature matrices found. Run the pipeline (Phases 1-3) before launching the dashboard.")
    st.stop()

st.sidebar.title("📈 ForeTicker")
ticker = st.sidebar.selectbox("Ticker", tickers)

df = load_features(ticker)

default_start = (df.index.max() - pd.DateOffset(years=1)).to_pydatetime()
default_start = max(default_start, df.index.min().to_pydatetime())

date_range = st.sidebar.slider(
    "Date range",
    min_value=df.index.min().to_pydatetime(),
    max_value=df.index.max().to_pydatetime(),
    value=(default_start, df.index.max().to_pydatetime()),
)
df = df[(df.index >= date_range[0]) & (df.index <= date_range[1])]

st.sidebar.subheader("Chart overlays")
show_ema = st.sidebar.checkbox("EMA 20 / 50", value=True)
show_bbands = st.sidebar.checkbox("Bollinger Bands", value=True)

st.sidebar.subheader("Panels")
show_rsi_macd = st.sidebar.checkbox("RSI / MACD", value=True)
show_sentiment = st.sidebar.checkbox("Sentiment", value=True)
show_backtest = st.sidebar.checkbox("Backtest performance", value=True)

with st.sidebar.expander("🔔 Real-time alerts"):
    st.caption(
        "Coming soon: a background watcher will monitor live news and "
        "fundamentals for each ticker and push an alert (sentiment spike, "
        "earnings surprise, sudden volume/price move) the moment it's "
        "detected — instead of finding out after the move already happened."
    )


# ---------------------------------------------------------------------------
# Header — price + fundamentals
# ---------------------------------------------------------------------------

fundamentals = load_fundamentals_cached(ticker)
latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest
day_change = (latest["Close"] - prev["Close"]) / prev["Close"] if prev["Close"] else 0

name = fundamentals.get("shortName") or ticker
sector = fundamentals.get("sector", "—")
industry = fundamentals.get("industry", "—")

st.title(f"{name} ({ticker})")
st.caption(f"{sector} · {industry}")

col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
col1.metric("Price", f"${latest['Close']:.2f}", f"{day_change * 100:.2f}%")
col2.metric("Market Cap", fmt_large(fundamentals.get("marketCap")))
col3.metric("P/E (trailing)", fmt_num(fundamentals.get("trailingPE")))
col4.metric("Forward P/E", fmt_num(fundamentals.get("forwardPE")))
col5.metric("Beta", fmt_num(fundamentals.get("beta")))
col6.metric("Div. Yield", fmt_pct(fundamentals.get("dividendYield", 0) and fundamentals.get("dividendYield") / 100))
col7.metric("52W High", fmt_num(fundamentals.get("fiftyTwoWeekHigh")))
col8.metric("52W Low", fmt_num(fundamentals.get("fiftyTwoWeekLow")))

rec = fundamentals.get("recommendationKey", "—")
target = fundamentals.get("targetMeanPrice")
n_analysts = fundamentals.get("numberOfAnalystOpinions")
if rec != "—":
    st.info(f"Analyst consensus: **{rec.upper()}** · mean target ${target:.2f} ({n_analysts} analysts)"
            if target else f"Analyst consensus: **{rec.upper()}**")


# ---------------------------------------------------------------------------
# Price chart with technical overlays
# ---------------------------------------------------------------------------

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25],
    vertical_spacing=0.03,
)

fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
    name="Price",
), row=1, col=1)

if show_ema:
    fig.add_trace(go.Scatter(x=df.index, y=df["ema_20"], name="EMA 20", line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ema_50"], name="EMA 50", line=dict(width=1)), row=1, col=1)

if show_bbands:
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_high"], name="BB High",
                              line=dict(width=1, dash="dot", color="gray")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_low"], name="BB Low",
                              line=dict(width=1, dash="dot", color="gray"),
                              fill="tonexty", fillcolor="rgba(128,128,128,0.1)"), row=1, col=1)

fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color="rgba(100,149,237,0.5)"), row=2, col=1)

fig.update_layout(height=550, xaxis_rangeslider_visible=False, margin=dict(t=20, b=20))
st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------
# RSI / MACD
# ---------------------------------------------------------------------------

if show_rsi_macd:
    c1, c2 = st.columns(2)

    with c1:
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=df.index, y=df["rsi_14"], name="RSI 14"))
        rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
        rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
        rsi_fig.update_layout(title="RSI (14)", height=300, margin=dict(t=40, b=20))
        st.plotly_chart(rsi_fig, width='stretch')

    with c2:
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD"))
        macd_fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="Signal"))
        macd_fig.add_trace(go.Bar(x=df.index, y=df["macd_diff"], name="Histogram",
                                   marker_color="rgba(100,149,237,0.5)"))
        macd_fig.update_layout(title="MACD", height=300, margin=dict(t=40, b=20))
        st.plotly_chart(macd_fig, width='stretch')


# ---------------------------------------------------------------------------
# Sentiment
# ---------------------------------------------------------------------------

if show_sentiment:
    st.subheader("News Sentiment")
    sent_fig = make_subplots(specs=[[{"secondary_y": True}]])
    sent_fig.add_trace(go.Scatter(x=df.index, y=df["sentiment_mean"], name="Sentiment (net)",
                                   line=dict(color="orange")), secondary_y=False)
    sent_fig.add_trace(go.Bar(x=df.index, y=df["article_count"], name="Article count",
                               marker_color="rgba(100,149,237,0.3)"), secondary_y=True)
    sent_fig.add_hline(y=0, line_dash="dash", line_color="gray", secondary_y=False)
    sent_fig.update_layout(height=300, margin=dict(t=20, b=20))
    sent_fig.update_yaxes(title_text="Net sentiment", secondary_y=False)
    sent_fig.update_yaxes(title_text="Article count", secondary_y=True)
    st.plotly_chart(sent_fig, width='stretch')

    if (df["is_earnings_day"] == 1).any():
        earnings_dates = df[df["is_earnings_day"] == 1].index.strftime("%Y-%m-%d").tolist()
        st.caption(f"📅 Earnings days in range: {', '.join(earnings_dates)}")


# ---------------------------------------------------------------------------
# Backtest performance
# ---------------------------------------------------------------------------

if show_backtest:
    st.subheader("Walk-Forward Backtest Performance")
    metrics = load_backtest_metrics(ticker)
    if metrics is None:
        st.caption("No backtest results yet for this ticker — run `backtest/walkforward.py`.")
    else:
        bcol1, bcol2, bcol3, bcol4 = st.columns(4)
        bcol1.metric("Direction Accuracy", fmt_pct(metrics.get("direction_accuracy")),
                     f"{(metrics.get('direction_accuracy', 0) - metrics.get('baseline_direction_accuracy', 0)) * 100:.2f}pp vs baseline")
        bcol2.metric("Sharpe Ratio", fmt_num(metrics.get("sharpe_ratio")),
                     f"{metrics.get('sharpe_ratio', 0) - metrics.get('baseline_sharpe_ratio', 0):.2f} vs baseline")
        bcol3.metric("Max Drawdown", fmt_pct(metrics.get("max_drawdown")))
        bcol4.metric("Annualized Return", fmt_pct(metrics.get("annualized_return")),
                     f"{(metrics.get('annualized_return', 0) - metrics.get('baseline_annualized_return', 0)) * 100:.2f}pp vs baseline")
