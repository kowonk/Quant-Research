import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import brentq
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import ta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import json
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="QuantLab", layout="wide", page_icon="📊")

RISK_FREE_RATE = 0.05  # update periodically or pull from FRED

# S&P 500 representative universe (top ~100 liquid names across sectors)
DEFAULT_UNIVERSE = [
    "AAPL","MSFT","AMZN","NVDA","GOOGL","META","TSLA","BRK-B","UNH","JNJ",
    "JPM","V","PG","MA","HD","MRK","ABBV","PEP","KO","AVGO",
    "COST","TMO","MCD","WMT","ACN","LIN","CSCO","ABT","CRM","DHR",
    "NKE","ADBE","TXN","CMCSA","NEE","PM","UNP","RTX","HON","LOW",
    "INTC","QCOM","AMAT","ISRG","MDLZ","ADP","BKNG","GILD","VRTX","AMT",
    "SYK","ADI","LRCX","MMC","SCHW","PGR","CI","CB","ZTS","REGN",
    "BDX","SO","DUK","CME","CL","ITW","SHW","MO","EOG","PYPL",
    "SNPS","APD","HUM","ORLY","MCK","CDNS","KLAC","MNST","MSI","AJG",
    "FTNT","CCI","GM","F","SQ","COIN","PLTR","SOFI","HOOD","RIVN",
    "AMD","MU","MRVL","ON","SMCI","ARM","CRWD","PANW","ZS","NET"
]


# ─────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=DM+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'DM Sans', sans-serif;
    }
    code, .stCode, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    .stApp {
        background-color: #0a0a0f;
    }
    .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: -0.02em;
    }
    .score-card {
        background: linear-gradient(135deg, #12121a 0%, #1a1a2e 100%);
        border: 1px solid #2a2a3e;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
    }
    .score-high { border-left: 4px solid #00ff88; }
    .score-mid { border-left: 4px solid #ffaa00; }
    .score-low { border-left: 4px solid #ff4444; }
    .metric-label {
        color: #8888aa;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .metric-value {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .thesis-box {
        background: #12121a;
        border: 1px solid #2a2a3e;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    div[data-testid="stMetric"] {
        background: #12121a;
        border: 1px solid #2a2a3e;
        border-radius: 10px;
        padding: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_data(ticker, period="1y"):
    """Fetch OHLCV data for a single ticker."""
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            return None
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_info(ticker):
    """Fetch fundamental info for a ticker."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return info
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_options_chain(ticker):
    """Fetch current options chain."""
    try:
        t = yf.Ticker(ticker)
        expirations = t.options
        if not expirations:
            return None, []
        chains = {}
        for exp in expirations[:6]:  # limit to first 6 expirations
            try:
                chain = t.option_chain(exp)
                chains[exp] = {
                    "calls": chain.calls,
                    "puts": chain.puts
                }
            except Exception:
                continue
        return chains, list(expirations)
    except Exception:
        return None, []


def compute_momentum_score(df, periods=[21, 63, 126, 252]):
    """Multi-period momentum composite. Returns 0-100 score."""
    if df is None or len(df) < 252:
        return np.nan
    close = df["Close"].values.flatten()
    scores = []
    for p in periods:
        if len(close) > p:
            ret = (close[-1] / close[-p]) - 1
            scores.append(ret)
    if not scores:
        return np.nan
    # Rank-normalize: simple percentile-ish via sigmoid
    avg_ret = np.mean(scores)
    score = 1 / (1 + np.exp(-avg_ret * 10))  # sigmoid scaling
    return round(score * 100, 1)


def compute_mean_reversion_score(df, window=20):
    """Z-score based mean reversion signal. Returns z-score."""
    if df is None or len(df) < window + 5:
        return np.nan
    close = df["Close"].values.flatten()
    sma = np.mean(close[-window:])
    std = np.std(close[-window:])
    if std == 0:
        return 0
    z = (close[-1] - sma) / std
    return round(z, 2)


def compute_volatility_metrics(df, window=21):
    """Realized vol, vol regime, vol-of-vol."""
    if df is None or len(df) < window + 5:
        return {}
    close = df["Close"].values.flatten()
    log_ret = np.diff(np.log(close))
    realized_vol = np.std(log_ret[-window:]) * np.sqrt(252)
    hist_vol_long = np.std(log_ret[-min(126, len(log_ret)):]) * np.sqrt(252)
    # Vol regime: current vs historical
    vol_ratio = realized_vol / hist_vol_long if hist_vol_long > 0 else 1
    # Vol of vol
    if len(log_ret) >= 63:
        rolling_vols = pd.Series(log_ret).rolling(window).std() * np.sqrt(252)
        vov = rolling_vols.dropna().std()
    else:
        vov = np.nan
    return {
        "realized_vol": round(realized_vol * 100, 1),
        "hist_vol_6m": round(hist_vol_long * 100, 1),
        "vol_regime": round(vol_ratio, 2),
        "vol_of_vol": round(vov * 100, 2) if not np.isnan(vov) else None
    }


def compute_fundamental_score(info):
    """Score fundamentals 0-100 based on quality + value factors."""
    score = 50  # neutral baseline
    # PE ratio (value)
    pe = info.get("trailingPE") or info.get("forwardPE")
    if pe and pe > 0:
        if pe < 15: score += 10
        elif pe < 25: score += 5
        elif pe > 50: score -= 10
    # Revenue growth (quality)
    rev_growth = info.get("revenueGrowth")
    if rev_growth:
        if rev_growth > 0.20: score += 15
        elif rev_growth > 0.10: score += 10
        elif rev_growth > 0: score += 5
        else: score -= 5
    # Profit margins (quality)
    margin = info.get("profitMargins")
    if margin:
        if margin > 0.25: score += 10
        elif margin > 0.10: score += 5
        elif margin < 0: score -= 10
    # ROE (quality)
    roe = info.get("returnOnEquity")
    if roe:
        if roe > 0.25: score += 10
        elif roe > 0.15: score += 5
    # Debt to equity (risk)
    dte = info.get("debtToEquity")
    if dte:
        if dte > 200: score -= 10
        elif dte > 100: score -= 5
        elif dte < 30: score += 5
    # Free cash flow yield
    mcap = info.get("marketCap")
    fcf = info.get("freeCashflow")
    if mcap and fcf and mcap > 0:
        fcf_yield = fcf / mcap
        if fcf_yield > 0.06: score += 10
        elif fcf_yield > 0.03: score += 5
    return max(0, min(100, score))


def compute_technical_signals(df):
    """Compute technical indicators using ta library."""
    if df is None or len(df) < 50:
        return {}
    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    volume = df["Volume"].squeeze()
    signals = {}
    try:
        signals["rsi_14"] = round(ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1], 1)
    except: signals["rsi_14"] = None
    try:
        macd_obj = ta.trend.MACD(close)
        signals["macd"] = round(macd_obj.macd().iloc[-1], 3)
        signals["macd_signal"] = round(macd_obj.macd_signal().iloc[-1], 3)
        signals["macd_hist"] = round(macd_obj.macd_diff().iloc[-1], 3)
    except: pass
    try:
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        signals["bb_pct"] = round(bb.bollinger_pband().iloc[-1], 3)
    except: signals["bb_pct"] = None
    try:
        signals["sma_20"] = round(close.rolling(20).mean().iloc[-1], 2)
        signals["sma_50"] = round(close.rolling(50).mean().iloc[-1], 2)
        signals["sma_200"] = round(close.rolling(200).mean().iloc[-1], 2) if len(close) >= 200 else None
    except: pass
    try:
        signals["adx"] = round(ta.trend.ADXIndicator(high, low, close).adx().iloc[-1], 1)
    except: signals["adx"] = None
    try:
        signals["atr"] = round(ta.volatility.AverageTrueRange(high, low, close).average_true_range().iloc[-1], 2)
    except: signals["atr"] = None
    # Volume trend
    try:
        avg_vol_20 = volume.rolling(20).mean().iloc[-1]
        signals["vol_ratio"] = round(volume.iloc[-1] / avg_vol_20, 2) if avg_vol_20 > 0 else None
    except: signals["vol_ratio"] = None
    return signals


def black_scholes(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes pricing."""
    if T <= 0 or sigma <= 0:
        return max(0, S - K) if option_type == "call" else max(0, K - S)
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == "call":
        return S * stats.norm.cdf(d1) - K * np.exp(-r*T) * stats.norm.cdf(d2)
    else:
        return K * np.exp(-r*T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)


def implied_vol(market_price, S, K, T, r, option_type="call"):
    """Newton-Raphson implied vol solver."""
    if T <= 0:
        return np.nan
    try:
        def objective(sigma):
            return black_scholes(S, K, T, r, sigma, option_type) - market_price
        iv = brentq(objective, 0.001, 5.0, maxiter=100)
        return iv
    except Exception:
        return np.nan


def compute_greeks(S, K, T, r, sigma, option_type="call"):
    """Compute option Greeks."""
    if T <= 0 or sigma <= 0:
        return {}
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    delta = stats.norm.cdf(d1) if option_type == "call" else stats.norm.cdf(d1) - 1
    gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta_call = (-S * stats.norm.pdf(d1) * sigma / (2*np.sqrt(T))
                  - r * K * np.exp(-r*T) * stats.norm.cdf(d2))
    theta_put = (-S * stats.norm.pdf(d1) * sigma / (2*np.sqrt(T))
                 + r * K * np.exp(-r*T) * stats.norm.cdf(-d2))
    theta = theta_call / 365 if option_type == "call" else theta_put / 365
    vega = S * stats.norm.pdf(d1) * np.sqrt(T) / 100
    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 4),
        "vega": round(vega, 4)
    }


def compute_composite_thesis_score(momentum, fundamental, technicals, vol_metrics):
    """
    Weighted composite score combining all signals.
    Returns score 0-100 and confidence level.
    """
    weights = {
        "momentum": 0.25,
        "fundamental": 0.30,
        "technical": 0.25,
        "volatility": 0.20
    }
    scores = {}
    # Momentum component (already 0-100)
    scores["momentum"] = momentum if not np.isnan(momentum) else 50
    # Fundamental (already 0-100)
    scores["fundamental"] = fundamental
    # Technical composite
    tech_score = 50
    rsi = technicals.get("rsi_14")
    if rsi:
        if 40 <= rsi <= 60: tech_score += 10  # neutral, room to run
        elif rsi < 30: tech_score += 15  # oversold bounce potential
        elif rsi > 70: tech_score -= 10  # overbought risk
    macd_h = technicals.get("macd_hist")
    if macd_h:
        if macd_h > 0: tech_score += 10
        else: tech_score -= 5
    adx = technicals.get("adx")
    if adx:
        if adx > 25: tech_score += 10  # strong trend
    sma_50 = technicals.get("sma_50")
    sma_200 = technicals.get("sma_200")
    if sma_50 and sma_200:
        if sma_50 > sma_200: tech_score += 10  # golden cross territory
        else: tech_score -= 5
    scores["technical"] = max(0, min(100, tech_score))
    # Volatility (lower realized vol relative to historical = more stable)
    vol_score = 50
    if vol_metrics:
        regime = vol_metrics.get("vol_regime", 1)
        if regime < 0.8: vol_score += 15  # calm
        elif regime < 1.0: vol_score += 5
        elif regime > 1.5: vol_score -= 15  # elevated
        rv = vol_metrics.get("realized_vol", 25)
        if rv < 20: vol_score += 10
        elif rv > 40: vol_score -= 10
    scores["volatility"] = max(0, min(100, vol_score))
    # Weighted composite
    composite = sum(scores[k] * weights[k] for k in weights)
    # Confidence based on data agreement
    score_values = list(scores.values())
    std_of_scores = np.std(score_values)
    if std_of_scores < 10:
        confidence = "HIGH"
    elif std_of_scores < 20:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
    return {
        "composite": round(composite, 1),
        "confidence": confidence,
        "components": scores,
        "weights": weights
    }


def generate_thesis(ticker, info, thesis_result, technicals, vol_metrics):
    """Generate a human-readable thesis summary."""
    score = thesis_result["composite"]
    conf = thesis_result["confidence"]
    components = thesis_result["components"]
    lines = []
    # Direction
    if score >= 65:
        lines.append(f"**BULLISH** on {ticker} (Score: {score}/100, Confidence: {conf})")
    elif score >= 45:
        lines.append(f"**NEUTRAL** on {ticker} (Score: {score}/100, Confidence: {conf})")
    else:
        lines.append(f"**BEARISH** on {ticker} (Score: {score}/100, Confidence: {conf})")
    # Momentum
    mom = components.get("momentum", 50)
    if mom > 65:
        lines.append("• Strong multi-timeframe momentum — price trend is accelerating")
    elif mom < 35:
        lines.append("• Weak momentum — price trend is decelerating or negative")
    # Fundamentals
    fund = components.get("fundamental", 50)
    pe = info.get("trailingPE") or info.get("forwardPE")
    rev_g = info.get("revenueGrowth")
    margin = info.get("profitMargins")
    if fund > 65:
        parts = []
        if pe and pe < 25: parts.append(f"reasonable valuation (PE {pe:.1f})")
        if rev_g and rev_g > 0.10: parts.append(f"strong revenue growth ({rev_g:.0%})")
        if margin and margin > 0.15: parts.append(f"healthy margins ({margin:.0%})")
        lines.append("• Solid fundamentals: " + ", ".join(parts) if parts else "• Solid fundamentals")
    elif fund < 35:
        lines.append("• Weak fundamentals — watch for deterioration")
    # Technicals
    tech = components.get("technical", 50)
    rsi = technicals.get("rsi_14")
    if tech > 65:
        lines.append(f"• Technicals favor upside — RSI {rsi}, trend confirmed by MACD/ADX")
    elif tech < 35:
        lines.append(f"• Technical weakness — RSI {rsi}, watch for breakdown")
    # Volatility
    vol_s = components.get("volatility", 50)
    rv = vol_metrics.get("realized_vol") if vol_metrics else None
    if vol_s > 65:
        lines.append(f"• Low volatility regime ({rv}% realized) — favorable risk environment")
    elif vol_s < 35:
        lines.append(f"• Elevated volatility ({rv}% realized) — size positions conservatively")
    # Risk factors
    lines.append("")
    lines.append("**Key Risks:**")
    if pe and pe > 40: lines.append("• High valuation leaves little margin for error")
    if rv and rv > 35: lines.append("• High vol means wider stop-losses required")
    dte = info.get("debtToEquity")
    if dte and dte > 150: lines.append(f"• Elevated leverage (D/E {dte:.0f}%)")
    if mom < 40: lines.append("• Momentum headwinds — don't catch a falling knife")
    return "\n".join(lines)


def backtest_momentum(df, lookback=63, hold_period=21):
    """Simple momentum backtest: buy when momentum > 0, hold for N days."""
    if df is None or len(df) < lookback + hold_period + 50:
        return None
    close = df["Close"].values.flatten()
    dates = df.index
    results = []
    for i in range(lookback, len(close) - hold_period):
        momentum = (close[i] / close[i - lookback]) - 1
        future_ret = (close[i + hold_period] / close[i]) - 1
        results.append({
            "date": dates[i],
            "signal": momentum,
            "forward_return": future_ret,
            "long": momentum > 0
        })
    if not results:
        return None
    rdf = pd.DataFrame(results)
    long_rets = rdf[rdf["long"]]["forward_return"]
    short_rets = rdf[~rdf["long"]]["forward_return"]
    # Cumulative returns
    rdf["strategy_ret"] = rdf.apply(
        lambda r: r["forward_return"] if r["long"] else -r["forward_return"], axis=1
    )
    rdf["cum_strategy"] = (1 + rdf["strategy_ret"]).cumprod()
    rdf["cum_buyhold"] = (1 + rdf["forward_return"]).cumprod()
    return {
        "df": rdf,
        "long_avg": round(long_rets.mean() * 100, 2) if len(long_rets) > 0 else 0,
        "short_avg": round(short_rets.mean() * 100, 2) if len(short_rets) > 0 else 0,
        "win_rate": round((long_rets > 0).mean() * 100, 1) if len(long_rets) > 0 else 0,
        "n_trades": len(long_rets),
        "sharpe": round(long_rets.mean() / long_rets.std() * np.sqrt(252/hold_period), 2) if long_rets.std() > 0 else 0
    }


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────

st.markdown("# 📊 QuantLab")
st.markdown("*Quantitative Research & Thesis Generation Platform*")
st.markdown("---")

tab_screener, tab_deep, tab_options, tab_backtest = st.tabs([
    "🔍 SCREENER", "📋 DEEP DIVE", "⚡ OPTIONS LAB", "🧪 BACKTEST"
])


# ═══════════════════════════════════════════════════════════
# TAB 1: SCREENER
# ═══════════════════════════════════════════════════════════
with tab_screener:
    st.markdown("### Universe Screener")
    st.caption("Scan stocks across momentum, fundamentals, technicals, and volatility — ranked by composite thesis score.")
    
    col_cfg1, col_cfg2 = st.columns([3, 1])
    with col_cfg1:
        custom_tickers = st.text_input(
            "Custom tickers (comma-separated) or leave blank for default universe",
            placeholder="e.g. AAPL, MSFT, NVDA, AMD, TSLA"
        )
    with col_cfg2:
        top_n = st.slider("Show top N", 5, 50, 20)
    
    universe = [t.strip().upper() for t in custom_tickers.split(",") if t.strip()] if custom_tickers else DEFAULT_UNIVERSE
    
    if st.button("🚀 Run Screener", type="primary", use_container_width=True):
        results = []
        progress = st.progress(0, text="Scanning universe...")
        
        for i, ticker in enumerate(universe):
            progress.progress((i + 1) / len(universe), text=f"Analyzing {ticker}... ({i+1}/{len(universe)})")
            
            df = fetch_stock_data(ticker, period="1y")
            if df is None or len(df) < 50:
                continue
            
            info = fetch_ticker_info(ticker)
            momentum = compute_momentum_score(df)
            fundamental = compute_fundamental_score(info)
            technicals = compute_technical_signals(df)
            vol_metrics = compute_volatility_metrics(df)
            
            thesis = compute_composite_thesis_score(momentum, fundamental, technicals, vol_metrics)
            
            close_val = df["Close"].values.flatten()[-1]
            
            results.append({
                "Ticker": ticker,
                "Price": round(float(close_val), 2),
                "Composite": thesis["composite"],
                "Confidence": thesis["confidence"],
                "Momentum": round(thesis["components"]["momentum"], 1),
                "Fundamental": round(thesis["components"]["fundamental"], 1),
                "Technical": round(thesis["components"]["technical"], 1),
                "Volatility": round(thesis["components"]["volatility"], 1),
                "RSI": technicals.get("rsi_14"),
                "RealVol%": vol_metrics.get("realized_vol"),
                "Sector": info.get("sector", "N/A"),
                "Name": info.get("shortName", ticker),
            })
        
        progress.empty()
        
        if results:
            rdf = pd.DataFrame(results).sort_values("Composite", ascending=False).head(top_n)
            st.session_state["screener_results"] = rdf
            
            # Summary metrics
            c1, c2, c3, c4 = st.columns(4)
            bullish = len(rdf[rdf["Composite"] >= 65])
            neutral = len(rdf[(rdf["Composite"] >= 45) & (rdf["Composite"] < 65)])
            bearish = len(rdf[rdf["Composite"] < 45])
            high_conf = len(rdf[rdf["Confidence"] == "HIGH"])
            c1.metric("Bullish", bullish)
            c2.metric("Neutral", neutral)
            c3.metric("Bearish", bearish)
            c4.metric("High Confidence", high_conf)
            
            # Color-coded table
            def color_composite(val):
                if val >= 65: return "color: #00ff88"
                elif val >= 45: return "color: #ffaa00"
                else: return "color: #ff4444"
            
            st.dataframe(
                rdf.style.applymap(color_composite, subset=["Composite"]),
                use_container_width=True,
                height=min(700, 35 * len(rdf) + 38)
            )
            
            # Scatter plot: Momentum vs Fundamental, sized by composite
            fig = px.scatter(
                rdf, x="Momentum", y="Fundamental",
                size="Composite", color="Composite",
                hover_name="Ticker", text="Ticker",
                color_continuous_scale=["#ff4444", "#ffaa00", "#00ff88"],
                range_color=[30, 80],
                title="Momentum vs. Fundamental Quality (size = composite score)"
            )
            fig.update_traces(textposition="top center", textfont_size=9)
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0a0a0f",
                plot_bgcolor="#12121a",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data returned. Check tickers or try again.")


# ═══════════════════════════════════════════════════════════
# TAB 2: DEEP DIVE
# ═══════════════════════════════════════════════════════════
with tab_deep:
    st.markdown("### Single Stock Deep Dive")
    st.caption("Full thesis generation with multi-factor scoring, technicals, and risk analysis.")
    
    deep_ticker = st.text_input("Ticker", value="NVDA", key="deep_ticker").upper()
    
    if st.button("📋 Generate Thesis", type="primary", key="deep_btn"):
        with st.spinner(f"Researching {deep_ticker}..."):
            df = fetch_stock_data(deep_ticker, period="2y")
            info = fetch_ticker_info(deep_ticker)
            
            if df is None or len(df) < 50:
                st.error(f"Insufficient data for {deep_ticker}")
            else:
                momentum = compute_momentum_score(df)
                fundamental = compute_fundamental_score(info)
                technicals = compute_technical_signals(df)
                vol_metrics = compute_volatility_metrics(df)
                thesis_result = compute_composite_thesis_score(momentum, fundamental, technicals, vol_metrics)
                
                # Header
                close_val = float(df["Close"].values.flatten()[-1])
                name = info.get("shortName", deep_ticker)
                sector = info.get("sector", "N/A")
                mcap = info.get("marketCap")
                mcap_str = f"${mcap/1e9:.1f}B" if mcap else "N/A"
                
                st.markdown(f"## {name} ({deep_ticker})")
                st.markdown(f"*{sector} · Market Cap: {mcap_str}*")
                
                # Score cards
                score = thesis_result["composite"]
                conf = thesis_result["confidence"]
                score_class = "score-high" if score >= 65 else ("score-mid" if score >= 45 else "score-low")
                
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Composite", f"{score}/100")
                c2.metric("Momentum", f"{thesis_result['components']['momentum']:.0f}")
                c3.metric("Fundamental", f"{thesis_result['components']['fundamental']:.0f}")
                c4.metric("Technical", f"{thesis_result['components']['technical']:.0f}")
                c5.metric("Volatility", f"{thesis_result['components']['volatility']:.0f}")
                
                # Thesis
                st.markdown("---")
                thesis_text = generate_thesis(deep_ticker, info, thesis_result, technicals, vol_metrics)
                st.markdown(f'<div class="thesis-box">{thesis_text}</div>', unsafe_allow_html=True)
                
                # Price chart with SMAs
                st.markdown("---")
                st.markdown("#### Price & Moving Averages")
                fig_price = go.Figure()
                close_series = df["Close"].squeeze()
                fig_price.add_trace(go.Scatter(
                    x=df.index, y=close_series, name="Price",
                    line=dict(color="#ffffff", width=1.5)
                ))
                for sma_w, color in [(20, "#00ff88"), (50, "#ffaa00"), (200, "#ff4444")]:
                    if len(close_series) >= sma_w:
                        sma_vals = close_series.rolling(sma_w).mean()
                        fig_price.add_trace(go.Scatter(
                            x=df.index, y=sma_vals, name=f"SMA {sma_w}",
                            line=dict(color=color, width=1, dash="dot")
                        ))
                fig_price.update_layout(
                    template="plotly_dark", paper_bgcolor="#0a0a0f",
                    plot_bgcolor="#12121a", height=400,
                    legend=dict(orientation="h", y=1.1),
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig_price, use_container_width=True)
                
                # Technical details
                st.markdown("#### Technical Indicators")
                tc1, tc2, tc3, tc4 = st.columns(4)
                tc1.metric("RSI (14)", technicals.get("rsi_14", "N/A"))
                tc2.metric("MACD Hist", technicals.get("macd_hist", "N/A"))
                tc3.metric("ADX", technicals.get("adx", "N/A"))
                tc4.metric("Vol Ratio", technicals.get("vol_ratio", "N/A"))
                
                tc5, tc6, tc7, tc8 = st.columns(4)
                tc5.metric("BB %B", technicals.get("bb_pct", "N/A"))
                tc6.metric("ATR", technicals.get("atr", "N/A"))
                rv = vol_metrics.get("realized_vol", "N/A")
                tc7.metric("Realized Vol", f"{rv}%" if rv != "N/A" else "N/A")
                vr = vol_metrics.get("vol_regime", "N/A")
                tc8.metric("Vol Regime", vr)
                
                # Fundamentals table
                st.markdown("#### Fundamentals")
                fund_data = {
                    "Trailing PE": info.get("trailingPE"),
                    "Forward PE": info.get("forwardPE"),
                    "PEG Ratio": info.get("pegRatio"),
                    "Revenue Growth": f"{info['revenueGrowth']:.1%}" if info.get("revenueGrowth") else None,
                    "Profit Margin": f"{info['profitMargins']:.1%}" if info.get("profitMargins") else None,
                    "ROE": f"{info['returnOnEquity']:.1%}" if info.get("returnOnEquity") else None,
                    "D/E Ratio": info.get("debtToEquity"),
                    "FCF Yield": f"{info['freeCashflow']/info['marketCap']:.1%}" if info.get("freeCashflow") and info.get("marketCap") else None,
                    "Beta": info.get("beta"),
                    "52W High": info.get("fiftyTwoWeekHigh"),
                    "52W Low": info.get("fiftyTwoWeekLow"),
                }
                fund_df = pd.DataFrame([{"Metric": k, "Value": v} for k, v in fund_data.items() if v is not None])
                if not fund_df.empty:
                    st.dataframe(fund_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════
# TAB 3: OPTIONS LAB
# ═══════════════════════════════════════════════════════════
with tab_options:
    st.markdown("### Options Analysis Lab")
    st.caption("Options chain analysis with IV surface, Greeks, and strategy screening.")
    
    opt_ticker = st.text_input("Ticker", value="AAPL", key="opt_ticker").upper()
    
    if st.button("⚡ Load Options Chain", type="primary", key="opt_btn"):
        with st.spinner(f"Fetching options for {opt_ticker}..."):
            chains, expirations = fetch_options_chain(opt_ticker)
            df_stock = fetch_stock_data(opt_ticker, period="6mo")
            
            if not chains:
                st.error(f"No options data for {opt_ticker}")
            else:
                spot = float(df_stock["Close"].values.flatten()[-1]) if df_stock is not None else 0
                st.markdown(f"**{opt_ticker}** — Spot: ${spot:.2f} | Expirations loaded: {len(chains)}")
                
                st.session_state["opt_chains"] = chains
                st.session_state["opt_spot"] = spot
                st.session_state["opt_ticker"] = opt_ticker
                
                # IV Surface
                st.markdown("#### Implied Volatility Surface")
                iv_data = []
                for exp, chain in chains.items():
                    T = max((pd.Timestamp(exp) - pd.Timestamp.now()).days / 365, 0.001)
                    for _, row in chain["calls"].iterrows():
                        strike = row.get("strike", 0)
                        iv_val = row.get("impliedVolatility")
                        last = row.get("lastPrice", 0)
                        if iv_val and iv_val > 0 and strike > 0:
                            moneyness = strike / spot if spot > 0 else 0
                            if 0.7 < moneyness < 1.3:
                                iv_data.append({
                                    "Expiration": exp,
                                    "DTE": int(T * 365),
                                    "Strike": strike,
                                    "Moneyness": round(moneyness, 3),
                                    "IV": round(iv_val * 100, 1),
                                    "Last": last
                                })
                
                if iv_data:
                    iv_df = pd.DataFrame(iv_data)
                    fig_iv = px.scatter(
                        iv_df, x="Moneyness", y="IV", color="DTE",
                        hover_data=["Strike", "Expiration", "Last"],
                        color_continuous_scale="Viridis",
                        title="IV Smile Across Expirations"
                    )
                    fig_iv.update_layout(
                        template="plotly_dark", paper_bgcolor="#0a0a0f",
                        plot_bgcolor="#12121a", height=450,
                        xaxis_title="Moneyness (Strike/Spot)",
                        yaxis_title="Implied Volatility (%)"
                    )
                    st.plotly_chart(fig_iv, use_container_width=True)
                
                # Chain explorer
                st.markdown("#### Chain Explorer")
                sel_exp = st.selectbox("Expiration", list(chains.keys()), key="chain_exp")
                if sel_exp:
                    T = max((pd.Timestamp(sel_exp) - pd.Timestamp.now()).days / 365, 0.001)
                    chain = chains[sel_exp]
                    
                    opt_type = st.radio("Type", ["Calls", "Puts"], horizontal=True, key="opt_type")
                    chain_df = chain["calls"] if opt_type == "Calls" else chain["puts"]
                    
                    # Filter near the money
                    if spot > 0 and "strike" in chain_df.columns:
                        atm_range = chain_df[
                            (chain_df["strike"] >= spot * 0.85) &
                            (chain_df["strike"] <= spot * 1.15)
                        ].copy()
                    else:
                        atm_range = chain_df.copy()
                    
                    # Compute Greeks for each
                    if not atm_range.empty and spot > 0:
                        greeks_list = []
                        ot = "call" if opt_type == "Calls" else "put"
                        for _, row in atm_range.iterrows():
                            K = row.get("strike", 0)
                            iv_val = row.get("impliedVolatility", 0.3)
                            if K > 0 and iv_val > 0:
                                g = compute_greeks(spot, K, T, RISK_FREE_RATE, iv_val, ot)
                                g["strike"] = K
                                g["iv"] = round(iv_val * 100, 1)
                                g["last"] = row.get("lastPrice", 0)
                                g["bid"] = row.get("bid", 0)
                                g["ask"] = row.get("ask", 0)
                                g["volume"] = row.get("volume", 0)
                                g["openInterest"] = row.get("openInterest", 0)
                                greeks_list.append(g)
                        
                        if greeks_list:
                            greeks_df = pd.DataFrame(greeks_list)
                            display_cols = ["strike", "last", "bid", "ask", "iv",
                                          "delta", "gamma", "theta", "vega",
                                          "volume", "openInterest"]
                            available = [c for c in display_cols if c in greeks_df.columns]
                            st.dataframe(greeks_df[available], use_container_width=True, hide_index=True)
                
                # Unusual activity scanner
                st.markdown("#### 🔥 Unusual Activity Scanner")
                unusual = []
                for exp, chain in chains.items():
                    for side, side_name in [(chain["calls"], "CALL"), (chain["puts"], "PUT")]:
                        for _, row in side.iterrows():
                            vol = row.get("volume") or 0
                            oi = row.get("openInterest") or 0
                            if oi > 0 and vol > 0:
                                ratio = vol / oi
                                if ratio > 2 and vol > 100:
                                    unusual.append({
                                        "Exp": exp,
                                        "Type": side_name,
                                        "Strike": row.get("strike"),
                                        "Last": row.get("lastPrice"),
                                        "Vol": int(vol),
                                        "OI": int(oi),
                                        "Vol/OI": round(ratio, 1),
                                        "IV%": round((row.get("impliedVolatility") or 0) * 100, 1)
                                    })
                
                if unusual:
                    unusual_df = pd.DataFrame(unusual).sort_values("Vol/OI", ascending=False).head(15)
                    st.dataframe(unusual_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No unusual activity detected (Vol/OI > 2x with Vol > 100)")


# ═══════════════════════════════════════════════════════════
# TAB 4: BACKTEST
# ═══════════════════════════════════════════════════════════
with tab_backtest:
    st.markdown("### Strategy Backtester")
    st.caption("Test momentum and mean-reversion strategies on historical data.")
    
    bt_col1, bt_col2, bt_col3 = st.columns(3)
    with bt_col1:
        bt_ticker = st.text_input("Ticker", value="SPY", key="bt_ticker").upper()
    with bt_col2:
        bt_lookback = st.selectbox("Lookback (days)", [21, 42, 63, 126, 252], index=2, key="bt_look")
    with bt_col3:
        bt_hold = st.selectbox("Hold period (days)", [5, 10, 21, 42, 63], index=2, key="bt_hold")
    
    if st.button("🧪 Run Backtest", type="primary", key="bt_btn"):
        with st.spinner("Backtesting..."):
            df = fetch_stock_data(bt_ticker, period="5y")
            if df is None:
                st.error("No data")
            else:
                bt_result = backtest_momentum(df, lookback=bt_lookback, hold_period=bt_hold)
                if bt_result is None:
                    st.error("Insufficient data for backtest")
                else:
                    # Metrics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Win Rate", f"{bt_result['win_rate']}%")
                    m2.metric("Avg Long Return", f"{bt_result['long_avg']}%")
                    m3.metric("Avg Short Return", f"{bt_result['short_avg']}%")
                    m4.metric("Sharpe Ratio", bt_result['sharpe'])
                    
                    st.metric("Total Trades", bt_result['n_trades'])
                    
                    # Cumulative returns chart
                    bt_df = bt_result["df"]
                    fig_bt = go.Figure()
                    fig_bt.add_trace(go.Scatter(
                        x=bt_df["date"], y=bt_df["cum_strategy"],
                        name="Momentum Strategy", line=dict(color="#00ff88", width=2)
                    ))
                    fig_bt.add_trace(go.Scatter(
                        x=bt_df["date"], y=bt_df["cum_buyhold"],
                        name="Buy & Hold", line=dict(color="#8888aa", width=1, dash="dot")
                    ))
                    fig_bt.update_layout(
                        title=f"Momentum Strategy vs Buy & Hold — {bt_ticker}",
                        template="plotly_dark", paper_bgcolor="#0a0a0f",
                        plot_bgcolor="#12121a", height=450,
                        legend=dict(orientation="h", y=1.1),
                        yaxis_title="Cumulative Return (1 = start)"
                    )
                    st.plotly_chart(fig_bt, use_container_width=True)
                    
                    # Return distribution
                    fig_dist = go.Figure()
                    long_rets = bt_df[bt_df["long"]]["forward_return"] * 100
                    fig_dist.add_trace(go.Histogram(
                        x=long_rets, nbinsx=40, name="Long Returns",
                        marker_color="#00ff88", opacity=0.7
                    ))
                    fig_dist.add_vline(x=0, line_dash="dash", line_color="#ffffff", opacity=0.5)
                    fig_dist.update_layout(
                        title="Distribution of Forward Returns (Long Signals)",
                        template="plotly_dark", paper_bgcolor="#0a0a0f",
                        plot_bgcolor="#12121a", height=350,
                        xaxis_title="Return (%)", yaxis_title="Count"
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)


# Footer
st.markdown("---")
st.caption("QuantLab — Built with free data (yfinance). Not financial advice. All models are backward-looking and carry no guarantee of future performance.")
