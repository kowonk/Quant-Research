# 📊 QuantLab — Quantitative Research & Thesis Generation Platform

A full quant research platform built on free data. Screens stocks, scores them across momentum/fundamentals/technicals/volatility, generates investment theses, analyzes options chains, and backtests strategies.

## Features

- **Universe Screener** — Scan 100 stocks across four factor dimensions, ranked by composite thesis score
- **Deep Dive** — Single-stock thesis generation with full technical/fundamental breakdown
- **Options Lab** — IV surface visualization, Greeks computation, unusual activity scanner
- **Backtester** — Momentum strategy backtesting with cumulative return curves and Sharpe ratios

## Quantitative Methods Used

| Category | Methods |
|----------|---------|
| Momentum | Multi-period ROC, cross-sectional ranking, sigmoid normalization |
| Mean Reversion | Z-score relative to rolling SMA, Bollinger Band %B |
| Volatility | Realized vol, vol regime ratio, vol-of-vol |
| Fundamentals | PE, revenue growth, margins, ROE, D/E, FCF yield composite |
| Technicals | RSI, MACD, ADX, Bollinger Bands, SMA crossovers, volume ratio |
| Options | Black-Scholes pricing, Newton-Raphson IV solver, full Greeks |
| Backtesting | Walk-forward momentum signal, cumulative returns, Sharpe ratio |

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/quant-research-platform.git
cd quant-research-platform
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `app.py` as the main file
5. Deploy

## Data Sources

All free:
- **yfinance** — OHLCV, fundamentals, options chains
- **ta** — Technical indicators (RSI, MACD, ADX, Bollinger Bands, ATR)

## Disclaimer

This is a research tool, not financial advice. All models are backward-looking. Past performance does not guarantee future results. Use at your own risk.
