"""
NSE Stock Analysis API — for n8n integration
================================================================================
A single FastAPI service that does ALL the real work (data pull + every
indicator built in our Jupyter session) and exposes it as one clean HTTP
endpoint. n8n calls this URL and gets back a complete JSON analysis for
ANY NSE-listed company — no coding required inside n8n itself.

REQUIREMENTS
------------
    pip install fastapi uvicorn yfinance pandas numpy

RUN LOCALLY (for testing in Jupyter/terminal)
-----------------------------------------------
    uvicorn nse_analysis_api:app --reload --port 8000

Then test in your browser or Jupyter:
    http://127.0.0.1:8000/analyze/RELIANCE

EXPOSING THIS TO n8n
---------------------
- If n8n runs on the SAME machine: use http://127.0.0.1:8000 directly (self-hosted n8n)
- If n8n is in the cloud (n8n.cloud) or on a different machine: you need a
  public URL. Easiest free options:
    1. ngrok (quick, good for testing):  ngrok http 8000
       -> gives you a temporary public URL like https://abc123.ngrok.io
    2. Deploy this file properly on Render.com or Railway.app (free tier)
       for a permanent URL — recommended once you move past testing

USAGE FROM n8n
---------------
In n8n, add an "HTTP Request" node:
    Method: GET
    URL: http://YOUR_URL/analyze/{{ $json.symbol }}
No code needed on the n8n side at all — this endpoint does everything.

NEW IN THIS VERSION
--------------------
    /backtest/{symbol}  Replay a trading rule on past prices (MA cross, RSI, MACD, Bollinger)
    /screen             Scan many stocks and keep the ones that pass your filters
    /risk/{symbol}      Beta vs Nifty 50, volatility, worst fall, Sharpe, ATR stop-loss

    Price history is cached for 15 minutes, which keeps Yahoo Finance requests low.

    streamlit_app.py calls these same functions directly (no separate API server needed),
    so the whole app can be hosted free on Streamlit Community Cloud straight from GitHub.
    Streamlit Cloud runs only the Streamlit app; to keep using the /analyze URL from n8n,
    run this file with uvicorn somewhere else (for example Render's free tier).
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time

app = FastAPI(title="NSE Stock Analysis API")

# Allow any frontend (v0.dev, Lovable, or anywhere else) to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── DATA LAYER: cached price history ──────────────────────────────────────
# Every endpoint used to download 2 years of prices from Yahoo on every call.
# This cache keeps each download for 15 minutes, so opening a watchlist, a
# screener and a backtest in a row costs one download per stock, not ten.
# (It lives in memory, so it resets whenever the host restarts the app.)

CACHE_TTL_SECONDS = 15 * 60
NIFTY_TICKER = "^NSEI"
_HISTORY_CACHE = {}  # {(ticker, period): (time_fetched, DataFrame)}


def _tidy_history(df):
    """Drop empty rows and turn the index into plain dates, so a stock and the
    Nifty 50 line up day-by-day without timezone mismatches."""
    if df is None or df.empty or "Close" not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=["Close"]).copy()
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    return df


def get_history(ticker, period="2y"):
    """Daily prices for one ticker, from the cache when fresh enough."""
    key = (ticker, period)
    hit = _HISTORY_CACHE.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL_SECONDS:
        return hit[1].copy()
    df = _tidy_history(yf.Ticker(ticker).history(period=period, auto_adjust=True))
    if not df.empty:
        _HISTORY_CACHE[key] = (time.time(), df)
    return df.copy()


def get_history_batch(tickers, period="2y"):
    """Prices for many tickers using ONE Yahoo request for everything not
    already cached. Returns {ticker: DataFrame}; a failed ticker maps to an
    empty DataFrame instead of raising."""
    result, missing = {}, []
    for t in tickers:
        hit = _HISTORY_CACHE.get((t, period))
        if hit and time.time() - hit[0] < CACHE_TTL_SECONDS:
            result[t] = hit[1].copy()
        else:
            missing.append(t)

    if len(missing) == 1:
        result[missing[0]] = get_history(missing[0], period)
    elif missing:
        try:
            raw = yf.download(missing, period=period, auto_adjust=True,
                              group_by="ticker", threads=True, progress=False)
        except Exception:
            raw = None
        for t in missing:
            try:
                df = _tidy_history(raw[t])
            except Exception:
                df = pd.DataFrame()
            if not df.empty:
                _HISTORY_CACHE[(t, period)] = (time.time(), df)
            result[t] = df
    return result


# ── INDICATOR FUNCTIONS (same ones built in our Jupyter session today) ────

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_roc(prices, period=12):
    return prices.pct_change(period) * 100

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def calculate_moving_averages(prices, short_window=50, long_window=200):
    ma_short = prices.rolling(short_window).mean()
    ma_long = prices.rolling(long_window).mean()
    return ma_short, ma_long

def calculate_adx(high, low, close, period=14):
    plus_dm = high.diff(); minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0; minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm - minus_dm) < 0] = 0; minus_dm[(minus_dm - plus_dm) < 0] = 0
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def calculate_52w_high_proximity(prices, window=252):
    return (prices / prices.rolling(window).max()) * 100

def calculate_bollinger_bands(prices, window=20, num_std=2):
    middle = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    percent_b = (prices - lower) / (upper - lower)
    bandwidth = (upper - lower) / middle * 100
    return percent_b, bandwidth

def get_yoy_earnings_growth(net_income, tolerance_days=45):
    """Date-matched YoY growth — the corrected version, fixing the bug
    we caught earlier today where fixed-position matching broke on
    stocks with an irregularly reported quarter."""
    if len(net_income) < 2:
        return None, None
    latest_date = net_income.index[0]
    latest_val = net_income.iloc[0]
    target_date = latest_date - pd.DateOffset(months=12)
    date_diffs = (net_income.index - target_date).days.to_series(index=net_income.index).abs()
    closest_idx = date_diffs.idxmin()
    if date_diffs[closest_idx] > tolerance_days:
        return None, None
    base_val = net_income.loc[closest_idx]
    if base_val == 0 or pd.isna(base_val):
        return None, None
    return round(((latest_val / base_val) - 1) * 100, 2), str(closest_idx.date())


def safe_round(val, decimals=2):
    if val is None or pd.isna(val):
        return None
    return round(float(val), decimals)


def get_yoy_revenue_growth(revenue, tolerance_days=45):
    """Same date-matched logic as earnings growth, applied to revenue."""
    return get_yoy_earnings_growth(revenue, tolerance_days)


def get_fundamentals(stock):
    """Extended fundamentals: earnings growth, revenue growth, margins, ROE, debt-to-equity.

    IMPORTANT: this deliberately avoids stock.info — Yahoo Finance blocks/rate-limits
    that specific endpoint for many cloud hosting providers (Render, Railway, Heroku),
    returning an almost-empty dict even though the price-history endpoint works fine.
    Instead, everything here is calculated directly from the income statement and
    balance sheet, which use the endpoint that's actually working for us."""
    fundamentals = {
        "yoy_earnings_growth_pct": None,
        "earnings_base_quarter": None,
        "yoy_revenue_growth_pct": None,
        "revenue_base_quarter": None,
        "operating_margin_pct": None,
        "net_margin_pct": None,
        "roe_pct": None,
        "debt_to_equity": None,
        "market_cap_cr": None,
    }

    latest_revenue = None
    latest_net_income = None
    latest_operating_income = None

    try:
        income_stmt = stock.quarterly_income_stmt
        if income_stmt is not None:
            if "Net Income" in income_stmt.index:
                net_income = income_stmt.loc["Net Income"].dropna()
                growth, base_date = get_yoy_earnings_growth(net_income)
                fundamentals["yoy_earnings_growth_pct"] = growth
                fundamentals["earnings_base_quarter"] = base_date
                if len(net_income) > 0:
                    latest_net_income = net_income.iloc[0]

            if "Total Revenue" in income_stmt.index:
                revenue = income_stmt.loc["Total Revenue"].dropna()
                growth, base_date = get_yoy_revenue_growth(revenue)
                fundamentals["yoy_revenue_growth_pct"] = growth
                fundamentals["revenue_base_quarter"] = base_date
                if len(revenue) > 0:
                    latest_revenue = revenue.iloc[0]

            if "Operating Income" in income_stmt.index:
                op_income = income_stmt.loc["Operating Income"].dropna()
                if len(op_income) > 0:
                    latest_operating_income = op_income.iloc[0]

        # margins calculated directly — same math a real analyst would do by hand
        if latest_revenue and latest_revenue != 0:
            if latest_net_income is not None:
                fundamentals["net_margin_pct"] = safe_round((latest_net_income / latest_revenue) * 100)
            if latest_operating_income is not None:
                fundamentals["operating_margin_pct"] = safe_round((latest_operating_income / latest_revenue) * 100)

    except Exception as e:
        fundamentals["_debug_income_stmt_error"] = str(e)

    try:
        balance_sheet = stock.quarterly_balance_sheet
        if balance_sheet is not None:
            equity = None
            total_debt = None

            for equity_label in ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"]:
                if equity_label in balance_sheet.index:
                    equity_series = balance_sheet.loc[equity_label].dropna()
                    if len(equity_series) > 0:
                        equity = equity_series.iloc[0]
                        break

            for debt_label in ["Total Debt", "Net Debt"]:
                if debt_label in balance_sheet.index:
                    debt_series = balance_sheet.loc[debt_label].dropna()
                    if len(debt_series) > 0:
                        total_debt = debt_series.iloc[0]
                        break

            if equity and equity != 0:
                if total_debt is not None:
                    fundamentals["debt_to_equity"] = safe_round(total_debt / equity)
                # ROE using latest quarter's net income annualized (x4) against latest equity —
                # a simplified TTM-style approximation, standard when only one quarter is available
                if latest_net_income is not None:
                    fundamentals["roe_pct"] = safe_round(((latest_net_income * 4) / equity) * 100)

    except Exception as e:
        fundamentals["_debug_balance_sheet_error"] = str(e)

    # market cap: try the lighter fast_info endpoint, which is less likely to be blocked
    # than the full .info dict — separate try block so a failure here doesn't affect anything above
    try:
        fast_info = stock.fast_info
        if fast_info and fast_info.get("marketCap"):
            fundamentals["market_cap_cr"] = safe_round(fast_info["marketCap"] / 1e7)
    except Exception:
        pass  # market cap is a nice-to-have; not worth failing the request over

    return fundamentals


def get_technical_snapshot(hist):
    """The core technical calculation, factored out so both /analyze and /watchlist can reuse it."""
    close = hist["Close"]
    high = hist["High"]
    low = hist["Low"]

    daily_pct_change = close.pct_change()
    corporate_action_flag = bool((daily_pct_change.abs() > 0.15).any())

    rsi = calculate_rsi(close)
    roc = calculate_roc(close)
    macd_line, signal_line, macd_hist = calculate_macd(close)
    ma50, ma200 = calculate_moving_averages(close)
    adx = calculate_adx(high, low, close)
    high52 = calculate_52w_high_proximity(close)
    percent_b, bandwidth = calculate_bollinger_bands(close)

    ma_trend = None
    if not pd.isna(ma50.iloc[-1]) and not pd.isna(ma200.iloc[-1]):
        ma_trend = "bullish" if ma50.iloc[-1] > ma200.iloc[-1] else "bearish"

    technical = {
        "current_price": safe_round(close.iloc[-1]),
        "rsi_14": safe_round(rsi.iloc[-1]),
        "roc_12": safe_round(roc.iloc[-1]),
        "macd_line": safe_round(macd_line.iloc[-1]),
        "macd_signal": safe_round(signal_line.iloc[-1]),
        "macd_histogram": safe_round(macd_hist.iloc[-1]),
        "ma_50": safe_round(ma50.iloc[-1]),
        "ma_200": safe_round(ma200.iloc[-1]),
        "ma_trend": ma_trend,
        "adx_14": safe_round(adx.iloc[-1]),
        "pct_of_52w_high": safe_round(high52.iloc[-1]),
        "bollinger_percent_b": safe_round(percent_b.iloc[-1], 3),
        "bollinger_bandwidth_pct": safe_round(bandwidth.iloc[-1]),
    }
    return technical, corporate_action_flag


@app.get("/")
def health_check():
    return {"status": "ok", "message": "NSE Stock Analysis API is running"}


@app.get("/analyze/{symbol}")
def analyze_stock(symbol: str):
    """Full technical + fundamental analysis for one NSE-listed company."""
    symbol = symbol.upper().strip()
    ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"

    try:
        stock = yf.Ticker(ticker)
        hist = get_history(ticker, "2y")
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}. Check the symbol is correct.")

        technical, corporate_action_flag = get_technical_snapshot(hist)
        fundamentals = get_fundamentals(stock)

        return {
            "symbol": ticker,
            "company_name": ticker,
            "timestamp": datetime.now().isoformat(),
            "corporate_action_flag": corporate_action_flag,
            "corporate_action_note": (
                "A >15% single-day price move was detected in the last 2 years — "
                "moving-average and Bollinger figures may be distorted by a demerger, "
                "split, or similar event. Verify before trusting the technical signals."
                if corporate_action_flag else None
            ),
            "technical": technical,
            "fundamentals": fundamentals,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing {ticker}: {str(e)}")


@app.get("/chart/{symbol}")
def chart_data(symbol: str, months: int = 6):
    """
    OHLC candlestick data + overlay indicator series for charting.
    Returns one row per trading day: date, OHLCV, RSI, MACD line/signal, MA50, MA200.
    `months` controls how much history to return in the chart (default 6, max 24).
    """
    symbol = symbol.upper().strip()
    ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
    months = min(max(months, 1), 24)

    try:
        stock = yf.Ticker(ticker)
        hist = get_history(ticker, "2y")  # need 2y so MA200 has enough lookback, even though we only return the recent slice
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}.")

        close = hist["Close"]
        rsi = calculate_rsi(close)
        macd_line, signal_line, _ = calculate_macd(close)
        ma50, ma200 = calculate_moving_averages(close)

        trading_days = months * 21  # approx trading days per month
        recent = hist.tail(trading_days)

        rows = []
        for date, row in recent.iterrows():
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": safe_round(row["Open"]),
                "high": safe_round(row["High"]),
                "low": safe_round(row["Low"]),
                "close": safe_round(row["Close"]),
                "volume": int(row["Volume"]) if not pd.isna(row["Volume"]) else None,
                "rsi": safe_round(rsi.loc[date]) if date in rsi.index else None,
                "macd_line": safe_round(macd_line.loc[date]) if date in macd_line.index else None,
                "macd_signal": safe_round(signal_line.loc[date]) if date in signal_line.index else None,
                "ma_50": safe_round(ma50.loc[date]) if date in ma50.index else None,
                "ma_200": safe_round(ma200.loc[date]) if date in ma200.index else None,
            })

        return {"symbol": ticker, "months": months, "data": rows}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chart data for {ticker}: {str(e)}")


@app.get("/watchlist")
def watchlist(symbols: str):
    """
    Lightweight multi-stock snapshot for a watchlist view.
    Pass symbols comma-separated: /watchlist?symbols=RELIANCE,TCS,INFY
    Capped at 10 symbols per request to keep response time reasonable on free hosting.
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:10]
    if not symbol_list:
        raise HTTPException(status_code=400, detail="Provide at least one symbol, e.g. ?symbols=RELIANCE,TCS")

    results = []
    for symbol in symbol_list:
        ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
        try:
            stock = yf.Ticker(ticker)
            hist = get_history(ticker, "2y")
            if hist.empty:
                results.append({"symbol": ticker, "error": "No data found"})
                continue

            technical, corporate_action_flag = get_technical_snapshot(hist)
            results.append({
                "symbol": ticker,
                "company_name": ticker,
                "current_price": technical["current_price"],
                "rsi_14": technical["rsi_14"],
                "macd_histogram": technical["macd_histogram"],
                "ma_trend": technical["ma_trend"],
                "pct_of_52w_high": technical["pct_of_52w_high"],
                "corporate_action_flag": corporate_action_flag,
            })
        except Exception as e:
            results.append({"symbol": ticker, "error": str(e)})

    return {"count": len(results), "stocks": results}


# ══════════════════════════════════════════════════════════════════════════
# NEW FEATURES: backtesting, stock screener, risk & relative strength
#
# Each feature is a plain compute_*() function that the API endpoints AND the
# Streamlit app both call, so the logic exists exactly once. The endpoints at
# the bottom of each section only translate errors into HTTP responses.
# ══════════════════════════════════════════════════════════════════════════

class DataNotFound(Exception):
    """No usable price data came back for a symbol."""


class AnalysisInputError(Exception):
    """The request itself is the problem (bad setting, not enough history)."""


def _to_ticker(symbol):
    symbol = symbol.upper().strip()
    return symbol if symbol.endswith(".NS") or symbol.startswith("^") else f"{symbol}.NS"


def _http_error(e):
    if isinstance(e, DataNotFound):
        return HTTPException(status_code=404, detail=str(e))
    if isinstance(e, AnalysisInputError):
        return HTTPException(status_code=400, detail=str(e))
    return HTTPException(status_code=500, detail=f"Unexpected error: {e}")


def _inr(x):
    """Format a number the Indian way: 145000 -> ₹1,45,000."""
    s = f"{abs(x):.0f}"
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts + [tail])
    return ("-" if x < 0 else "") + "₹" + s


def calculate_atr(high, low, close, period=14):
    """Average True Range: the typical daily price swing, in rupees."""
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()


# ── FEATURE 1: BACKTESTING ────────────────────────────────────────────────
# Replays a trading rule on past prices: "if I had bought and sold by this
# rule, what would ₹1,00,000 have become?"

BACKTEST_STRATEGIES = {
    "ma_cross": "Hold while the fast moving average is above the slow one, otherwise stay in cash",
    "rsi_reversion": "Buy when RSI falls below the buy level, sell when it rises above the sell level",
    "macd_cross": "Hold while the MACD line is above its signal line, otherwise stay in cash",
    "bollinger_bounce": "Buy near the lower Bollinger band, sell once price recovers to the middle band",
}


def _hold_between(entry, exit_):
    """Turn 'enter when A, leave when B' into a 1/0 holding series."""
    state, out = 0, []
    for e, x in zip(entry.values, exit_.values):
        if state == 0 and e:
            state = 1
        elif state == 1 and x:
            state = 0
        out.append(state)
    return pd.Series(out, index=entry.index)


def build_signal(close, strategy, fast=50, slow=200, rsi_buy=30, rsi_sell=70):
    """1 = want to hold the stock at today's close, 0 = want to be in cash."""
    if strategy == "ma_cross":
        if fast >= slow:
            raise AnalysisInputError("The fast moving average must be shorter than the slow one.")
        return (close.rolling(fast).mean() > close.rolling(slow).mean()).astype(int)
    if strategy == "rsi_reversion":
        if rsi_buy >= rsi_sell:
            raise AnalysisInputError("The RSI buy level must be lower than the sell level.")
        rsi = calculate_rsi(close)
        return _hold_between(rsi < rsi_buy, rsi > rsi_sell)
    if strategy == "macd_cross":
        macd_line, signal_line, _ = calculate_macd(close)
        return (macd_line > signal_line).astype(int)
    if strategy == "bollinger_bounce":
        percent_b, _ = calculate_bollinger_bands(close)
        return _hold_between(percent_b < 0.1, percent_b > 0.5)
    raise AnalysisInputError(f"Unknown strategy '{strategy}'. Choose from: {', '.join(BACKTEST_STRATEGIES)}.")


def _extract_trades(close, signal, cost):
    """List each buy/sell pair. Trades happen at the close of the signal day."""
    trades, entry_date, entry_price, prev = [], None, None, 0
    for date, sig in signal.items():
        if sig == 1 and prev == 0:
            entry_date, entry_price = date, float(close.loc[date])
        elif sig == 0 and prev == 1:
            exit_price = float(close.loc[date])
            trades.append({
                "entry_date": str(entry_date.date()), "entry_price": safe_round(entry_price),
                "exit_date": str(date.date()), "exit_price": safe_round(exit_price),
                "days_held": (date - entry_date).days,
                "profit_pct": safe_round(((exit_price / entry_price - 1) - 2 * cost) * 100),
                "status": "closed",
            })
        prev = sig
    if prev == 1 and entry_date is not None:  # still holding at the end: mark to market
        last_date, last_price = close.index[-1], float(close.iloc[-1])
        trades.append({
            "entry_date": str(entry_date.date()), "entry_price": safe_round(entry_price),
            "exit_date": None, "exit_price": safe_round(last_price),
            "days_held": (last_date - entry_date).days,
            "profit_pct": safe_round(((last_price / entry_price - 1) - cost) * 100),
            "status": "open",
        })
    return trades


def _perf_stats(equity, capital, risk_free):
    """Headline numbers for one equity curve (value of the money over time)."""
    equity = equity.dropna()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
    final = float(equity.iloc[-1])
    cagr = (final / capital) ** (1 / years) - 1 if final > 0 else -1.0
    drawdown = equity / equity.cummax() - 1
    daily = equity.pct_change().dropna()
    sharpe = None
    if len(daily) > 2 and daily.std() > 0:
        sharpe = (daily.mean() - risk_free / 252) / daily.std() * np.sqrt(252)
    return {
        "final_value": safe_round(final),
        "total_return_pct": safe_round((final / capital - 1) * 100),
        "cagr_pct": safe_round(cagr * 100),
        "max_drawdown_pct": safe_round(drawdown.min() * 100),
        "sharpe": safe_round(sharpe),
    }


def compute_backtest(symbol, strategy="ma_cross", years=5, capital=100000.0, cost_pct=0.1,
                     fast=50, slow=200, rsi_buy=30, rsi_sell=70, risk_free_pct=6.5,
                     include_nifty=True):
    ticker = _to_ticker(symbol)
    years = int(min(max(years, 1), 15))
    capital = float(capital)
    if capital <= 0:
        raise AnalysisInputError("Starting capital must be above zero.")
    cost = max(float(cost_pct), 0.0) / 100
    risk_free = float(risk_free_pct) / 100

    # Download extra history so the slow indicators (like the 200-day average)
    # are already "warmed up" on the first day of the test window.
    period = "5y" if years <= 3 else "10y" if years <= 8 else "max"
    hist = get_history(ticker, period)
    if hist.empty:
        raise DataNotFound(f"No data found for {ticker}. Check the symbol is correct.")

    close = hist["Close"]
    signal_full = build_signal(close, strategy, int(fast), int(slow), float(rsi_buy), float(rsi_sell))

    start = close.index[-1] - pd.DateOffset(years=years)
    close_w = close[close.index >= start]
    if len(close_w) < 60:
        raise AnalysisInputError(f"Only {len(close_w)} trading days of history are available. Try a shorter test period.")
    signal = signal_full.loc[close_w.index]

    daily_ret = close_w.pct_change().fillna(0.0)
    position = signal.shift(1).fillna(0)                 # a signal at today's close earns TOMORROW's return
    changes = signal.diff().fillna(signal).abs()         # 1 on each buy or sell day
    strat_ret = position * daily_ret - changes * cost    # pay the trading cost on those days
    equity = capital * (1 + strat_ret).cumprod()
    buy_hold = capital * (1 + daily_ret).cumprod()

    nifty_equity = None
    if include_nifty:
        try:
            nh = get_history(NIFTY_TICKER, period)
            if not nh.empty:
                nclose = nh["Close"].reindex(close_w.index).ffill().bfill()
                nifty_equity = capital * (1 + nclose.pct_change().fillna(0.0)).cumprod()
        except Exception:
            nifty_equity = None  # the Nifty comparison is a bonus, never a reason to fail

    trades = _extract_trades(close_w, signal, cost)
    closed = [t for t in trades if t["status"] == "closed"]
    wins = [t for t in closed if t["profit_pct"] is not None and t["profit_pct"] > 0]

    strategy_stats = _perf_stats(equity, capital, risk_free)
    strategy_stats.update({
        "trades": len(trades),
        "closed_trades": len(closed),
        "win_rate_pct": safe_round(len(wins) / len(closed) * 100) if closed else None,
        "avg_trade_pct": safe_round(np.mean([t["profit_pct"] for t in closed])) if closed else None,
        "time_invested_pct": safe_round(position.mean() * 100),
    })
    buy_hold_stats = _perf_stats(buy_hold, capital, risk_free)
    nifty_stats = _perf_stats(nifty_equity, capital, risk_free) if nifty_equity is not None else None

    beat = strategy_stats["final_value"] > buy_hold_stats["final_value"]
    summary = (
        f"{_inr(capital)} would have become {_inr(strategy_stats['final_value'])} with this rule "
        f"versus {_inr(buy_hold_stats['final_value'])} by simply holding. "
        f"The worst fall along the way was {strategy_stats['max_drawdown_pct']}% for the rule "
        f"and {buy_hold_stats['max_drawdown_pct']}% for holding."
    )

    curve = []
    for i, date in enumerate(close_w.index):
        curve.append({
            "date": str(date.date()),
            "strategy": safe_round(equity.iloc[i]),
            "buy_hold": safe_round(buy_hold.iloc[i]),
            "nifty": safe_round(nifty_equity.iloc[i]) if nifty_equity is not None else None,
        })

    return {
        "symbol": ticker,
        "strategy": strategy,
        "description": BACKTEST_STRATEGIES[strategy],
        "params": {"years": years, "capital": capital, "cost_pct": cost_pct, "fast": fast, "slow": slow,
                   "rsi_buy": rsi_buy, "rsi_sell": rsi_sell, "risk_free_pct": risk_free_pct},
        "period": {"start": str(close_w.index[0].date()), "end": str(close_w.index[-1].date()),
                   "trading_days": len(close_w)},
        "summary": summary,
        "beat_buy_and_hold": bool(beat),
        "strategy_stats": strategy_stats,
        "buy_hold_stats": buy_hold_stats,
        "nifty_stats": nifty_stats,
        "trades": trades,
        "equity_curve": curve,
        "assumptions": [
            "Trades happen at the closing price of the day the signal appears; the money earns returns from the next day.",
            f"Each buy or sell costs {cost_pct}% (brokerage, taxes and slippage together). Change cost_pct to test other levels.",
            "Prices are adjusted for splits and dividends. Returns are before income tax.",
            f"Sharpe ratio uses a {risk_free_pct}% annual risk-free rate.",
            "Only today's listed stocks can be tested, so stocks that collapsed or were delisted are missing (survivorship bias).",
            "This is a simulation of the past, not a forecast.",
        ],
    }


@app.get("/backtest/{symbol}")
def backtest(symbol: str, strategy: str = "ma_cross", years: int = 5, capital: float = 100000,
             cost_pct: float = 0.1, fast: int = 50, slow: int = 200,
             rsi_buy: float = 30, rsi_sell: float = 70):
    """
    Replay a trading rule on past prices.
    Example: /backtest/RELIANCE?strategy=ma_cross&fast=50&slow=200&years=5&capital=100000
    Strategies: ma_cross, rsi_reversion, macd_cross, bollinger_bounce
    """
    try:
        return compute_backtest(symbol, strategy, years, capital, cost_pct, fast, slow, rsi_buy, rsi_sell)
    except Exception as e:
        raise _http_error(e)


# ── FEATURE 2: STOCK SCREENER ─────────────────────────────────────────────
# Scans many stocks at once and keeps only those that pass your filters,
# e.g. "RSI below 35 AND long-term trend still bullish".

UNIVERSES = {
    # A starter list of large NSE companies, close to (not guaranteed identical to) the Nifty 50.
    # Index membership changes and tickers get renamed after demergers, so edit this list
    # whenever you like. Symbols that fail to load are reported back in "failed".
    "large_caps": [
        "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE",
        "BAJAJFINSV", "BEL", "BHARTIARTL", "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "GRASIM",
        "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK",
        "INDUSINDBK", "INFY", "ITC", "JSWSTEEL", "KOTAKBANK", "LT", "M&M", "MARUTI", "NESTLEIND",
        "NTPC", "ONGC", "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN", "SUNPHARMA",
        "TATACONSUM", "TATASTEEL", "TCS", "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
    ],
}

MAX_SCREEN_SYMBOLS = 100

SCREEN_SORT_FIELDS = [
    "rsi_14", "pct_of_52w_high", "adx_14", "macd_histogram", "bollinger_percent_b",
    "return_1m_pct", "return_3m_pct", "current_price",
]


def compute_screen(symbols=None, universe="large_caps", rsi_lt=None, rsi_gt=None, ma_trend=None,
                   pct_52w_gt=None, adx_gt=None, macd_hist_positive=False,
                   sort_by="rsi_14", order="asc", limit=20):
    if symbols:
        tickers = list(dict.fromkeys(_to_ticker(s) for s in symbols if s and s.strip()))[:MAX_SCREEN_SYMBOLS]
        universe_name = "custom list"
    elif universe in UNIVERSES:
        tickers = [_to_ticker(s) for s in UNIVERSES[universe]]
        universe_name = universe
    else:
        raise AnalysisInputError(f"Unknown universe '{universe}'. Choose from: {', '.join(UNIVERSES)}, or pass your own symbols.")
    if not tickers:
        raise AnalysisInputError("Provide at least one symbol.")
    if sort_by not in SCREEN_SORT_FIELDS:
        raise AnalysisInputError(f"Cannot sort by '{sort_by}'. Choose from: {', '.join(SCREEN_SORT_FIELDS)}.")
    if ma_trend not in (None, "bullish", "bearish"):
        raise AnalysisInputError("ma_trend must be 'bullish' or 'bearish'.")

    histories = get_history_batch(tickers, "2y")
    rows, failed = [], []
    for t in tickers:
        hist = histories.get(t)
        if hist is None or len(hist) < 210:  # need about 200 days for the 200-day average
            failed.append(t)
            continue
        try:
            technical, flag = get_technical_snapshot(hist)
            close = hist["Close"]
            row = {
                "symbol": t,
                "current_price": technical["current_price"],
                "rsi_14": technical["rsi_14"],
                "ma_trend": technical["ma_trend"],
                "adx_14": technical["adx_14"],
                "pct_of_52w_high": technical["pct_of_52w_high"],
                "macd_histogram": technical["macd_histogram"],
                "bollinger_percent_b": technical["bollinger_percent_b"],
                "return_1m_pct": safe_round((close.iloc[-1] / close.iloc[-22] - 1) * 100),
                "return_3m_pct": safe_round((close.iloc[-1] / close.iloc[-64] - 1) * 100),
                "corporate_action_flag": flag,
            }
        except Exception:
            failed.append(t)
            continue
        rows.append(row)

    def passes(r):
        if rsi_lt is not None and (r["rsi_14"] is None or r["rsi_14"] >= rsi_lt):
            return False
        if rsi_gt is not None and (r["rsi_14"] is None or r["rsi_14"] <= rsi_gt):
            return False
        if ma_trend is not None and r["ma_trend"] != ma_trend:
            return False
        if pct_52w_gt is not None and (r["pct_of_52w_high"] is None or r["pct_of_52w_high"] < pct_52w_gt):
            return False
        if adx_gt is not None and (r["adx_14"] is None or r["adx_14"] < adx_gt):
            return False
        if macd_hist_positive and (r["macd_histogram"] is None or r["macd_histogram"] <= 0):
            return False
        return True

    matched = [r for r in rows if passes(r)]
    has_value = [r for r in matched if r.get(sort_by) is not None]
    no_value = [r for r in matched if r.get(sort_by) is None]
    has_value.sort(key=lambda r: r[sort_by], reverse=(order == "desc"))
    ordered = has_value + no_value

    return {
        "universe": universe_name,
        "scanned": len(rows),
        "matched": len(matched),
        "filters": {"rsi_lt": rsi_lt, "rsi_gt": rsi_gt, "ma_trend": ma_trend, "pct_52w_gt": pct_52w_gt,
                    "adx_gt": adx_gt, "macd_hist_positive": macd_hist_positive},
        "sorted_by": f"{sort_by} ({order})",
        "failed": failed,
        "results": ordered[:max(int(limit), 1)],
    }


@app.get("/screen")
def screen(symbols: str = "", universe: str = "large_caps", rsi_lt: float = None, rsi_gt: float = None,
           ma_trend: str = None, pct_52w_gt: float = None, adx_gt: float = None,
           macd_hist_positive: bool = False, sort_by: str = "rsi_14", order: str = "asc", limit: int = 20):
    """
    Scan a list of stocks and keep the ones that pass every filter you set.
    Example: /screen?rsi_lt=35&ma_trend=bullish&sort_by=rsi_14
    Leave `symbols` empty to scan the built-in large-cap list, or pass your own:
    /screen?symbols=TCS,INFY,WIPRO&pct_52w_gt=90   (use %26 for the & in M&M)
    """
    try:
        sym_list = [s for s in symbols.split(",") if s.strip()] if symbols else None
        return compute_screen(sym_list, universe, rsi_lt, rsi_gt, ma_trend, pct_52w_gt, adx_gt,
                              macd_hist_positive, sort_by, order, limit)
    except Exception as e:
        raise _http_error(e)


# ── FEATURE 3: RISK & RELATIVE STRENGTH ───────────────────────────────────
# How risky is the stock, and is it beating the market? Beta, volatility,
# worst fall, Sharpe, a "bad day" loss estimate, and an ATR-based stop-loss.

def compute_risk(symbol, years=3, risk_free_pct=6.5):
    ticker = _to_ticker(symbol)
    years = int(min(max(years, 1), 10))
    risk_free = float(risk_free_pct) / 100

    period = "5y" if years <= 4 else "10y"
    hist = get_history(ticker, period)
    if hist.empty:
        raise DataNotFound(f"No data found for {ticker}. Check the symbol is correct.")
    nifty = get_history(NIFTY_TICKER, period)
    if nifty.empty:
        raise DataNotFound("Could not load Nifty 50 data (^NSEI) to compare against. Try again in a minute.")

    cutoff = hist.index[-1] - pd.DateOffset(years=years)
    stock_close = hist["Close"][hist.index >= cutoff]
    both = pd.concat([stock_close.rename("stock"), nifty["Close"].rename("nifty")], axis=1, join="inner").dropna()
    if len(both) < 60:
        raise AnalysisInputError("Not enough overlapping history with the Nifty 50. Try a longer period or another stock.")

    rets = both.pct_change().dropna()
    beta = rets["stock"].cov(rets["nifty"]) / rets["nifty"].var()
    corr = rets["stock"].corr(rets["nifty"])
    vol = rets["stock"].std() * np.sqrt(252) * 100
    nifty_vol = rets["nifty"].std() * np.sqrt(252) * 100

    curve = both["stock"]
    drawdown = curve / curve.cummax() - 1
    trough = drawdown.idxmin()
    peak = curve.loc[:trough].idxmax()
    sharpe = None
    if rets["stock"].std() > 0:
        sharpe = (rets["stock"].mean() - risk_free / 252) / rets["stock"].std() * np.sqrt(252)
    var_95 = -np.percentile(rets["stock"], 5) * 100  # on the worst 1-in-20 days, the loss was at least this

    relative = {}
    for label, n in {"1m": 21, "3m": 63, "6m": 126, "1y": 252}.items():
        if len(both) > n:
            s = both["stock"].iloc[-1] / both["stock"].iloc[-1 - n] - 1
            m = both["nifty"].iloc[-1] / both["nifty"].iloc[-1 - n] - 1
            relative[label] = {"stock_pct": safe_round(s * 100), "nifty_pct": safe_round(m * 100),
                               "outperformance_pts": safe_round((s - m) * 100)}
        else:
            relative[label] = None

    atr = calculate_atr(hist["High"], hist["Low"], hist["Close"]).iloc[-1]
    price = float(hist["Close"].iloc[-1])
    atr_block = None
    if not pd.isna(atr):
        atr_block = {
            "price": safe_round(price),
            "atr_14": safe_round(atr),
            "atr_pct_of_price": safe_round(atr / price * 100),
            "stop_loss_2x_atr": safe_round(price - 2 * atr),
            "stop_distance_pct": safe_round(2 * atr / price * 100),
        }

    if beta > 1.2:
        beta_note = "Tends to move more than the Nifty 50: bigger gains in rallies, bigger falls in dips."
    elif beta < 0.8:
        beta_note = "Tends to move less than the Nifty 50: calmer, but it may lag in strong rallies."
    else:
        beta_note = "Tends to move roughly in line with the Nifty 50."

    return {
        "symbol": ticker,
        "period": {"start": str(both.index[0].date()), "end": str(both.index[-1].date()), "years": years},
        "beta": safe_round(beta),
        "beta_note": beta_note,
        "correlation_with_nifty": safe_round(corr),
        "annual_volatility_pct": safe_round(vol),
        "nifty_annual_volatility_pct": safe_round(nifty_vol),
        "max_drawdown_pct": safe_round(drawdown.min() * 100),
        "max_drawdown_peak_date": str(peak.date()),
        "max_drawdown_trough_date": str(trough.date()),
        "sharpe": safe_round(sharpe),
        "bad_day_loss_pct": safe_round(var_95),
        "bad_day_note": f"On roughly 1 trading day in 20, the stock fell {safe_round(var_95)}% or more. "
                        f"That is about ₹{safe_round(var_95)} lost on every ₹100 held.",
        "relative_performance": relative,
        "atr": atr_block,
    }


@app.get("/risk/{symbol}")
def risk(symbol: str, years: int = 3, risk_free_pct: float = 6.5):
    """
    Risk and relative strength versus the Nifty 50.
    Example: /risk/TCS?years=3
    """
    try:
        return compute_risk(symbol, years, risk_free_pct)
    except Exception as e:
        raise _http_error(e)
