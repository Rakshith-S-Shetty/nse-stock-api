"""
NSE terminal: Streamlit front end
=================================
Four views on top of nse_analysis_api.py: stock analysis, backtest, screener, risk.

It imports the analysis functions directly, so no separate API server is needed.
That is what lets Streamlit Community Cloud host the whole app for free:

    1. Push this repo to GitHub (streamlit_app.py, nse_analysis_api.py, requirements.txt).
    2. Go to share.streamlit.io, click "Create app", pick the repo and branch.
    3. Set the main file path to streamlit_app.py and click Deploy.

Run it locally with:   streamlit run streamlit_app.py
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fastapi import HTTPException

import nse_analysis_api as nse

st.set_page_config(page_title="NSE terminal", layout="wide")

STRATEGY_LABELS = {
    "ma_cross": "Moving average crossover",
    "rsi_reversion": "RSI buy low, sell high",
    "macd_cross": "MACD crossover",
    "bollinger_bounce": "Bollinger band bounce",
}
SORT_LABELS = {
    "rsi_14": "RSI",
    "pct_of_52w_high": "Closeness to 52-week high",
    "adx_14": "Trend strength (ADX)",
    "macd_histogram": "MACD histogram",
    "bollinger_percent_b": "Bollinger %B",
    "return_1m_pct": "1-month return",
    "return_3m_pct": "3-month return",
    "current_price": "Price",
}
FUNDAMENTAL_LABELS = {
    "yoy_earnings_growth_pct": "Earnings growth, year on year (%)",
    "yoy_revenue_growth_pct": "Revenue growth, year on year (%)",
    "operating_margin_pct": "Operating margin (%)",
    "net_margin_pct": "Net margin (%)",
    "roe_pct": "Return on equity (%)",
    "debt_to_equity": "Debt to equity",
    "market_cap_cr": "Market cap (₹ crore)",
}


# ── cached wrappers: one Yahoo download serves every click for 15 minutes ──

@st.cache_data(ttl=900, show_spinner=False)
def run_analysis(symbol):
    return nse.analyze_stock(symbol)


@st.cache_data(ttl=900, show_spinner=False)
def run_chart(symbol, months):
    return nse.chart_data(symbol, months)


@st.cache_data(ttl=900, show_spinner=False)
def run_backtest(symbol, strategy, years, capital, cost_pct, fast, slow, rsi_buy, rsi_sell):
    return nse.compute_backtest(symbol, strategy, years, capital, cost_pct, fast, slow, rsi_buy, rsi_sell)


@st.cache_data(ttl=900, show_spinner=False)
def run_screen(symbols, rsi_lt, rsi_gt, ma_trend, pct_52w_gt, adx_gt, macd_pos, sort_by, order, limit):
    return nse.compute_screen(list(symbols) if symbols else None, "large_caps", rsi_lt, rsi_gt, ma_trend,
                              pct_52w_gt, adx_gt, macd_pos, sort_by, order, limit)


@st.cache_data(ttl=900, show_spinner=False)
def run_risk(symbol, years):
    return nse.compute_risk(symbol, years)


def call(fn, *args):
    """Run a cached function and return (data, error_message)."""
    try:
        return fn(*args), None
    except HTTPException as e:
        return None, str(e.detail)
    except (nse.DataNotFound, nse.AnalysisInputError) as e:
        return None, str(e)
    except Exception as e:
        return None, f"Something went wrong while loading data: {e}. Yahoo Finance may be busy; try again in a minute."


def fmt(value, suffix="", digits=2):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:,.{digits}f}{suffix}"
    return f"{value}{suffix}"


# ── view 1: stock analysis ────────────────────────────────────────────────

def show_analysis(symbol):
    data, err = call(run_analysis, symbol)
    if err:
        st.error(err)
        return
    if data["corporate_action_flag"]:
        st.warning(data["corporate_action_note"])

    t, f = data["technical"], data["fundamentals"]
    row1 = st.columns(4)
    row1[0].metric("Price (₹)", fmt(t["current_price"]))
    row1[1].metric("RSI (14)", fmt(t["rsi_14"]))
    row1[2].metric("Long-term trend", (t["ma_trend"] or "n/a").capitalize(), help="Bullish when the 50-day average is above the 200-day average.")
    row1[3].metric("Trend strength (ADX)", fmt(t["adx_14"]), help="Above about 25 suggests a strong trend, in either direction.")
    row2 = st.columns(4)
    row2[0].metric("Price vs 52-week high", fmt(t["pct_of_52w_high"], "%"))
    row2[1].metric("MACD histogram", fmt(t["macd_histogram"]))
    row2[2].metric("Bollinger %B", fmt(t["bollinger_percent_b"], digits=3), help="Below 0 is under the lower band; above 1 is over the upper band.")
    row2[3].metric("Market cap (₹ crore)", fmt(f.get("market_cap_cr"), digits=0))

    months = st.slider("Chart history (months)", 3, 24, 6)
    chart, err = call(run_chart, symbol, months)
    if err:
        st.error(err)
    else:
        df = pd.DataFrame(chart["data"])
        fig = go.Figure(go.Candlestick(x=df["date"], open=df["open"], high=df["high"], low=df["low"],
                                       close=df["close"], name="Price"))
        fig.add_trace(go.Scatter(x=df["date"], y=df["ma_50"], name="50-day average", line=dict(width=1.5)))
        fig.add_trace(go.Scatter(x=df["date"], y=df["ma_200"], name="200-day average", line=dict(width=1.5)))
        fig.update_layout(height=460, margin=dict(l=0, r=0, t=10, b=0), xaxis_rangeslider_visible=False,
                          yaxis_title="₹", legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig)

    st.subheader("Fundamentals")
    rows = [{"Measure": label, "Value": fmt(f.get(key))} for key, label in FUNDAMENTAL_LABELS.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Figures come from the latest quarterly statements on Yahoo Finance. Return on equity annualises one quarter, so treat it as a rough guide.")


# ── view 2: backtest ──────────────────────────────────────────────────────

def show_backtest(symbol):
    st.write("Test a trading rule on past prices. The result shows what your money would have become, and how bumpy the ride was.")
    with st.form("backtest_form"):
        c1, c2, c3 = st.columns(3)
        strategy = c1.selectbox("Rule", list(STRATEGY_LABELS), format_func=STRATEGY_LABELS.get)
        years = c2.slider("History (years)", 1, 10, 5)
        capital = c3.number_input("Starting money (₹)", min_value=1000, value=100000, step=10000)
        c4, c5, c6, c7, c8 = st.columns(5)
        cost = c4.number_input("Cost per trade (%)", min_value=0.0, max_value=2.0, value=0.1, step=0.05,
                               help="Brokerage, taxes and slippage together, charged on every buy and sell.")
        fast = c5.number_input("Fast average (days)", min_value=5, max_value=100, value=50, help="Used by the moving average crossover.")
        slow = c6.number_input("Slow average (days)", min_value=20, max_value=300, value=200, help="Used by the moving average crossover.")
        rsi_buy = c7.number_input("Buy below RSI", min_value=5, max_value=50, value=30, help="Used by the RSI rule.")
        rsi_sell = c8.number_input("Sell above RSI", min_value=50, max_value=95, value=70, help="Used by the RSI rule.")
        submitted = st.form_submit_button("Run backtest", type="primary")

    if submitted:
        with st.spinner("Replaying the rule on past prices..."):
            data, err = call(run_backtest, symbol, strategy, years, float(capital), cost,
                             int(fast), int(slow), float(rsi_buy), float(rsi_sell))
        st.session_state["backtest"] = {"data": data, "err": err}

    result = st.session_state.get("backtest")
    if not result:
        return
    if result["err"]:
        st.error(result["err"])
        return

    d = result["data"]
    st.subheader(f"{d['symbol']}: {STRATEGY_LABELS[d['strategy']]}")
    (st.success if d["beat_buy_and_hold"] else st.info)(d["summary"])

    names = [("This rule", d["strategy_stats"]), ("Buy and hold", d["buy_hold_stats"]), ("Nifty 50", d["nifty_stats"])]
    table = [{
        "": name,
        "Final value (₹)": fmt(s["final_value"], digits=0),
        "Total return (%)": fmt(s["total_return_pct"]),
        "Yearly return (%)": fmt(s["cagr_pct"]),
        "Worst fall (%)": fmt(s["max_drawdown_pct"]),
        "Sharpe ratio": fmt(s["sharpe"]),
    } for name, s in names if s]
    st.dataframe(pd.DataFrame(table), hide_index=True)

    ss = d["strategy_stats"]
    m = st.columns(4)
    m[0].metric("Trades", ss["trades"])
    m[1].metric("Winning trades", fmt(ss["win_rate_pct"], "%", 0), help="Share of completed trades that made money.")
    m[2].metric("Average trade", fmt(ss["avg_trade_pct"], "%"))
    m[3].metric("Time invested", fmt(ss["time_invested_pct"], "%", 0), help="Share of days the money was in the stock rather than cash.")

    curve = pd.DataFrame(d["equity_curve"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["date"], y=curve["strategy"], name="This rule", line=dict(width=2.5)))
    fig.add_trace(go.Scatter(x=curve["date"], y=curve["buy_hold"], name="Buy and hold", line=dict(width=1.5)))
    if curve["nifty"].notna().any():
        fig.add_trace(go.Scatter(x=curve["date"], y=curve["nifty"], name="Nifty 50", line=dict(width=1.5, dash="dot")))
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Value of your money (₹)",
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig)

    with st.expander(f"All trades ({len(d['trades'])})"):
        if d["trades"]:
            st.dataframe(pd.DataFrame(d["trades"]), hide_index=True)
        else:
            st.write("The rule never triggered a trade in this period.")
    with st.expander("How this was calculated"):
        for line in d["assumptions"]:
            st.write(f"- {line}")


# ── view 3: screener ──────────────────────────────────────────────────────

def show_screener():
    st.write("Scan many stocks at once and keep only those that pass every filter you set.")
    with st.form("screen_form"):
        source = st.radio("Stocks to scan", ["Built-in large-cap list", "My own symbols"], horizontal=True)
        own = st.text_area("Your symbols, separated by commas", placeholder="TCS, INFY, WIPRO, HDFCBANK",
                           help="Only used when 'My own symbols' is selected. Up to 100 symbols.")
        c1, c2, c3 = st.columns(3)
        rsi_range = c1.slider("RSI between", 0, 100, (0, 100), help="RSI below 30 is often called oversold; above 70, overbought.")
        trend = c2.selectbox("Long-term trend", ["Any", "bullish", "bearish"])
        near_high = c3.slider("At least this close to the 52-week high (%)", 0, 100, 0)
        c4, c5, c6 = st.columns(3)
        adx_min = c4.slider("Trend strength (ADX) at least", 0, 60, 0)
        macd_pos = c5.checkbox("MACD histogram above zero")
        limit = c6.number_input("Show up to", min_value=5, max_value=100, value=20, step=5)
        c7, c8 = st.columns(2)
        sort_by = c7.selectbox("Sort by", list(SORT_LABELS), format_func=SORT_LABELS.get)
        order = c8.radio("Order", ["Lowest first", "Highest first"], horizontal=True)
        submitted = st.form_submit_button("Scan", type="primary")

    if submitted:
        symbols = tuple(s.strip() for s in own.split(",") if s.strip()) if source == "My own symbols" else ()
        if source == "My own symbols" and not symbols:
            st.session_state["screen"] = {"data": None, "err": "Type at least one symbol, or switch to the built-in list."}
        else:
            with st.spinner("Scanning stocks. The first scan can take up to a minute..."):
                data, err = call(
                    run_screen, symbols,
                    rsi_range[1] if rsi_range[1] < 100 else None,
                    rsi_range[0] if rsi_range[0] > 0 else None,
                    None if trend == "Any" else trend,
                    near_high if near_high > 0 else None,
                    adx_min if adx_min > 0 else None,
                    macd_pos, sort_by, "asc" if order == "Lowest first" else "desc", int(limit))
            st.session_state["screen"] = {"data": data, "err": err}

    result = st.session_state.get("screen")
    if not result:
        return
    if result["err"]:
        st.error(result["err"])
        return

    d = result["data"]
    st.write(f"Scanned {d['scanned']} stocks, {d['matched']} passed your filters.")
    if d["results"]:
        df = pd.DataFrame(d["results"]).rename(columns={
            "symbol": "Symbol", "current_price": "Price (₹)", "rsi_14": "RSI", "ma_trend": "Trend",
            "adx_14": "ADX", "pct_of_52w_high": "% of 52-week high", "macd_histogram": "MACD histogram",
            "bollinger_percent_b": "Bollinger %B", "return_1m_pct": "1-month (%)", "return_3m_pct": "3-month (%)",
            "corporate_action_flag": "Check for split or demerger"})
        st.dataframe(df, hide_index=True)
    else:
        st.info("No stock passed every filter. Loosen one or two filters and scan again.")
    if d["failed"]:
        with st.expander(f"{len(d['failed'])} symbols could not be loaded"):
            st.write(", ".join(d["failed"]))
            st.caption("Usually a renamed ticker, a recent listing with under a year of history, or Yahoo Finance being busy.")
    st.caption("The built-in list is a starting point close to the Nifty 50, not the official index. Edit UNIVERSES in nse_analysis_api.py to change it.")


# ── view 4: risk and relative strength ────────────────────────────────────

def show_risk(symbol):
    years = st.slider("Look-back period (years)", 1, 5, 3)
    d, err = call(run_risk, symbol, years)
    if err:
        st.error(err)
        return

    row1 = st.columns(4)
    row1[0].metric("Beta vs Nifty 50", fmt(d["beta"]), help="1.0 means it moves with the market. 1.5 means it tends to move 1.5 times as much.")
    row1[1].metric("Yearly volatility", fmt(d["annual_volatility_pct"], "%"), help=f"The Nifty 50 was {fmt(d['nifty_annual_volatility_pct'], '%')} over the same period.")
    row1[2].metric("Worst fall", fmt(d["max_drawdown_pct"], "%"), help=f"From {d['max_drawdown_peak_date']} to {d['max_drawdown_trough_date']}.")
    row1[3].metric("Sharpe ratio", fmt(d["sharpe"]), help="Return earned per unit of risk. Higher is better.")
    st.write(d["beta_note"])
    st.write(d["bad_day_note"])

    st.subheader("Against the Nifty 50")
    rel = {k: v for k, v in d["relative_performance"].items() if v}
    if rel:
        labels = {"1m": "1 month", "3m": "3 months", "6m": "6 months", "1y": "1 year"}
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[labels[k] for k in rel], y=[v["stock_pct"] for v in rel.values()], name=d["symbol"]))
        fig.add_trace(go.Bar(x=[labels[k] for k in rel], y=[v["nifty_pct"] for v in rel.values()], name="Nifty 50"))
        fig.update_layout(barmode="group", height=340, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Return (%)",
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig)
        latest = list(rel.items())[-1]
        gap = latest[1]["outperformance_pts"]
        word = "ahead of" if gap >= 0 else "behind"
        st.write(f"Over {labels[latest[0]]}, the stock is {abs(gap):.1f} percentage points {word} the Nifty 50.")

    atr = d["atr"]
    if atr:
        st.subheader("Stop-loss guide")
        a = st.columns(3)
        a[0].metric("Typical daily swing (ATR)", f"₹{atr['atr_14']:,.2f}", help=f"{atr['atr_pct_of_price']}% of the price.")
        a[1].metric("Stop-loss at 2 × ATR", f"₹{atr['stop_loss_2x_atr']:,.2f}")
        a[2].metric("Distance from price", fmt(atr["stop_distance_pct"], "%"))
        st.caption("A stop two ATRs below the price sits outside a normal day's wobble, so ordinary noise is less likely to trigger it.")


# ── page ──────────────────────────────────────────────────────────────────

st.title("NSE terminal")
st.caption("Price data comes from Yahoo Finance and can be delayed. This is an educational tool, not investment advice.")

view = st.sidebar.radio("View", ["Stock analysis", "Backtest", "Screener", "Risk"])
if view != "Screener":
    symbol = st.sidebar.text_input("NSE symbol", value="RELIANCE", help="For example RELIANCE, TCS, HDFCBANK.").strip().upper()
    if not symbol:
        st.info("Type an NSE symbol in the sidebar to begin.")
        st.stop()

if view == "Stock analysis":
    show_analysis(symbol)
elif view == "Backtest":
    show_backtest(symbol)
elif view == "Screener":
    show_screener()
else:
    show_risk(symbol)
