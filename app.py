import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data
import nlp
import theme
from catalog import BENCHMARKS, UNIVERSES
from engine import RULES, drawdown, metrics, run_strategy, start_year_sensitivity

st.set_page_config(page_title="Backtest Lab", layout="wide")

try:
    theme.inject()
except Exception:
    pass

MIN_START = 2005
PICK_RULES = [k for k in RULES if k != "random"]
DEFAULTS = dict(universe="large_cap", nyse_n=250, custom_tickers="AAPL, NVDA, MSFT, TSLA, AMZN",
                bench_key="sp500", custom_bench="QQQ", start_year=2012, rule="prior_year_winner",
                top_n=1, initial=10_000, annual_contrib=0.0, stop_pct=15, trailing=False, after_stop="cash",
                cost_bps=5, tax_pct=0, nl_text="", basket_tickers="AAPL, MSFT, GOOGL, NVDA")
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
S = st.session_state


def pct(x):
    return f"{x:.1%}"


def get_api_key():
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY")


def apply_nl():
    text = S.get("nl_text", "").strip()
    if not text:
        return
    if S.get("nl_calls", 0) >= 10:
        S["nl_msg"] = ("error", "Scenario limit reached for this session.", "")
        return
    S["nl_calls"] = S.get("nl_calls", 0) + 1
    try:
        out = nlp.parse_question(text, get_api_key())
        for k, v in out["widgets"].items():
            S[k] = v
        S["nl_msg"] = ("ok", out["explanation"], out["unsupported"])
    except Exception as e:
        S["nl_msg"] = ("error", f"Couldn't read that question: {e}", "")


# ---------------- sidebar ----------------
sb = st.sidebar
sb.subheader("Universe")
sb.selectbox("Stocks to pick from", list(UNIVERSES), format_func=UNIVERSES.get, key="universe")
if S["universe"] == "custom":
    sb.text_area("Tickers (comma-separated)", key="custom_tickers")
elif S["universe"] == "nyse":
    sb.select_slider("Most-traded NYSE stocks", [100, 250, 500, 1000], key="nyse_n")

sb.subheader("Benchmark")
sb.selectbox("Compare against", list(BENCHMARKS), format_func=lambda k: BENCHMARKS[k]["label"], key="bench_key")
if S["bench_key"] == "custom":
    sb.text_input("Ticker", key="custom_bench")

sb.subheader("Pick rule / Custom Fund")
sb.selectbox("How to choose stocks each January", PICK_RULES, format_func=RULES.get, key="rule")

custom_weights = None
if S["rule"] == "custom_basket":
    sb.markdown("---")
    sb.markdown("### Custom Mutual Fund Weights")
    basket_input = sb.text_input("Basket Tickers (comma-separated)", key="basket_tickers")
    b_list = [t.strip().upper() for t in basket_input.split(",") if t.strip()]
    custom_weights = {}
    for t in b_list:
        custom_weights[t] = sb.slider(f"Weight: {t}", 0.0, 1.0, 1.0 / len(b_list) if b_list else 1.0, 0.05)
    sb.markdown("---")

if S["rule"] != "custom_basket":
    sb.slider("Number of stocks to buy", 1, 10, key="top_n")

sb.slider("First buy year", MIN_START, 2025, key="start_year")
sb.number_input("Starting amount ($)", 100, 10_000_000, step=500, key="initial")
sb.number_input("Yearly Contribution / DCA ($)", 0, 100_000, step=500, key="annual_contrib")

sb.subheader("Risk rule")
sb.slider("Sell if down X% (0 = no stop-loss)", 0, 50, key="stop_pct")
sb.checkbox("Trailing (measured from peak)", key="trailing")
sb.radio("After a sell, hold…", ["cash", "benchmark"], key="after_stop",
         format_func=lambda x: "Cash" if x == "cash" else "The benchmark until next January")

sb.subheader("Costs and taxes")
sb.slider("Trading cost (basis points)", 0, 50, key="cost_bps")
sb.slider("Tax on realized gains (%)", 0, 50, key="tax_pct")

# ---------------- main execution with error handling ----------------
try:
    if S["universe"] == "nyse":
        with st.spinner("Ranking NYSE stocks by trading volume…"):
            universe_list = data.get_universe("nyse", "", S["nyse_n"])
    else:
        universe_list = data.get_universe(S["universe"], S["custom_tickers"], S["nyse_n"])

    if S["rule"] == "custom_basket" and custom_weights:
        universe_list = list(set(universe_list + list(custom_weights.keys())))

    if not universe_list:
        st.error("Couldn't build the stock universe.")
        st.stop()

    stock_prices = data.load_many(universe_list, "Downloading price history")
    if stock_prices.empty:
        st.error("No price data returned from Yahoo Finance.")
        st.stop()

    bench, bench_series = data.build_benchmark(S["bench_key"], S["custom_bench"], stock_prices)
    if bench_series is None or bench_series.dropna().empty:
        st.error(f"No data for benchmark {bench}.")
        st.stop()

    prices = stock_prices.drop(columns=[bench], errors="ignore")
    prices[bench] = bench_series.reindex(prices.index).ffill()

    fv = prices[bench].first_valid_index()
    min_ok = max(MIN_START, fv.year + (0 if fv.dayofyear <= 10 else 1))
    start_year = max(S["start_year"], min_ok)

    params = dict(rule=S["rule"], top_n=S["top_n"], stop_pct=S["stop_pct"], trailing=S["trailing"],
                  after_stop=S["after_stop"], cost_bps=S["cost_bps"], tax_rate=S["tax_pct"] / 100,
                  initial=S["initial"], annual_contribution=S["annual_contrib"], custom_weights=custom_weights)
    initial = S["initial"]

    res = run_strategy(prices, bench, start_year, **params)
    if res is None:
        st.error("Not enough data for this setup. Try a later start year.")
        st.stop()
        
    eq, be, yrs = res["equity"], res["bench"], res["years"]
    ms, mb = metrics(eq), metrics(be)

    st.title("Portfolio Backtest Results")
    st.metric("Final Portfolio Value", f"${ms['Final value']:,.2f}", f"{ms['CAGR']*100:.1f}% CAGR")
    
    # Simple interactive chart
    fig = go.Figure()
    fig.add_scatter(x=eq.index, y=eq, name="Strategy")
    fig.add_scatter(x=be.index, y=be, name=bench, line=dict(dash='dash'))
    fig.update_layout(height=400, hovermode="x unified", yaxis_tickprefix="$")
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"An error occurred while running the app: {e}")
