import os
from io import StringIO

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

import nlp
from engine import RULES, drawdown, metrics, run_strategy, start_year_sensitivity

st.set_page_config(page_title="Backtest Lab", page_icon="📈", layout="wide")

LOAD_FROM, MIN_START = "2004-01-01", 2005
LARGE_CAPS = ("AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO JPM V UNH XOM JNJ WMT PG "
              "MA HD COST NFLX AMD CRM ORCL ADBE KO PEP LLY ABBV MRK CVX BAC").split()
UNIVERSES = ["Large-cap sample (30)", "S&P 500 (current members)", "Custom"]
UNIV_MAP = dict(zip(["large_cap", "sp500", "custom"], UNIVERSES))
PICK_RULES = [k for k in RULES if k != "random"]

# Widget defaults live in session_state so the natural-language box can overwrite them.
DEFAULTS = dict(universe=UNIVERSES[0], custom_tickers="AAPL, NVDA, MSFT, TSLA, AMZN",
                bench="SPY", start_year=2012, rule="prior_year_winner", top_n=1,
                initial=10_000, stop_pct=15, trailing=False, after_stop="cash",
                cost_bps=5, tax_pct=0, nl_text="")
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


# ---------------- helpers ----------------
def pct(x):
    return f"{x:.1%}"


def get_api_key():
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY")


def apply_nl():
    """Button callback: runs BEFORE the script reruns, so it may set widget state."""
    text = st.session_state.get("nl_text", "").strip()
    if not text:
        return
    try:
        out = nlp.parse_question(text, get_api_key())
    except Exception as e:
        st.session_state["nl_msg"] = ("error", f"Couldn't read that question: {e}", "")
        return
    w = out["widgets"]
    if "universe" in w:
        w["universe"] = UNIV_MAP[w["universe"]]
    for k, v in w.items():
        st.session_state[k] = v
    st.session_state["nl_msg"] = ("ok", out["explanation"], out["unsupported"])


@st.cache_data(ttl=86_400, show_spinner=False)
def sp500_tickers():
    html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                        headers={"User-Agent": "Mozilla/5.0"}, timeout=20).text
    return [s.replace(".", "-") for s in pd.read_html(StringIO(html))[0]["Symbol"]]


@st.cache_data(ttl=3_600, show_spinner=False)
def load_prices(tickers: tuple):
    df = yf.download(list(tickers), start=LOAD_FROM, auto_adjust=True,
                     progress=False, threads=True)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(tickers[0])
    return df.dropna(how="all").dropna(axis=1, how="all")


# Heavy analyses are cached so moving one slider doesn't recompute everything.
# `_prices` (leading underscore) is skipped by the hasher; `key` identifies the dataset.
@st.cache_data(show_spinner="Sweeping stop-loss levels…", max_entries=16)
def c_sweep(_prices, key, bench, start_year, params):
    rows = []
    for s in [0, 5, 10, 15, 20, 25, 30, 40]:
        r = run_strategy(_prices, bench, start_year, **{**params, "stop_pct": s})
        if r:
            m = metrics(r["equity"])
            rows.append({"Stop-loss %": f"{s}%" if s else "None", "Final value": m["Final value"],
                         "CAGR": m["CAGR"], "Max drawdown": m["Max drawdown"],
                         "Sharpe": m["Sharpe (rf=0)"]})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Comparing rules…", max_entries=16)
def c_compare(_prices, key, bench, start_year, params):
    out = []
    for rule in PICK_RULES:
        r = run_strategy(_prices, bench, start_year, **{**params, "rule": rule})
        if r:
            out.append((rule, r["equity"], metrics(r["equity"])))
    return out


@st.cache_data(show_spinner="Re-running from every start year…", max_entries=16)
def c_sens(_prices, key, bench, params):
    return start_year_sensitivity(_prices, bench, MIN_START, **params)


# ---------------- sidebar ----------------
st.sidebar.title("Strategy")
st.sidebar.selectbox("Stock universe", UNIVERSES, key="universe")
if st.session_state["universe"] == "Custom":
    raw = st.sidebar.text_area("Tickers (comma-separated)", key="custom_tickers")
    universe_list = [t.strip().upper() for t in raw.split(",") if t.strip()]
elif st.session_state["universe"].startswith("S&P"):
    universe_list = sp500_tickers()
else:
    universe_list = LARGE_CAPS

st.sidebar.text_input("Benchmark", key="bench")
st.sidebar.slider("First buy year", MIN_START, 2025, key="start_year")
st.sidebar.number_input("Starting amount ($)", 100, 10_000_000, step=500, key="initial")

st.sidebar.subheader("Pick rule")
st.sidebar.selectbox("How to choose stocks each January", PICK_RULES,
                     format_func=RULES.get, key="rule")
st.sidebar.slider("Number of stocks to buy", 1, 10, key="top_n")

st.sidebar.subheader("Risk rule")
st.sidebar.slider("Sell if down X% (0 = no stop-loss)", 0, 50, key="stop_pct")
st.sidebar.checkbox("Trailing (measured from the peak, not entry)", key="trailing")
bench = st.session_state["bench"].strip().upper()
st.sidebar.radio("After a sell, hold…", ["cash", "benchmark"], key="after_stop",
                 format_func=lambda x: "Cash" if x == "cash" else f"{bench} until next January")

st.sidebar.subheader("Real-world friction")
st.sidebar.slider("Trading cost per trade (basis points)", 0, 50, key="cost_bps",
                  help="5 bps = 0.05% on every buy and every sell.")
st.sidebar.slider("Tax on realized gains (%)", 0, 50, key="tax_pct",
                  help="Applied each year-end when the portfolio rotates. Losses carry forward.")

# ---------------- header + natural-language box ----------------
st.title("📈 Backtest Lab")
st.caption("Ask 'what if…' about the past. Educational tool, not investment advice.")

with st.container(border=True):
    has_key = bool(get_api_key())
    st.text_input("Ask a what-if question", key="nl_text", disabled=not has_key,
                  placeholder="What if I bought the 3 best stocks of the previous year every January "
                              "and sold anything that fell 10%?")
    st.button("Set up simulation", on_click=apply_nl, type="primary", disabled=not has_key)
    if not has_key:
        st.info("Add `ANTHROPIC_API_KEY` to `.streamlit/secrets.toml` (or as an environment variable) "
                "to turn on plain-English setup. The sidebar works without it.")
    msg = st.session_state.get("nl_msg")
    if msg:
        kind, text, extra = msg
        (st.success if kind == "ok" else st.error)(text)
        if extra:
            st.warning(f"Not modeled: {extra}")

# ---------------- data + run ----------------
with st.spinner(f"Loading prices for {len(universe_list)} tickers…"):
    prices = load_prices(tuple(sorted(set(universe_list + [bench]))))
if bench not in prices.columns:
    st.error(f"No data for benchmark {bench}.")
    st.stop()
prices[bench] = prices[bench].ffill()
data_key = (tuple(prices.columns), str(prices.index[-1].date()))

S = st.session_state
params = dict(rule=S["rule"], top_n=S["top_n"], stop_pct=S["stop_pct"], trailing=S["trailing"],
              after_stop=S["after_stop"], cost_bps=S["cost_bps"], tax_rate=S["tax_pct"] / 100,
              initial=S["initial"])
start_year = S["start_year"]

res = run_strategy(prices, bench, start_year, **params)
if res is None:
    st.error("Not enough data for this setup. Try a later start year or different tickers.")
    st.stop()
eq, be, yrs = res["equity"], res["bench"], res["years"]
ms, mb = metrics(eq), metrics(be)
initial = S["initial"]

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Results", "Year by year", "Compare rules", "Start-year test", "Stop-loss sweep", "Reality check"])

# ---- Results
with tab1:
    c = st.columns(4)
    c[0].metric("Strategy final value", f"${ms['Final value']:,.0f}", f"{ms['Final value'] / initial - 1:.0%} total")
    c[1].metric(f"{bench} final value", f"${mb['Final value']:,.0f}", f"{mb['Final value'] / initial - 1:.0%} total")
    c[2].metric("Worst year (strategy)", pct(yrs["Strategy"].min()))
    c[3].metric("Years beating benchmark", f"{(yrs['Strategy'] > yrs['Benchmark']).sum()} / {len(yrs)}")
    if params["tax_rate"] > 0:
        st.caption(f"Total tax paid by the strategy: ${yrs['Tax paid'].sum():,.0f}. "
                   f"{bench} is shown as untaxed buy-and-hold, which flatters it slightly.")

    log = st.toggle("Log scale", value=False)
    fig = go.Figure()
    fig.add_scatter(x=eq.index, y=eq, name="Strategy", line=dict(width=2.5))
    fig.add_scatter(x=be.index, y=be, name=bench, line=dict(width=2, dash="dot"))
    fig.update_layout(height=430, hovermode="x unified", yaxis_title="Portfolio value ($)",
                      yaxis_type="log" if log else "linear", margin=dict(t=20))
    st.plotly_chart(fig)

    dfig = go.Figure()
    dfig.add_scatter(x=eq.index, y=drawdown(eq), name="Strategy", fill="tozeroy")
    dfig.add_scatter(x=be.index, y=drawdown(be), name=bench)
    dfig.update_layout(height=280, title="Drawdown (peak-to-trough loss)",
                       yaxis_tickformat=".0%", hovermode="x unified", margin=dict(t=40))
    st.plotly_chart(dfig)

    fmt = {"Final value": "${:,.0f}", "CAGR": "{:.1%}", "Volatility": "{:.1%}",
           "Sharpe (rf=0)": "{:.2f}", "Max drawdown": "{:.1%}"}
    st.dataframe(pd.DataFrame({name: {k: fmt[k].format(v) for k, v in m.items()}
                               for name, m in [("Strategy", ms), (bench, mb)]}))

# ---- Year by year
with tab2:
    show = yrs.copy()
    for col in ["Signal", "Strategy", "Benchmark"]:
        show[col] = show[col].map(pct)
    show["Tax paid"] = show["Tax paid"].map(lambda v: f"${v:,.0f}")
    st.dataframe(show, hide_index=True)
    bar = go.Figure()
    bar.add_bar(x=yrs["Year"], y=yrs["Strategy"], name="Strategy")
    bar.add_bar(x=yrs["Year"], y=yrs["Benchmark"], name=bench)
    bar.update_layout(barmode="group", height=320, yaxis_tickformat=".0%", margin=dict(t=20))
    st.plotly_chart(bar)

# ---- Compare rules (+ random control)
with tab3:
    st.write("Your current settings (picks, stop-loss, costs, taxes) applied to each selection rule.")
    cmp = c_compare(prices, data_key, bench, start_year, params)
    cfig, rows = go.Figure(), []
    for rule, e, m in cmp:
        cfig.add_scatter(x=e.index, y=e, name=RULES[rule])
        rows.append({"Rule": RULES[rule], "Final value": m["Final value"], "CAGR": m["CAGR"],
                     "Max drawdown": m["Max drawdown"], "Sharpe": m["Sharpe (rf=0)"]})
    cfig.add_scatter(x=be.index, y=be, name=bench, line=dict(dash="dot", color="gray"))
    cfig.update_layout(height=400, hovermode="x unified", yaxis_type="log",
                       yaxis_title="Portfolio value ($, log)", margin=dict(t=20))
    st.plotly_chart(cfig)
    st.dataframe(pd.DataFrame(rows).style.format(
        {"Final value": "${:,.0f}", "CAGR": "{:.1%}", "Max drawdown": "{:.1%}", "Sharpe": "{:.2f}"}),
        hide_index=True)

    st.subheader("Is it better than luck?")
    st.caption("Runs the same settings with randomly chosen stocks many times. "
               "If your rule can't clearly beat random picks, it isn't telling you much.")
    n_sims = st.slider("Random simulations", 10, 100, 30)
    if st.button("Run random-pick test"):
        cagrs, prog = [], st.progress(0.0)
        for s in range(n_sims):
            r = run_strategy(prices, bench, start_year, **{**params, "rule": "random", "seed": s})
            cagrs.append(metrics(r["equity"])["CAGR"])
            prog.progress((s + 1) / n_sims)
        prog.empty()
        rnd = pd.Series(cagrs)
        a, b, c2 = st.columns(3)
        a.metric("Your rule's CAGR", pct(ms["CAGR"]))
        b.metric("Random picks: median", pct(rnd.median()),
                 f"10th–90th pct: {pct(rnd.quantile(.1))} to {pct(rnd.quantile(.9))}", delta_color="off")
        c2.metric("Random runs your rule beat", pct((ms["CAGR"] > rnd).mean()))

# ---- Start-year sensitivity
with tab4:
    st.write("Same strategy, started in every possible year and held to today. "
             "If the answer swings wildly with the start year, the result was partly timing luck.")
    sens = c_sens(prices, data_key, bench, params)
    if sens.empty:
        st.info("Not enough history for this test.")
    else:
        a, b, c3, d = st.columns(4)
        a.metric("Start years tested", len(sens))
        b.metric("Start years beating benchmark", f"{(sens['Excess'] > 0).sum()} / {len(sens)}")
        c3.metric("Median edge vs benchmark (CAGR)", f"{sens['Excess'].median() * 100:+.1f} pts")
        d.metric("Worst start year (CAGR)", pct(sens["Strategy CAGR"].min()))
        sfig = go.Figure()
        sfig.add_bar(x=sens["Start year"], y=sens["Strategy CAGR"], name="Strategy")
        sfig.add_bar(x=sens["Start year"], y=sens["Benchmark CAGR"], name=bench)
        sfig.update_layout(barmode="group", height=340, yaxis_tickformat=".0%",
                           xaxis_title="Start year", yaxis_title="CAGR to today", margin=dict(t=20))
        st.plotly_chart(sfig)
        st.dataframe(sens.style.format({
            "Strategy CAGR": "{:.1%}", "Benchmark CAGR": "{:.1%}", "Excess": "{:+.1%}",
            "Strategy final": "${:,.0f}", "Benchmark final": "${:,.0f}", "Max drawdown": "{:.1%}"}),
            hide_index=True)

# ---- Stop-loss sweep
with tab5:
    st.write("Same strategy, different stop-loss levels. Look for a stable plateau, not one lucky spike. "
             "A spike usually means overfitting.")
    sw = c_sweep(prices, data_key, bench, start_year, params)
    sfig = go.Figure(go.Bar(x=sw["Stop-loss %"], y=sw["CAGR"]))
    sfig.update_layout(height=300, yaxis_tickformat=".0%", xaxis_title="Stop-loss %",
                       yaxis_title="CAGR", margin=dict(t=20))
    st.plotly_chart(sfig)
    st.dataframe(sw.style.format({"Final value": "${:,.0f}", "CAGR": "{:.1%}",
                                  "Max drawdown": "{:.1%}", "Sharpe": "{:.2f}"}), hide_index=True)

# ---- Reality check
with tab6:
    st.subheader("How much should you trust this?")
    st.warning(
        "**Survivorship bias:** the S&P 500 list is *today's* members. Companies that collapsed "
        "or were dropped aren't in it, which flatters any 'pick the winner' strategy.\n\n"
        "**Stops use daily closes.** Real stops can fill worse after overnight gaps.\n\n"
        "**Costs and taxes are simplified.** Flat per-trade cost, one tax rate on net yearly gains, "
        "and the portfolio is treated as fully sold and rebought every January even if the same "
        "stock is picked again.\n\n"
        "**Small sample:** a few picks a year over ~15 years is not much data. "
        "Use the Start-year test and the random-pick test before believing any result.")
    st.subheader("Current leaders (year-to-date)")
    this_year = prices[prices.index.year == prices.index[-1].year].drop(columns=[bench], errors="ignore")
    ytd = (this_year.iloc[-1] / this_year.iloc[0] - 1).dropna().sort_values(ascending=False).head(10)
    st.caption("What the strategy would be looking at right now. A description of the past, not a prediction.")
    st.dataframe(ytd.map(pct).rename("YTD return"))
