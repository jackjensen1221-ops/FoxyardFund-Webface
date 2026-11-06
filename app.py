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
theme.inject()

MIN_START = 2005
PICK_RULES = [k for k in RULES if k != "random"]
DEFAULTS = dict(universe="large_cap", nyse_n=250, custom_tickers="AAPL, NVDA, MSFT, TSLA, AMZN",
                bench_key="sp500", custom_bench="QQQ", start_year=2012, rule="prior_year_winner",
                top_n=1, initial=10_000, stop_pct=15, trailing=False, after_stop="cash",
                cost_bps=5, tax_pct=0, nl_text="")
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
    """Button callback: runs BEFORE the rerun, so it may set widget state."""
    text = S.get("nl_text", "").strip()
    if not text:
        return
    if S.get("nl_calls", 0) >= 10:
        S["nl_msg"] = ("error", "Scenario limit reached for this session. Use the sidebar controls.", "")
        return
    S["nl_calls"] = S.get("nl_calls", 0) + 1
    try:
        out = nlp.parse_question(text, get_api_key())
    except Exception as e:
        S["nl_msg"] = ("error", f"Couldn't read that question: {e}", "")
        return
    for k, v in out["widgets"].items():
        S[k] = v
    S["nl_msg"] = ("ok", out["explanation"], out["unsupported"])


# Heavy analyses are cached; `_prices` (leading underscore) is not hashed, `key` identifies the dataset.
@st.cache_data(show_spinner="Sweeping stop-loss levels…", max_entries=16)
def c_sweep(_prices, key, bench, start_year, params):
    rows = []
    for s in [0, 5, 10, 15, 20, 25, 30, 40]:
        r = run_strategy(_prices, bench, start_year, **{**params, "stop_pct": s})
        if r:
            m = metrics(r["equity"])
            rows.append({"Stop-loss": f"{s}%" if s else "None", "Final value": m["Final value"],
                         "CAGR": m["CAGR"], "Max drawdown": m["Max drawdown"], "Sharpe": m["Sharpe (rf=0)"]})
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
def c_sens(_prices, key, bench, first, params):
    return start_year_sensitivity(_prices, bench, first, **params)


# ---------------- sidebar ----------------
sb = st.sidebar
sb.subheader("Universe")
sb.selectbox("Stocks to pick from", list(UNIVERSES), format_func=UNIVERSES.get, key="universe")
if S["universe"] == "custom":
    sb.text_area("Tickers (comma-separated)", key="custom_tickers")
elif S["universe"] == "nyse":
    sb.select_slider("Most-traded NYSE stocks", [100, 250, 500, 1000], key="nyse_n")
    sb.caption("Ranked by recent dollar volume. 500+ stocks can take a minute or two on first load.")

sb.subheader("Benchmark")
sb.selectbox("Compare against", list(BENCHMARKS), format_func=lambda k: BENCHMARKS[k]["label"], key="bench_key")
if S["bench_key"] == "custom":
    sb.text_input("Ticker", key="custom_bench")
sb.caption(BENCHMARKS[S["bench_key"]]["note"])

sb.subheader("Pick rule")
sb.selectbox("How to choose stocks each January", PICK_RULES, format_func=RULES.get, key="rule")
sb.slider("Number of stocks to buy", 1, 10, key="top_n")
sb.slider("First buy year", MIN_START, 2025, key="start_year")
sb.number_input("Starting amount ($)", 100, 10_000_000, step=500, key="initial")

sb.subheader("Risk rule")
sb.slider("Sell if down X% (0 = no stop-loss)", 0, 50, key="stop_pct")
sb.checkbox("Trailing (measured from the peak, not entry)", key="trailing")
sb.radio("After a sell, hold…", ["cash", "benchmark"], key="after_stop",
         format_func=lambda x: "Cash" if x == "cash" else "The benchmark until next January")

sb.subheader("Costs and taxes")
sb.slider("Trading cost per trade (basis points)", 0, 50, key="cost_bps",
          help="5 bps = 0.05% on every buy and every sell.")
sb.slider("Tax on realized gains (%)", 0, 50, key="tax_pct",
          help="Applied each year-end when the portfolio rotates. Losses carry forward.")

# ---------------- header + scenario bar ----------------
theme.header("Test an investing rule against history before risking real money. Educational tool, not investment advice.")

has_key = bool(get_api_key())
col_in, col_btn = st.columns([6, 1], vertical_alignment="bottom")
col_in.text_input("Describe a scenario", key="nl_text", disabled=not has_key, label_visibility="collapsed",
                  placeholder="Describe a scenario: buy the 3 best NYSE stocks of last year, sell anything down 10%, compare to hedge funds")
col_btn.button("Apply", on_click=apply_nl, type="primary", disabled=not has_key)
if not has_key:
    st.caption("Plain-English setup is off. Add ANTHROPIC_API_KEY to your app's secrets to enable it; the sidebar works without it.")
msg = S.get("nl_msg")
if msg:
    kind, text, extra = msg
    st.caption(f"Simulating: {text}") if kind == "ok" else st.error(text)
    if extra:
        st.caption(f"Not modeled: {extra}")

# ---------------- data ----------------
if S["universe"] == "nyse":
    with st.spinner("Ranking NYSE stocks by trading volume (first run takes a minute)…"):
        universe_list = data.get_universe("nyse", "", S["nyse_n"])
else:
    universe_list = data.get_universe(S["universe"], S["custom_tickers"], S["nyse_n"])
if not universe_list:
    st.error("Couldn't build the stock universe. Check your tickers or try again in a minute.")
    st.stop()

stock_prices = data.load_many(universe_list, "Downloading price history")
if stock_prices.empty:
    st.error("No price data came back. Yahoo may be rate-limiting; wait a minute and refresh.")
    st.stop()

bench, bench_series = data.build_benchmark(S["bench_key"], S["custom_bench"], stock_prices)
if bench_series is None or bench_series.dropna().empty:
    st.error(f"No data for benchmark {bench}. The fund may have closed, or the ticker is wrong.")
    st.stop()

prices = stock_prices.drop(columns=[bench], errors="ignore")
prices[bench] = bench_series.reindex(prices.index).ffill()

# The benchmark must have data from the first buy date.
fv = prices[bench].first_valid_index()
min_ok = max(MIN_START, fv.year + (0 if fv.dayofyear <= 10 else 1))
start_year = max(S["start_year"], min_ok)
if start_year > 2025 or start_year > prices.index[-1].year - 1:
    st.error(f"{bench} doesn't have enough history. Pick another benchmark.")
    st.stop()
if start_year != S["start_year"]:
    st.caption(f"{bench} data begins {fv:%b %Y}, so the first buy year was moved to {start_year}.")

data_key = (tuple(prices.columns), str(prices.index[-1].date()))
params = dict(rule=S["rule"], top_n=S["top_n"], stop_pct=S["stop_pct"], trailing=S["trailing"],
              after_stop=S["after_stop"], cost_bps=S["cost_bps"], tax_rate=S["tax_pct"] / 100,
              initial=S["initial"])
initial = S["initial"]

res = run_strategy(prices, bench, start_year, **params)
if res is None:
    st.error("Not enough data for this setup. Try a later start year or different tickers.")
    st.stop()
eq, be, yrs = res["equity"], res["bench"], res["years"]
ms, mb = metrics(eq), metrics(be)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Overview", "Year by year", "Compare rules", "Start-year test", "Stop-loss sweep", "Reality check"])

# ---- Overview
with tab1:
    theme.hero("Portfolio value", f"${ms['Final value']:,.0f}", ms["Final value"] / initial - 1,
               f"since {eq.index[0]:%b %Y}  ·  {bench} {pct(be.iloc[-1] / initial - 1)}")
    n_stocks = prices.shape[1] - 1
    chips_list = [RULES[S["rule"]], f"Top {S['top_n']}",
                  (f"Stop-loss {S['stop_pct']}%" + (" trailing" if S["trailing"] else "")) if S["stop_pct"] else "No stop-loss",
                  f"{S['cost_bps']} bps costs"]
    if S["tax_pct"]:
        chips_list.append(f"{S['tax_pct']}% tax")
    chips_list += [f"{UNIVERSES[S['universe']]}: {n_stocks} stocks", f"vs {bench}"]
    theme.chips(chips_list)

    left, right = st.columns([4, 3], vertical_alignment="bottom")
    with left:
        theme.section("Performance")
    with right:
        opts = ["1Y", "3Y", "5Y", "10Y", "MAX"]
        if hasattr(st, "segmented_control"):
            period = st.segmented_control("Range", opts, default="MAX", key="period",
                                          label_visibility="collapsed") or "MAX"
        else:
            period = st.radio("Range", opts, index=4, horizontal=True, key="period", label_visibility="collapsed")

    def window(s):
        if period == "MAX":
            return s
        return s[s.index >= s.index[-1] - pd.DateOffset(years=int(period[:-1]))]

    e_p, b_p = window(eq), window(be)
    e_p, b_p = e_p / e_p.iloc[0] * initial, b_p / b_p.iloc[0] * initial
    log = st.toggle("Log scale", value=False)
    fig = go.Figure()
    fig.add_scatter(x=e_p.index, y=e_p, name="Strategy", line=dict(width=2.5, color=theme.ACCENT))
    fig.add_scatter(x=b_p.index, y=b_p, name=bench, line=dict(width=1.75, color=theme.BENCH))
    fig.update_layout(height=400, hovermode="x unified", yaxis_type="log" if log else "linear",
                      yaxis_tickprefix="$")
    theme.chart(fig)
    st.caption(f"{period}: strategy {e_p.iloc[-1] / initial - 1:+.1%}  ·  {bench} {b_p.iloc[-1] / initial - 1:+.1%}")

    last = yrs.iloc[-1]
    cur = prices[prices.index.year == last["Year"]]
    tiles = []
    for t in last["Bought"].split(", "):
        r = cur[t].ffill().iloc[-1] / cur[t].iloc[0] - 1
        tiles.append((t, r, f"{t} (" in last["Stopped out"]))
    theme.section(f"Holdings in {last['Year']}", "bought on the first trading day")
    theme.picks(tiles)

    theme.section("Key numbers", f"strategy vs {bench}")
    k = st.columns(4)
    k[0].metric("CAGR", pct(ms["CAGR"]), f"{bench}: {pct(mb['CAGR'])}", delta_color="off")
    k[1].metric("Max drawdown", pct(ms["Max drawdown"]), f"{bench}: {pct(mb['Max drawdown'])}", delta_color="off")
    k[2].metric("Worst year", pct(yrs["Strategy"].min()), f"{bench}: {pct(yrs['Benchmark'].min())}", delta_color="off")
    k[3].metric("Years beating benchmark", f"{(yrs['Strategy'] > yrs['Benchmark']).sum()} / {len(yrs)}")
    if params["tax_rate"] > 0:
        st.caption(f"Total tax paid by the strategy: ${yrs['Tax paid'].sum():,.0f}. "
                   f"{bench} is shown as untaxed buy-and-hold, which flatters it slightly.")

    theme.section("Drawdown", "decline from the previous peak")
    dfig = go.Figure()
    dfig.add_scatter(x=eq.index, y=drawdown(eq), name="Strategy", line=dict(width=1.5, color=theme.RED),
                     fill="tozeroy", fillcolor="rgba(255,92,108,0.10)")
    dfig.add_scatter(x=be.index, y=drawdown(be), name=bench, line=dict(width=1.25, color=theme.BENCH))
    dfig.update_layout(height=250, yaxis_tickformat=".0%", hovermode="x unified")
    theme.chart(dfig)

# ---- Year by year
with tab2:
    sty = yrs.style.format({"Year": "{:d}", "Signal": "{:.1%}", "Strategy": "{:+.1%}",
                            "Benchmark": "{:+.1%}", "Tax paid": "${:,.0f}"})
    st.dataframe(theme.color_cols(sty, ["Strategy", "Benchmark"]), hide_index=True)
    bar = go.Figure()
    bar.add_bar(x=yrs["Year"], y=yrs["Strategy"], name="Strategy", marker_color=theme.ACCENT)
    bar.add_bar(x=yrs["Year"], y=yrs["Benchmark"], name=bench, marker_color=theme.BENCH)
    bar.update_layout(barmode="group", height=320, yaxis_tickformat=".0%")
    theme.chart(bar)

# ---- Compare rules (+ random control)
with tab3:
    st.write("Your current settings (picks, stop-loss, costs, taxes) applied to each selection rule.")
    cfig, rows = go.Figure(), []
    for rule, e, m in c_compare(prices, data_key, bench, start_year, params):
        cfig.add_scatter(x=e.index, y=e, name=RULES[rule])
        rows.append({"Rule": RULES[rule], "Final value": m["Final value"], "CAGR": m["CAGR"],
                     "Max drawdown": m["Max drawdown"], "Sharpe": m["Sharpe (rf=0)"]})
    cfig.add_scatter(x=be.index, y=be, name=bench, line=dict(width=1.75, color=theme.BENCH))
    cfig.update_layout(height=400, hovermode="x unified", yaxis_type="log", yaxis_tickprefix="$")
    theme.chart(cfig)
    st.dataframe(pd.DataFrame(rows).style.format(
        {"Final value": "${:,.0f}", "CAGR": "{:.1%}", "Max drawdown": "{:.1%}", "Sharpe": "{:.2f}"}),
        hide_index=True)

    theme.section("Is it better than luck?")
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
    sens = c_sens(prices, data_key, bench, min_ok, params)
    if sens.empty:
        st.info("Not enough history for this test.")
    else:
        a, b, c3, d = st.columns(4)
        a.metric("Start years tested", len(sens))
        b.metric("Start years beating benchmark", f"{(sens['Excess'] > 0).sum()} / {len(sens)}")
        c3.metric("Median edge vs benchmark", f"{sens['Excess'].median() * 100:+.1f} pts CAGR")
        d.metric("Worst start year (CAGR)", pct(sens["Strategy CAGR"].min()))
        sfig = go.Figure()
        sfig.add_bar(x=sens["Start year"], y=sens["Strategy CAGR"], name="Strategy", marker_color=theme.ACCENT)
        sfig.add_bar(x=sens["Start year"], y=sens["Benchmark CAGR"], name=bench, marker_color=theme.BENCH)
        sfig.update_layout(barmode="group", height=340, yaxis_tickformat=".0%", xaxis_title="Start year",
                           yaxis_title="CAGR to today")
        theme.chart(sfig)
        sty = sens.style.format({"Start year": "{:d}", "Strategy CAGR": "{:.1%}", "Benchmark CAGR": "{:.1%}",
                                 "Excess": "{:+.1%}", "Strategy final": "${:,.0f}",
                                 "Benchmark final": "${:,.0f}", "Max drawdown": "{:.1%}"})
        st.dataframe(theme.color_cols(sty, ["Excess"]), hide_index=True)

# ---- Stop-loss sweep
with tab5:
    st.write("Same strategy, different stop-loss levels. Look for a stable plateau, not one lucky spike. "
             "A spike usually means overfitting.")
    sw = c_sweep(prices, data_key, bench, start_year, params)
    sfig = go.Figure(go.Bar(x=sw["Stop-loss"], y=sw["CAGR"], marker_color=theme.ACCENT))
    sfig.update_layout(height=300, yaxis_tickformat=".0%", xaxis_title="Stop-loss", yaxis_title="CAGR")
    theme.chart(sfig)
    st.dataframe(sw.style.format({"Final value": "${:,.0f}", "CAGR": "{:.1%}",
                                  "Max drawdown": "{:.1%}", "Sharpe": "{:.2f}"}), hide_index=True)

# ---- Reality check
with tab6:
    theme.section("Limitations", "read before trusting any result")
    st.markdown("""
- **Survivorship bias.** The S&P 500 and NYSE universes are built from *today's* stocks. Companies that collapsed or were delisted are missing, which flatters any pick-the-winner rule. The "Equal-weight universe" benchmark shows how much of the gain is just from owning today's survivors.
- **NYSE universe.** Stocks come from the official NYSE listing, minus preferreds, warrants, funds and SPACs, then the most-traded names are kept. Ranking by *current* trading volume adds more hindsight.
- **Hedge-fund benchmarks are proxies.** Real hedge-fund indices are paid data. The ETFs here mimic hedge-fund strategies and have short histories.
- **Stops use daily closes.** Real stops can fill worse after an overnight gap.
- **Costs and taxes are simplified.** A flat per-trade cost, one tax rate on net yearly gains, and the portfolio is treated as fully sold and rebought every January.
- **Small sample.** A few picks a year over about 15 years is not much data. Run the start-year test and the random-pick test before believing a result.
""")
    theme.section("Current leaders", "year-to-date")
    this_year = prices[prices.index.year == prices.index[-1].year].drop(columns=[bench], errors="ignore")
    ytd = (this_year.iloc[-1] / this_year.iloc[0] - 1).dropna().sort_values(ascending=False).head(10)
    st.caption("What the strategy would be looking at right now. A description of the past, not a prediction.")
    st.dataframe(theme.color_cols(ytd.rename("YTD return").to_frame().style.format("{:+.1%}"), ["YTD return"]))
