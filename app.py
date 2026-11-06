from io import StringIO

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

from engine import drawdown, metrics, run_strategy

st.set_page_config(page_title="Backtest Lab", page_icon="📈", layout="wide")

LARGE_CAPS = ("AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO JPM V UNH XOM JNJ WMT PG "
              "MA HD COST NFLX AMD CRM ORCL ADBE KO PEP LLY ABBV MRK CVX BAC").split()


@st.cache_data(ttl=86_400, show_spinner=False)
def sp500_tickers():
    html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                        headers={"User-Agent": "Mozilla/5.0"}, timeout=20).text
    return [s.replace(".", "-") for s in pd.read_html(StringIO(html))[0]["Symbol"]]


@st.cache_data(ttl=3_600, show_spinner=False)
def load_prices(tickers: tuple, start: str):
    df = yf.download(list(tickers), start=start, auto_adjust=True,
                     progress=False, threads=True)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(tickers[0])
    return df.dropna(how="all").dropna(axis=1, how="all")


def pct(x):
    return f"{x:.1%}"


# ---------------- Sidebar ----------------
st.sidebar.title("Strategy")
st.sidebar.caption("Buy last year's winners on the first trading day of each year.")

universe = st.sidebar.selectbox("Stock universe",
                                ["Large-cap sample (30)", "S&P 500 (current members)", "Custom"])
if universe == "Custom":
    raw = st.sidebar.text_area("Tickers (comma-separated)", "AAPL, NVDA, MSFT, TSLA, AMZN")
    universe_list = [t.strip().upper() for t in raw.split(",") if t.strip()]
elif universe.startswith("S&P"):
    universe_list = sp500_tickers()
else:
    universe_list = LARGE_CAPS

bench = st.sidebar.text_input("Benchmark", "SPY").upper()
start_year = st.sidebar.slider("First buy year", 2005, 2025, 2012)
top_n = st.sidebar.slider("Number of top performers to buy", 1, 10, 1)
initial = st.sidebar.number_input("Starting amount ($)", 100, 10_000_000, 10_000, 500)

st.sidebar.subheader("Risk rule")
stop_pct = st.sidebar.slider("Sell if down X% (0 = no stop-loss)", 0, 50, 15)
trailing = st.sidebar.checkbox("Trailing (measured from the peak, not entry)")
after_stop = st.sidebar.radio("After a sell, hold…", ["cash", "benchmark"],
                              format_func=lambda x: "Cash" if x == "cash" else f"{bench} until next January")

# ---------------- Data + run ----------------
st.title("📈 Backtest Lab")
st.caption("Ask 'what if…' about the past. Educational tool, not investment advice.")

with st.spinner(f"Loading prices for {len(universe_list)} tickers…"):
    prices = load_prices(tuple(sorted(set(universe_list + [bench]))), f"{start_year - 1}-01-01")

if bench not in prices.columns:
    st.error(f"No data for benchmark {bench}.")
    st.stop()
prices[bench] = prices[bench].ffill()

res = run_strategy(prices, bench, start_year, top_n, stop_pct, trailing, after_stop, initial)
if res is None:
    st.error("Not enough data to run this setup. Try a later start year or different tickers.")
    st.stop()

eq, be, yrs = res["equity"], res["bench"], res["years"]
ms, mb = metrics(eq), metrics(be)

tab1, tab2, tab3, tab4 = st.tabs(["Results", "Year by year", "Stop-loss sweep", "Reality check"])

with tab1:
    c = st.columns(4)
    c[0].metric("Strategy final value", f"${ms['Final value']:,.0f}",
                f"{ms['Final value'] / initial - 1:.0%} total")
    c[1].metric(f"{bench} final value", f"${mb['Final value']:,.0f}",
                f"{mb['Final value'] / initial - 1:.0%} total")
    c[2].metric("Worst year (strategy)", pct(yrs["Strategy"].min()))
    c[3].metric("Years beating benchmark", f"{(yrs['Strategy'] > yrs['Benchmark']).sum()} / {len(yrs)}")

    log = st.toggle("Log scale", value=False)
    fig = go.Figure()
    fig.add_scatter(x=eq.index, y=eq, name="Strategy", line=dict(width=2.5))
    fig.add_scatter(x=be.index, y=be, name=bench, line=dict(width=2, dash="dot"))
    fig.update_layout(height=430, hovermode="x unified", yaxis_title="Portfolio value ($)",
                      yaxis_type="log" if log else "linear", margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

    dfig = go.Figure()
    dfig.add_scatter(x=eq.index, y=drawdown(eq), name="Strategy", fill="tozeroy")
    dfig.add_scatter(x=be.index, y=drawdown(be), name=bench)
    dfig.update_layout(height=280, title="Drawdown (peak-to-trough loss)",
                       yaxis_tickformat=".0%", hovermode="x unified", margin=dict(t=40))
    st.plotly_chart(dfig, use_container_width=True)

    table = pd.DataFrame({"Strategy": ms, bench: mb})
    fmt = {"Final value": "${:,.0f}", "CAGR": "{:.1%}", "Volatility": "{:.1%}",
           "Sharpe (rf=0)": "{:.2f}", "Max drawdown": "{:.1%}"}
    st.dataframe(table.apply(lambda col: [fmt[i].format(v) for i, v in col.items()]),
                 use_container_width=True)

with tab2:
    show = yrs.copy()
    for col in ["Prior-yr return", "Strategy", "Benchmark"]:
        show[col] = show[col].map(pct)
    st.dataframe(show, hide_index=True, use_container_width=True)
    bar = go.Figure()
    bar.add_bar(x=yrs["Year"], y=yrs["Strategy"], name="Strategy")
    bar.add_bar(x=yrs["Year"], y=yrs["Benchmark"], name=bench)
    bar.update_layout(barmode="group", height=320, yaxis_tickformat=".0%", margin=dict(t=20))
    st.plotly_chart(bar, use_container_width=True)

with tab3:
    st.write("Same strategy, different stop-loss levels. Look for a *stable plateau*, "
             "not a single lucky spike. A spike usually means overfitting.")
    rows = []
    for s in [0, 5, 10, 15, 20, 25, 30, 40]:
        r = run_strategy(prices, bench, start_year, top_n, s, trailing, after_stop, initial)
        if r:
            m = metrics(r["equity"])
            rows.append({"Stop-loss %": s or "None", "Final value": m["Final value"],
                         "CAGR": m["CAGR"], "Max drawdown": m["Max drawdown"],
                         "Sharpe": m["Sharpe (rf=0)"]})
    sw = pd.DataFrame(rows)
    sfig = go.Figure(go.Bar(x=sw["Stop-loss %"].astype(str), y=sw["CAGR"]))
    sfig.update_layout(height=300, yaxis_tickformat=".0%", xaxis_title="Stop-loss %",
                       yaxis_title="CAGR", margin=dict(t=20))
    st.plotly_chart(sfig, use_container_width=True)
    st.dataframe(sw.style.format({"Final value": "${:,.0f}", "CAGR": "{:.1%}",
                                  "Max drawdown": "{:.1%}", "Sharpe": "{:.2f}"}),
                 hide_index=True, use_container_width=True)

with tab4:
    st.subheader("How much should you trust this?")
    st.warning(
        "**Survivorship bias:** the S&P 500 list is *today's* members. Companies that "
        "collapsed or were dropped aren't in it, which flatters any 'pick the winner' strategy.\n\n"
        "**Stops use daily closes.** Real stops can fill worse after overnight gaps. "
        "**No fees, taxes or slippage** are modeled. Short-term gains taxes could matter a lot here.\n\n"
        "**Small sample:** one pick per year over ~14 years is only ~14 data points.")
    st.subheader("Current leaders (year-to-date)")
    this_year = prices[prices.index.year == prices.index[-1].year].drop(columns=[bench], errors="ignore")
    ytd = (this_year.iloc[-1] / this_year.iloc[0] - 1).dropna().sort_values(ascending=False).head(10)
    st.caption("This is what the strategy would be looking at, not a prediction of what will keep winning.")
    st.dataframe(ytd.map(pct).rename("YTD return"), use_container_width=True)
