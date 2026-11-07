import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data
import nlp
from catalog import BENCHMARKS, UNIVERSES
from engine import RULES, drawdown, factor_attribution, metrics, rolling_metrics, run_strategy, run_custom_fund

st.set_page_config(
    page_title="Institutional Quantitative Terminal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Fintech CSS & Clean Dark Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    section[data-testid="stSidebar"] {
        background-color: #11161d !important;
        border-right: 1px solid #30363d !important;
    }
    .stMetric {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 12px;
        border-radius: 8px;
    }
    .psych-card {
        background-color: #1f1a1d;
        border-left: 4px solid #e74c3c;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

MIN_START = 2005
PICK_RULES = [k for k in RULES if k != "random"]
DEFAULTS = dict(universe="large_cap", nyse_n=250, custom_tickers="AAPL, NVDA, MSFT, TSLA, AMZN",
                bench_key="sp500", secondary_benchs=["nasdaq100"], custom_bench="QQQ", start_year=2012, rule="prior_year_winner",
                top_n=1, initial=10_000, annual_contrib=10_000, stop_pct=15, trailing=False, after_stop="cash",
                cost_bps=5, tax_pct=0, reinvest_dividends=False, nl_text="", basket_tickers="AAPL, MSFT, CASH", show_principal=True,
                chart_metric="Portfolio Value ($)", ai_mode=False, fund_name="Apex Alpha Growth Fund",
                fund_tickers="AAPL, NVDA, MSFT, AMZN, GOOGL")
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
S = st.session_state


def pct(x):
    return f"{x:.1%}"


def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"] or st.secrets["GOOGLE_API_KEY"]
    except Exception:
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def apply_nl():
    if not S.get("ai_mode", False):
        return
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
        S["nl_msg"] = ("error", f"Couldn't parse prompt with Gemini: {e}", "")


# ---------------- Sidebar Controls ----------------
sb = st.sidebar
sb.title("⚙️ Terminal Config")
sb.markdown("---")

sb.toggle("🤖 Enable Gemini AI Mode", key="ai_mode", help="Turn on for natural language prompts & portfolio critiques.")

with sb.expander("🌐 Asset Universe & Benchmarks", expanded=True):
    sb.selectbox("Asset Universe", list(UNIVERSES), format_func=UNIVERSES.get, key="universe")
    if S["universe"] == "custom":
        sb.text_area("Custom Tickers", key="custom_tickers")
    elif S["universe"] == "nyse":
        sb.select_slider("NYSE Liquid Tier", [100, 250, 500, 1000], key="nyse_n")
    sb.selectbox("Primary Benchmark", list(BENCHMARKS), format_func=lambda k: BENCHMARKS[k]["label"], key="bench_key")
    sb.multiselect("Secondary Overlays", list(BENCHMARKS), format_func=lambda k: BENCHMARKS[k]["label"], key="secondary_benchs")

with sb.expander("📈 Strategy & Capital Rules", expanded=False):
    sb.selectbox("Selection Rule", PICK_RULES, format_func=RULES.get, key="rule")
    if S["rule"] != "custom_basket":
        sb.slider("Top N Positions", 1, 10, key="top_n")
    sb.slider("Start Year", MIN_START, 2025, key="start_year")
    sb.number_input("Initial Capital ($)", 100, 10_000_000, step=1000, key="initial")
    sb.number_input("Annual DCA ($)", 0, 100_000, step=500, key="annual_contrib")

with sb.expander("🛡️ Risk & Friction Controls", expanded=False):
    sb.slider("Stop-Loss (%)", 0, 50, key="stop_pct")
    sb.checkbox("Trailing Stop-Loss", key="trailing")
    sb.slider("Execution Cost (bps)", 0, 50, key="cost_bps")
    sb.checkbox("Reinvest Dividends (DRIP)", key="reinvest_dividends", help="Toggle off to payouts as un-reinvested cash stream.")
    sb.checkbox("Show Invested Principal", key="show_principal")

# ---------------- Main Dashboard Layout ----------------
st.title("📊 Institutional Quantitative Terminal")
st.markdown("Advanced multi-asset strategy simulator with dividend payout tracking, custom mutual fund sandbox, and multi-benchmark attribution.")

ai_enabled = S.get("ai_mode", False)
has_key = bool(get_api_key())

if ai_enabled:
    col_in, col_btn = st.columns([6, 1], vertical_alignment="bottom")
    col_in.text_input("Gemini AI Prompt", key="nl_text", label_visibility="collapsed", placeholder="e.g., Backtest momentum stocks starting in 2015 with a 15% stop loss")
    col_btn.button("Run AI", on_click=apply_nl, type="primary", disabled=not has_key)

    msg = S.get("nl_msg")
    if msg:
        kind, text, extra = msg
        if kind == "ok":
            st.success(f"🤖 **AI Applied:** {text}")
        else:
            st.error(text)

st.markdown("---")

try:
    if S["universe"] == "nyse":
        universe_list = data.get_universe("nyse", "", S["nyse_n"])
    else:
        universe_list = data.get_universe(S["universe"], S["custom_tickers"], S["nyse_n"])

    stock_prices = data.load_many(universe_list, "Fetching historical market data") if universe_list else pd.DataFrame()
    bench, bench_series = data.build_benchmark(S["bench_key"], S["custom_bench"], stock_prices if not stock_prices.empty else None)
    
    sec_bench_series = {}
    for sb_key in S["secondary_benchs"]:
        if sb_key != S["bench_key"]:
            b_name, b_ser = data.build_benchmark(sb_key, S["custom_bench"], stock_prices if not stock_prices.empty else None)
            if b_ser is not None:
                sec_bench_series[b_name] = b_ser

    prices = stock_prices.drop(columns=[bench], errors="ignore") if not stock_prices.empty else pd.DataFrame(index=bench_series.index)
    prices[bench] = bench_series.reindex(prices.index).ffill()

    fv = prices[bench].first_valid_index()
    min_ok = max(MIN_START, fv.year + (0 if fv.dayofyear <= 10 else 1))
    start_year = max(S["start_year"], min_ok)

    params = dict(rule=S["rule"], top_n=S["top_n"], stop_pct=S["stop_pct"], trailing=S["trailing"],
                  after_stop=S["after_stop"], cost_bps=S["cost_bps"], tax_rate=S["tax_pct"] / 100,
                  initial=S["initial"], annual_contribution=S["annual_contrib"], reinvest_dividends=S["reinvest_dividends"])

    res = run_strategy(prices, bench, start_year, **params)
    if res is None:
        st.error("Insufficient historical data for selected start year.")
        st.stop()
        
    eq, be, yrs, asset_bd, principal, asset_metrics = res["equity"], res["bench"], res["years"], res["asset_breakdown"], res["principal"], res["asset_metrics"]
    div_curve, annual_divs, total_cash_divs = res["dividend_curve"], res["annual_dividends"], res["total_cash_dividends"]
    ms, mb = metrics(eq), metrics(be)
    roll = rolling_metrics(eq, be)
    attribution = factor_attribution(eq, be)

    # Executive Summary Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stock Equity Value", f"${ms['Final value']:,.2f}", f"{ms['CAGR']*100:+.2f}% CAGR")
    m2.metric(f"Benchmark ({bench})", f"${mb['Final value']:,.2f}", f"{mb['CAGR']*100:+.2f}% CAGR")
    m3.metric("Un-reinvested Cash Dividends", f"${total_cash_divs:,.2f}", "Accumulated Cash" if not S["reinvest_dividends"] else "Reinvested (DRIP)")
    m4.metric("Sharpe Ratio", f"{ms.get('Sharpe (rf=0)', 0.0):.2f}", f"Corr: {attribution['Correlation']:.2f}")

    if ai_enabled and has_key:
        with st.expander("🧠 Gemini AI Portfolio Review"):
            with st.spinner("Analyzing risk profile..."):
                st.markdown(nlp.generate_ai_commentary(ms, RULES.get(S["rule"], S["rule"]), get_api_key()))

    # Main Chart View
    col_hdr, col_drop = st.columns([3, 1])
    col_hdr.markdown("### Performance Trajectory")
    col_drop.selectbox("Metric View", ["Portfolio Value ($)", "Rolling 1Y CAGR (%)", "Rolling 1Y Volatility (%)", "Drawdown (%)"], key="chart_metric", label_visibility="collapsed")
    
    fig = go.Figure()
    metric_mode = S["chart_metric"]

    if metric_mode == "Portfolio Value ($)":
        fig.add_scatter(x=eq.index, y=eq, name="Strategy Stock Equity", line=dict(width=3, color="#2ecc71"))
        if not S["reinvest_dividends"]:
            fig.add_scatter(x=eq.index, y=eq + div_curve, name="Total Wealth (Equity + Cash Dividends)", line=dict(width=2, color="#f1c40f", dash="dot"))
        if S["show_principal"] and not principal.empty:
            fig.add_scatter(x=principal.index, y=principal, name="Principal Invested", line=dict(width=1.5, dash="dash", color="#e67e22"))
        fig.add_scatter(x=be.index, y=be, name=f"Primary Benchmark ({bench})", line=dict(width=1.8, color="#3498db"))
        for b_name, b_ser in sec_bench_series.items():
            scaled_b = b_ser.reindex(eq.index).ffill()
            scaled_b = scaled_b / scaled_b.iloc[0] * S["initial"]
            fig.add_scatter(x=scaled_b.index, y=scaled_b, name=f"Overlay: {b_name}", line=dict(width=1.2, dash="dot"))
        yaxis_format = "$"
    elif metric_mode == "Rolling 1Y CAGR (%)":
        fig.add_scatter(x=roll["strat_cagr"].index, y=roll["strat_cagr"], name="Strategy CAGR", line=dict(width=2, color="#2ecc71"))
        fig.add_scatter(x=roll["bench_cagr"].index, y=roll["bench_cagr"], name="Benchmark CAGR", line=dict(width=1.5, color="#3498db", dash="dot"))
        yaxis_format = ".0%"
    elif metric_mode == "Rolling 1Y Volatility (%)":
        fig.add_scatter(x=roll["strat_vol"].index, y=roll["strat_vol"], name="Strategy Volatility", line=dict(width=2, color="#e74c3c"))
        fig.add_scatter(x=roll["bench_vol"].index, y=roll["bench_vol"], name="Benchmark Volatility", line=dict(width=1.5, color="#3498db", dash="dot"))
        yaxis_format = ".0%"
    else:
        fig.add_scatter(x=eq.index, y=drawdown(eq), name="Strategy Drawdown", line=dict(width=1.5, color="#e74c3c"), fill="tozeroy")
        yaxis_format = ".0%"

    fig.update_layout(template="plotly_dark", height=440, margin=dict(l=20, r=20, t=10, b=10), hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

    # 5-Tab Layout featuring the new "💰 Dividend Cash Income" tab
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💰 Dividend Cash Income",
        "📋 Annual & Asset Breakdown", 
        "📈 Factor Attribution", 
        "🏗️ Mutual Fund Architect", 
        "🧠 Behavioral Risk Coach"
    ])

    with tab1:
        st.markdown("### 💰 Dividend Cash Payout & Income Stream Analysis")
        st.markdown("Tracking cash generated from un-reinvested dividends over time.")
        
        d_col1, d_col2, d_col3 = st.columns(3)
        last_yr_div = annual_divs["Annual Dividend Income ($)"].iloc[-1] if not annual_divs.empty else 0.0
        d_col1.metric("Total Cumulative Cash Payout", f"${total_cash_divs:,.2f}")
        d_col2.metric("Latest Annual Cash Income", f"${last_yr_div:,.2f}")
        d_col3.metric("Est. Monthly Cash Stream", f"${last_yr_div / 12:,.2f}")

        div_fig = go.Figure()
        div_fig.add_scatter(x=div_curve.index, y=div_curve, name="Cumulative Dividend Cash", line=dict(width=3, color="#f1c40f"), fill="tozeroy")
        div_fig.update_layout(template="plotly_dark", height=320, title="Cumulative Cash Dividends Collected ($)", yaxis_tickprefix="$")
        st.plotly_chart(div_fig, use_container_width=True)
        
        st.markdown("#### Annual Cash Payout Breakdown")
        st.dataframe(annual_divs.style.format({"Year": "{:d}", "Annual Dividend Income ($)": "${:,.2f}"}), use_container_width=True)

    with tab2:
        c_t1, c_t2 = st.columns(2)
        with c_t1:
            st.markdown("#### Annual Performance Table")
            st.dataframe(yrs.style.format({"Year": "{:d}", "Strategy": "{:+.1%}", "Benchmark": "{:+.1%}"}), use_container_width=True)
        with c_t2:
            st.markdown("#### Asset Return Summary")
            if not asset_metrics.empty:
                st.dataframe(asset_metrics.style.format({"Final Value ($)": "${:,.2f}", "Annualized CAGR": "{:+.2%}"}), use_container_width=True)
            else:
                st.info("No asset metrics available for this rule configuration.")

    with tab3:
        st.markdown("### Quantitative Factor Regression")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Alpha", f"{attribution['Alpha']*100:+.2f}%")
        f2.metric("Beta", f"{attribution['Beta']:.2f}")
        f3.metric("R-Squared (R²)", f"{attribution['R-squared']:.2f}")
        f4.metric("Correlation", f"{attribution['Correlation']:.2f}")

    with tab4:
        st.markdown("### Custom Mutual Fund Sandbox")
        fm1, fm2 = st.columns([2, 3])
        fm1.text_input("Fund Name", key="fund_name")
        fm2.text_input("Constituents (comma-separated)", key="fund_tickers")
        fund_ticker_list = [t.strip().upper() for t in S["fund_tickers"].split(",") if t.strip()]
        
        if fund_ticker_list:
            c_w = st.columns(min(len(fund_ticker_list), 5))
            fund_weights = {}
            for idx, t in enumerate(fund_ticker_list):
                with c_w[idx % len(c_w)]:
                    fund_weights[t] = st.slider(f"{t} Weight", 0.0, 1.0, 1.0 / len(fund_ticker_list), 0.05, key=f"w_{t}")
            
            fund_prices = data.load_many(fund_ticker_list, "Fetching custom fund data")
            if not fund_prices.empty:
                f_res = run_custom_fund(fund_prices, fund_weights, start_year, S["initial"], S["annual_contrib"])
                if f_res:
                    f_eq, f_p, f_w = f_res["equity"], f_res["principal"], f_res["weights"]
                    f_ms = metrics(f_eq)
                    
                    fm_a, fm_b, fm_c = st.columns(3)
                    fm_a.metric("Fund Final Value", f"${f_ms['Final value']:,.2f}", f"{f_ms['CAGR']*100:+.2f}% CAGR")
                    fm_b.metric("Fund Sharpe", f"{f_ms['Sharpe (rf=0)']:.2f}")
                    fm_c.metric("Fund Max Drawdown", pct(f_ms["Max drawdown"]))
                    
                    ffig = go.Figure()
                    ffig.add_scatter(x=f_eq.index, y=f_eq, name=S["fund_name"], line=dict(width=2.5, color="#9b59b6"))
                    ffig.add_scatter(x=be.index, y=be, name=f"Benchmark ({bench})", line=dict(width=1.5, color="#3498db", dash="dot"))
                    ffig.update_layout(template="plotly_dark", height=320, margin=dict(l=20, r=20, t=10, b=10), yaxis_tickprefix="$")
                    st.plotly_chart(ffig, use_container_width=True)

    with tab5:
        st.markdown("### Behavioral Risk & Emotional Vulnerability Audit")
        max_dd = abs(ms["Max drawdown"])
        vol = ms["Volatility"]
        if max_dd > 0.30:
            st.markdown(f'<div class="psych-card"><h4>⚠️ High Loss Aversion Risk (Max Drawdown: {max_dd:.1%})</h4><p>Exceeds the 30% threshold where panic selling historically spikes.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="background-color: #162621; border-left: 4px solid #2ecc71; padding: 15px; border-radius: 4px;"><h4 style="color: #2ecc71;">✅ Moderate Drawdown ({max_dd:.1%})</h4><p>Within comfortable behavioral tolerance bands.</p></div>', unsafe_allow_html=True)

except Exception as e:
    st.error(f"Runtime Exception: {e}")
