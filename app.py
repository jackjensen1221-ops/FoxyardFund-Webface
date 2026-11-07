import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data
import nlp
from catalog import BENCHMARKS, UNIVERSES
from engine import RULES, drawdown, metrics, rolling_metrics, run_strategy, start_year_sensitivity

st.set_page_config(
    page_title="Institutional Quantitative Terminal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Fintech CSS & Guaranteed Sidebar Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    section[data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
        background-color: #11161d !important;
        border-right: 1px solid #30363d !important;
    }
    button[kind="header"] {
        background-color: #1f242c !important;
        color: #ffffff !important;
        border: 1px solid #30363d !important;
    }
    .stMetric {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 8px;
    }
    .card {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
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
                bench_key="sp500", custom_bench="QQQ", start_year=2012, rule="prior_year_winner",
                top_n=1, initial=10_000, annual_contrib=10_000, stop_pct=15, trailing=False, after_stop="cash",
                cost_bps=5, tax_pct=0, nl_text="", basket_tickers="AAPL, MSFT, CASH", show_principal=True,
                chart_metric="Portfolio Value ($)", ai_mode=False)
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

# AI Mode Toggle Switch
sb.subheader("🤖 AI Intelligence")
sb.toggle("Enable Gemini AI Mode", key="ai_mode", help="Turn on to allow natural language prompts and AI portfolio reviews. Turn off to save your API quota.")

sb.markdown("---")
sb.subheader("1. Universe Selection")
sb.selectbox("Asset Universe", list(UNIVERSES), format_func=UNIVERSES.get, key="universe")
if S["universe"] == "custom":
    sb.text_area("Custom Tickers (comma-separated)", key="custom_tickers")
elif S["universe"] == "nyse":
    sb.select_slider("Most-traded NYSE stocks", [100, 250, 500, 1000], key="nyse_n")

sb.subheader("2. Benchmark")
sb.selectbox("Primary Benchmark", list(BENCHMARKS), format_func=lambda k: BENCHMARKS[k]["label"], key="bench_key")
if S["bench_key"] == "custom":
    sb.text_input("Custom Benchmark Ticker", key="custom_bench")

sb.subheader("3. Strategy & Allocation")
sb.selectbox("Selection Rule", PICK_RULES, format_func=RULES.get, key="rule")

custom_weights = None
if S["rule"] == "custom_basket":
    sb.markdown("#### Custom Fund / Cash Weights")
    basket_input = sb.text_input("Tickers & Cash (comma-separated)", key="basket_tickers")
    b_list = [t.strip().upper() for t in basket_input.split(",") if t.strip()]
    custom_weights = {}
    for t in b_list:
        custom_weights[t] = sb.slider(f"Weight: {t}", 0.0, 1.0, 1.0 / len(b_list) if b_list else 1.0, 0.05)

if S["rule"] != "custom_basket":
    sb.slider("Top N Positions", 1, 10, key="top_n")

sb.slider("Backtest Start Year", MIN_START, 2025, key="start_year")
sb.number_input("Initial Capital ($)", 100, 10_000_000, step=1000, key="initial")
sb.number_input("Annual DCA Contribution ($)", 0, 100_000, step=500, key="annual_contrib")

sb.subheader("4. Chart Display Options")
sb.checkbox("Show Total Invested Principal (No Interest)", key="show_principal")

sb.subheader("5. Risk Controls & Behavioral Stops")
sb.slider("Stop-Loss (%)", 0, 50, key="stop_pct")
sb.checkbox("Trailing Stop-Loss", key="trailing")
sb.radio("Post-Stop Allocation", ["cash", "benchmark"], key="after_stop",
         format_func=lambda x: "Hold Cash" if x == "cash" else "Switch to Benchmark")

sb.subheader("6. Frictions")
sb.slider("Execution Cost (bps)", 0, 50, key="cost_bps")
sb.slider("Capital Gains Tax Rate (%)", 0, 50, key="tax_pct")

# ---------------- Main Dashboard Layout ----------------
st.title("📊 Institutional Quantitative Backtest Terminal")
st.markdown("Advanced multi-asset strategy simulator equipped with optional Gemini AI intelligence and behavioral risk modeling.")

# Gemini Natural Language Prompt Bar (Only active if AI Mode is toggled ON)
ai_enabled = S.get("ai_mode", False)
has_key = bool(get_api_key())

col_in, col_btn = st.columns([6, 1], vertical_alignment="bottom")
prompt_disabled = not ai_enabled or not has_key
col_in.text_input(
    "Gemini AI Strategy Prompt", 
    key="nl_text", 
    disabled=prompt_disabled, 
    label_visibility="collapsed",
    placeholder="Enable 'Gemini AI Mode' in sidebar to use natural language prompts..." if not ai_enabled else "e.g., Backtest momentum stocks starting in 2015 with a 15% stop loss"
)
col_btn.button("Run AI Prompt", on_click=apply_nl, type="primary", disabled=prompt_disabled)

if ai_enabled and not has_key:
    st.caption("💡 *AI Mode is ON, but your `GEMINI_API_KEY` is missing from Streamlit secrets.*")
elif not ai_enabled:
    st.caption("💤 *AI Mode is OFF (Quota Protected). Toggle it on in the sidebar whenever you want to use AI prompts.*")

msg = S.get("nl_msg")
if msg and ai_enabled:
    kind, text, extra = msg
    if kind == "ok":
        st.success(f"🤖 **Gemini AI Applied:** {text}")
    else:
        st.error(text)
    if extra:
        st.caption(f"Note: {extra}")

st.markdown("---")

try:
    if S["universe"] == "nyse":
        with st.spinner("Compiling liquid NYSE universe data…"):
            universe_list = data.get_universe("nyse", "", S["nyse_n"])
    else:
        universe_list = data.get_universe(S["universe"], S["custom_tickers"], S["nyse_n"])

    if S["rule"] == "custom_basket" and custom_weights:
        stock_basket_keys = [t for t in custom_weights.keys() if t.upper() != "CASH"]
        universe_list = list(set(universe_list + stock_basket_keys))

    if not universe_list and not (custom_weights and len(custom_weights) > 0):
        st.error("Invalid stock universe configuration.")
        st.stop()

    stock_prices = data.load_many(universe_list, "Fetching historical market data") if universe_list else pd.DataFrame()
    
    bench, bench_series = data.build_benchmark(S["bench_key"], S["custom_bench"], stock_prices if not stock_prices.empty else None)
    if bench_series is None or bench_series.dropna().empty:
        st.error(f"Invalid benchmark configuration for {bench}.")
        st.stop()

    if not stock_prices.empty:
        prices = stock_prices.drop(columns=[bench], errors="ignore")
    else:
        prices = pd.DataFrame(index=bench_series.index)
        
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
        st.error("Insufficient historical data for selected start year.")
        st.stop()
        
    eq, be, yrs, asset_bd, principal, asset_metrics = res["equity"], res["bench"], res["years"], res["asset_breakdown"], res["principal"], res["asset_metrics"]
    ms, mb = metrics(eq), metrics(be)
    roll = rolling_metrics(eq, be)

    # Top Executive Summary Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Final Portfolio Value", f"${ms['Final value']:,.2f}", f"{ms['CAGR']*100:+.2f}% CAGR")
    m2.metric("Benchmark Value", f"${mb['Final value']:,.2f}", f"{mb['CAGR']*100:+.2f}% CAGR")
    m3.metric("Max Drawdown", pct(ms["Max drawdown"]), f"Benchmark: {pct(mb['Max drawdown'])}")
    
    bench_sharpe_val = mb.get('Sharpe (rf=0)', 0.0)
    bench_sharpe_text = f"Benchmark: {bench_sharpe_val:.2f}" if 'Sharpe (rf=0)' in mb else ""
    strat_sharpe_val = ms.get('Sharpe (rf=0)', 0.0)
    m4.metric("Sharpe Ratio", f"{strat_sharpe_val:.2f}", bench_sharpe_text)

    # Gemini AI Portfolio Review Banner (Only runs if AI Mode is toggled ON)
    if ai_enabled and has_key:
        with st.expander("🧠 Gemini AI Quantitative & Behavioral Review", expanded=False):
            with st.spinner("Gemini is analyzing your risk profile and performance..."):
                ai_review = nlp.generate_ai_commentary(ms, RULES.get(S["rule"], S["rule"]), get_api_key())
                st.markdown(ai_review)

    # Chart Header & Metric Selector Dropdown
    col_hdr, col_drop = st.columns([2, 1])
    col_hdr.markdown("### Performance Trajectory & Analytics")
    col_drop.selectbox(
        "Chart Metric View", 
        ["Portfolio Value ($)", "Rolling 1Y CAGR (%)", "Rolling 1Y Volatility (%)", "Drawdown (%)"],
        key="chart_metric",
        label_visibility="collapsed"
    )
    
    # Professional Plotly Chart Generation
    fig = go.Figure()
    metric_mode = S["chart_metric"]

    if metric_mode == "Portfolio Value ($)":
        fig.add_scatter(x=eq.index, y=eq, name="Strategy Portfolio", line=dict(width=3, color="#2ecc71"))
        if S["show_principal"] and not principal.empty:
            fig.add_scatter(x=principal.index, y=principal, name="Total Cash Invested (No Interest)", line=dict(width=1.8, dash="dash", color="#e67e22"))
        fig.add_scatter(x=be.index, y=be, name=f"Benchmark ({bench})", line=dict(width=1.5, color="#3498db", dash="dot"))
        yaxis_format = "$"

    elif metric_mode == "Rolling 1Y CAGR (%)":
        fig.add_scatter(x=roll["strat_cagr"].index, y=roll["strat_cagr"], name="Strategy Rolling 1Y CAGR", line=dict(width=2, color="#2ecc71"))
        fig.add_scatter(x=roll["bench_cagr"].index, y=roll["bench_cagr"], name=f"Benchmark Rolling 1Y CAGR ({bench})", line=dict(width=1.5, color="#3498db", dash="dot"))
        yaxis_format = ".0%"

    elif metric_mode == "Rolling 1Y Volatility (%)":
        fig.add_scatter(x=roll["strat_vol"].index, y=roll["strat_vol"], name="Strategy Rolling 1Y Volatility", line=dict(width=2, color="#e74c3c"))
        fig.add_scatter(x=roll["bench_vol"].index, y=roll["bench_vol"], name=f"Benchmark Rolling 1Y Volatility ({bench})", line=dict(width=1.5, color="#3498db", dash="dot"))
        yaxis_format = ".0%"

    else:  # Drawdown (%)
        fig.add_scatter(x=eq.index, y=drawdown(eq), name="Strategy Drawdown", line=dict(width=1.5, color="#e74c3c"), fill="tozeroy")
        fig.add_scatter(x=be.index, y=drawdown(be), name=f"Benchmark Drawdown ({bench})", line=dict(width=1, color="#3498db", dash="dot"))
        yaxis_format = ".0%"

    # Market Crash Annotations
    crashes = {"2008-09-15": "2008 GFC", "2020-03-16": "COVID Crash", "2022-01-03": "2022 Tightening"}
    for date_str, label in crashes.items():
        dt = pd.to_datetime(date_str)
        if eq.index.min() <= dt <= eq.index.max():
            fig.add_vline(x=dt, line_dash="dash", line_color="rgba(255,255,255,0.3)")
            fig.add_annotation(x=dt, y=eq.max() if metric_mode == "Portfolio Value ($)" else 0, text=label, showarrow=False, font=dict(color="gray", size=10))

    fig.update_layout(
        template="plotly_dark",
        height=480,
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode="x unified",
        yaxis_tickprefix="$" if metric_mode == "Portfolio Value ($)" else "",
        yaxis_tickformat="" if metric_mode == "Portfolio Value ($)" else yaxis_format,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Detailed Tabs including Behavioral Psychology Coach
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Annual Breakdown", 
        "📊 Asset Return Breakdown", 
        "📉 Risk Analytics", 
        "🧠 Behavioral Risk Coach",
        "ℹ️ Methodology"
    ])
    
    with tab1:
        st.dataframe(yrs.style.format({"Year": "{:d}", "Strategy": "{:+.1%}", "Benchmark": "{:+.1%}", "Tax paid": "${:,.0f}"}), use_container_width=True)
        
    with tab2:
        st.markdown("#### Individual Asset Return & CAGR Summary")
        if not asset_metrics.empty:
            st.dataframe(asset_metrics.style.format({"Final Value ($)": "${:,.2f}", "Annualized CAGR": "{:+.2%}"}), use_container_width=True)
        else:
            st.info("No individual asset metrics available for this rule configuration.")

    with tab3:
        dfig = go.Figure()
        dfig.add_scatter(x=eq.index, y=drawdown(eq), name="Strategy Drawdown", line=dict(color="#e74c3c", width=1.2), fill="tozeroy")
        dfig.update_layout(template="plotly_dark", height=300, yaxis_tickformat=".0%", margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(dfig, use_container_width=True)

    with tab4:
        st.markdown("### Behavioral Finance & Emotional Vulnerability Audit")
        st.markdown("Quantitative models assume rational actors, but human investors suffer from cognitive biases. Here is how your psychological risk profile looks under this strategy:")
        
        max_dd = abs(ms["Max drawdown"])
        vol = ms["Volatility"]
        
        if max_dd > 0.30:
            st.markdown(f"""
            <div class="psych-card">
                <h4>⚠️ Severe Loss Aversion Trigger (Max Drawdown: {max_dd:.1%})</h4>
                <p>This strategy experienced a peak-to-trough decline greater than 30%. Behavioral research shows that investors experience acute panic selling when losses cross 25%. Ensure your stop-loss or psychological fortitude matches this volatility.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #162621; border-left: 4px solid #2ecc71; padding: 15px; border-radius: 4px; margin-bottom: 15px;">
                <h4 style="color: #2ecc71;">✅ Moderate Drawdown Envelope (Max Drawdown: {max_dd:.1%})</h4>
                <p>Your drawdown remains within historically manageable thresholds, reducing the probability of emotional panic-selling.</p>
            </div>
            """, unsafe_allow_html=True)

        if vol > 0.25:
            st.markdown(f"""
            <div class="psych-card">
                <h4>⚠️ High Volatility & Recency Bias Risk (Volatility: {vol:.1%})</h4>
                <p>High annualized volatility often induces recency bias—chasing recent winners and abandoning discipline after a single bad month.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #162621; border-left: 4px solid #2ecc71; padding: 15px; border-radius: 4px; margin-bottom: 15px;">
                <h4 style="color: #2ecc71;">✅ Stable Volatility Profile (Volatility: {vol:.1%})</h4>
                <p>The return stream is relatively steady, supporting long-term adherence without triggering obsessive portfolio checking.</p>
            </div>
            """, unsafe_allow_html=True)

    with tab5:
        st.markdown("""
        * **AI Mode Toggle:** Keeps your Gemini API quota protected by disabling AI calls unless explicitly toggled on.
        * **Behavioral Audit:** Flags cognitive bias triggers (loss aversion, panic-selling thresholds, and volatility fatigue).
        * **Asset Breakdown:** Calculates annualized CAGR and ending value for each individual asset across the tested timeframe.
        * **Rolling Metrics:** Calculates 1-year (252 trading day) rolling compound returns and annualized volatility.
        """)

except Exception as e:
    st.error(f"Runtime Exception: {e}")
