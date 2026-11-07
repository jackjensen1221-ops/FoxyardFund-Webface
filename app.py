import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data
import nlp
from catalog import BENCHMARKS, UNIVERSES
from engine import RULES, drawdown, factor_attribution, metrics, rolling_metrics, run_strategy

st.set_page_config(
    page_title="Institutional Quantitative Terminal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
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
                cost_bps=5, tax_pct=0, reinvest_dividends=True, nl_text="", basket_tickers="AAPL, MSFT, CASH", show_principal=True,
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

sb.subheader("🤖 AI Intelligence")
sb.toggle("Enable Gemini AI Mode", key="ai_mode", help="Turn on for natural language prompts and AI portfolio reviews.")

sb.markdown("---")
sb.subheader("1. Universe Selection")
sb.selectbox("Asset Universe", list(UNIVERSES), format_func=UNIVERSES.get, key="universe")
if S["universe"] == "custom":
    sb.text_area("Custom Tickers (comma-separated)", key="custom_tickers")
elif S["universe"] == "nyse":
    sb.select_slider("Most-traded NYSE stocks", [100, 250, 500, 1000], key="nyse_n")

sb.subheader("2. Benchmarks & Multi-Overlay")
sb.selectbox("Primary Benchmark", list(BENCHMARKS), format_func=lambda k: BENCHMARKS[k]["label"], key="bench_key")
sb.multiselect("Secondary Benchmark Overlays", list(BENCHMARKS), format_func=lambda k: BENCHMARKS[k]["label"], key="secondary_benchs")
if S["bench_key"] == "custom" or "custom" in S["secondary_benchs"]:
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
sb.checkbox("Show Total Invested Principal", key="show_principal")

sb.subheader("5. Risk Controls & Behavioral Stops")
sb.slider("Stop-Loss (%)", 0, 50, key="stop_pct")
sb.checkbox("Trailing Stop-Loss", key="trailing")
sb.radio("Post-Stop Allocation", ["cash", "benchmark"], key="after_stop",
         format_func=lambda x: "Hold Cash" if x == "cash" else "Switch to Benchmark")

sb.subheader("6. Frictions & Dividends")
sb.slider("Execution Cost (bps)", 0, 50, key="cost_bps")
sb.slider("Capital Gains Tax Rate (%)", 0, 50, key="tax_pct")
sb.checkbox("Reinvest Dividends (DRIP)", key="reinvest_dividends", help="Automatically reinvest dividend payouts back into stock holdings to compound returns.")

# ---------------- Main Dashboard Layout ----------------
st.title("📊 Institutional Quantitative Backtest Terminal")
st.markdown("Advanced multi-asset strategy simulator with DRIP dividend compounding, custom fund creation, and multi-benchmark attribution.")

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

    sec_bench_series = {}
    for sb_key in S["secondary_benchs"]:
        if sb_key != S["bench_key"]:
            b_name, b_ser = data.build_benchmark(sb_key, S["custom_bench"], stock_prices if not stock_prices.empty else None)
            if b_ser is not None:
                sec_bench_series[b_name] = b_ser

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
                  initial=S["initial"], annual_contribution=S["annual_contrib"], custom_weights=custom_weights,
                  reinvest_dividends=S["reinvest_dividends"])

    res = run_strategy(prices, bench, start_year, **params)
    if res is None:
        st.error("Insufficient historical data for selected start year.")
        st.stop()
        
    eq, be, yrs, asset_bd, principal, asset_metrics = res["equity"], res["bench"], res["years"], res["asset_breakdown"], res["principal"], res["asset_metrics"]
    ms, mb = metrics(eq), metrics(be)
    roll = rolling_metrics(eq, be)
    attribution = factor_attribution(eq, be)

    # Top Executive Summary Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Final Portfolio Value", f"${ms['Final value']:,.2f}", f"{ms['CAGR']*100:+.2f}% CAGR")
    m2.metric(f"Benchmark ({bench})", f"${mb['Final value']:,.2f}", f"{mb['CAGR']*100:+.2f}% CAGR")
    m3.metric("Alpha vs Benchmark", f"{attribution['Alpha']*100:+.2f}%", f"Beta: {attribution['Beta']:.2f}")
    m4.metric("Sharpe Ratio", f"{ms.get('Sharpe (rf=0)', 0.0):.2f}", f"Correlation: {attribution['Correlation']:.2f}")

    if ai_enabled and has_key:
        with st.expander("🧠 Gemini AI Quantitative & Behavioral Review", expanded=False):
            with st.spinner("Gemini is analyzing your risk profile and performance..."):
                ai_review = nlp.generate_ai_commentary(ms, RULES.get(S["rule"], S["rule"]), get_api_key())
                st.markdown(ai_review)

    col_hdr, col_drop = st.columns([2, 1])
    col_hdr.markdown("### Performance Trajectory & Multi-Benchmark Attribution")
    col_drop.selectbox(
        "Chart Metric View", 
        ["Portfolio Value ($)", "Rolling 1Y CAGR (%)", "Rolling 1Y Volatility (%)", "Drawdown (%)"],
        key="chart_metric",
        label_visibility="collapsed"
    )
    
    fig = go.Figure()
    metric_mode = S["chart_metric"]

    if metric_mode == "Portfolio Value ($)":
        fig.add_scatter(x=eq.index, y=eq, name="Strategy Portfolio", line=dict(width=3, color="#2ecc71"))
        if S["show_principal"] and not principal.empty:
            fig.add_scatter(x=principal.index, y=principal, name="Total Cash Invested", line=dict(width=1.8, dash="dash", color="#e67e22"))
        fig.add_scatter(x=be.index, y=be, name=f"Primary Benchmark ({bench})", line=dict(width=1.8, color="#3498db"))
        
        colors = ["#9b59b6", "#e74c3c", "#f1c40f", "#1abc9c"]
        for idx, (b_name, b_ser) in enumerate(sec_bench_series.items()):
            col_c = colors[idx % len(colors)]
            scaled_b = b_ser.reindex(eq.index).ffill()
            scaled_b = scaled_b / scaled_b.iloc[0] * S["initial"]
            fig.add_scatter(x=scaled_b.index, y=scaled_b, name=f"Overlay: {b_name}", line=dict(width=1.2, dash="dot", color=col_c))
            
        yaxis_format = "$"

    elif metric_mode == "Rolling 1Y CAGR (%)":
        fig.add_scatter(x=roll["strat_cagr"].index, y=roll["strat_cagr"], name="Strategy Rolling 1Y CAGR", line=dict(width=2, color="#2ecc71"))
        fig.add_scatter(x=roll["bench_cagr"].index, y=roll["bench_cagr"], name=f"Benchmark Rolling 1Y CAGR ({bench})", line=dict(width=1.5, color="#3498db", dash="dot"))
        yaxis_format = ".0%"

    elif metric_mode == "Rolling 1Y Volatility (%)":
        fig.add_scatter(x=roll["strat_vol"].index, y=roll["strat_vol"], name="Strategy Rolling 1Y Volatility", line=dict(width=2, color="#e74c3c"))
        fig.add_scatter(x=roll["bench_vol"].index, y=roll["bench_vol"], name=f"Benchmark Rolling 1Y Volatility ({bench})", line=dict(width=1.5, color="#3498db", dash="dot"))
        yaxis_format = ".0%"

    else:
        fig.add_scatter(x=eq.index, y=drawdown(eq), name="Strategy Drawdown", line=dict(width=1.5, color="#e74c3c"), fill="tozeroy")
        fig.add_scatter(x=be.index, y=drawdown(be), name=f"Benchmark Drawdown ({bench})", line=dict(width=1, color="#3498db", dash="dot"))
        yaxis_format = ".0%"

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

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Annual Breakdown", 
        "📊 Asset Return Breakdown", 
        "📈 Factor Attribution", 
        "🏗️ Mutual Fund Builder",
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
        st.markdown("### Quantitative Factor Attribution & Regression Analysis")
        st.markdown("Evaluating whether strategy outperformance stems from true stock selection skill (Alpha) or broader systemic market exposure (Beta).")
        
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Annualized Alpha", f"{attribution['Alpha']*100:+.2f}%")
        f2.metric("Portfolio Beta", f"{attribution['Beta']:.2f}")
        f3.metric("R-Squared (R²)", f"{attribution['R-squared']:.2f}")
        f4.metric("Pearson Correlation", f"{attribution['Correlation']:.2f}")
        
        st.info("💡 **Interpretation:** A Beta close to 1.0 indicates your strategy moves in lockstep with the benchmark. A high Alpha with a moderate Beta signifies true risk-adjusted excess returns.")

    with tab4:
        st.markdown("### 🏗️ Mutual Fund Sandbox & Custom Portfolio Architect")
        st.markdown("Tinker with individual equity weightings to design, name, and stress-test your custom fund against broad market benchmarks.")
        
        c_meta1, c_meta2 = st.columns([2, 3])
        c_meta1.text_input("Mutual Fund Name", key="fund_name")
        c_meta2.text_input("Constituent Tickers (comma-separated)", key="fund_tickers")
        
        fund_ticker_list = [t.strip().upper() for t in S["fund_tickers"].split(",") if t.strip()]
        
        if fund_ticker_list:
            st.markdown("#### Constituent Allocation Weights")
            col_weights = st.columns(min(len(fund_ticker_list), 5))
            fund_weights = {}
            for idx, ticker in enumerate(fund_ticker_list):
                with col_weights[idx % len(col_weights)]:
                    fund_weights[ticker] = st.slider(f"{ticker} Weight", 0.0, 1.0, 1.0 / len(fund_ticker_list), 0.05, key=f"w_{ticker}")
                    
            fund_prices = data.load_many(fund_ticker_list, "Fetching fund constituent prices")
            if not fund_prices.empty:
                fund_res = run_custom_fund(fund_prices, fund_weights, start_year, S["initial"], S["annual_contrib"])
                if fund_res:
                    fund_eq = fund_res["equity"]
                    fund_prin = fund_res["principal"]
                    norm_w = fund_res["weights"]
                    fund_ms = metrics(fund_eq)
                    
                    st.markdown("---")
                    fm1, fm2, fm3, fm4 = st.columns(4)
                    fm1.metric(f"Fund Value ({S['fund_name']})", f"${fund_ms['Final value']:,.2f}", f"{fund_ms['CAGR']*100:+.2f}% CAGR")
                    fm2.metric("Fund Sharpe Ratio", f"{fund_ms['Sharpe (rf=0)']:.2f}")
                    fm3.metric("Fund Volatility", pct(fund_ms["Volatility"]))
                    fm4.metric("Max Drawdown", pct(fund_ms["Max drawdown"]))
                    
                    c_chart1, c_chart2 = st.columns([3, 2])
                    with c_chart1:
                        ffig = go.Figure()
                        ffig.add_scatter(x=fund_eq.index, y=fund_eq, name=S["fund_name"], line=dict(width=3, color="#9b59b6"))
                        ffig.add_scatter(x=be.index, y=be, name=f"Benchmark ({bench})", line=dict(width=1.8, color="#3498db", dash="dot"))
                        if S["show_principal"]:
                            ffig.add_scatter(x=fund_prin.index, y=fund_prin, name="Total Invested Principal", line=dict(width=1.5, dash="dash", color="#e67e22"))
                        ffig.update_layout(template="plotly_dark", height=380, title=f"Historical Growth of {S['fund_name']}", yaxis_tickprefix="$")
                        st.plotly_chart(ffig, use_container_width=True)
                        
                    with c_chart2:
                        pie_df = pd.DataFrame({"Ticker": list(norm_w.keys()), "Normalized Weight": list(norm_w.values())})
                        pie_fig = px.pie(pie_df, values="Normalized Weight", names="Ticker", title="Target Portfolio Allocation", hole=0.4, template="plotly_dark")
                        st.plotly_chart(pie_fig, use_container_width=True)

    with tab5:
        st.markdown("### Behavioral Finance & Emotional Vulnerability Audit")
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

    with tab6:
        st.markdown("""
        * **Dividend Reinvestment (DRIP):** Toggle whether dividend distributions are automatically reinvested to compound total return or held as cash.
        * **Mutual Fund Builder:** Architect custom stock baskets, specify custom weighting schemes, name your fund, and track compounding trajectories against benchmark baselines.
        * **Multi-Benchmark Overlays:** Overlay multiple benchmark series (S&P 500, Nasdaq, Russell, Bonds) simultaneously onto the main trajectory chart.
        * **Factor Attribution:** Calculates Alpha, Beta, R², and correlation via daily return linear regression against the primary benchmark.
        * **AI Mode Toggle:** Keeps your Gemini API quota protected by disabling AI calls unless explicitly toggled on.
        """)

except Exception as e:
    st.error(f"Runtime Exception: {e}")
