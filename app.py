import streamlit as st
import yfinance as yf
import pandas as pd

# 1. App Header & Title
st.set_page_config(page_title="Portfolio Backtest Simulator", layout="wide")
st.title("📈 Hypothetical Portfolio Backtest Simulator")
st.caption("Enter tickers, allocate percentages, and simulate historical performance against the S&P 500.")

# 2. Sidebar Configurations (User Inputs)
st.sidebar.header("Simulation Parameters")

# Ticker input box (comma-separated)
ticker_input = st.sidebar.text_input("Enter Tickers (comma-separated):", "AAPL, NVDA, MSFT")
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

# Investment variables
initial_investment = st.sidebar.number_input("Initial Investment ($):", min_value=100, value=10000, step=500)
start_date = st.sidebar.date_input("Start Date:", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date:", pd.to_datetime("2026-01-01"))

# Benchmark toggle
benchmark_ticker = "SPY" # S&P 500 ETF

# 3. Data Fetching & Processing Engine
if st.sidebar.button("Run Simulation"):
    if not tickers:
        st.error("Please enter at least one valid ticker symbol.")
    else:
        with st.spinner("Downloading historical stock data..."):
            try:
                # Fetch target portfolio data
                all_tickers = tickers + [benchmark_ticker]
                data = yf.download(all_tickers, start=start_date, end=end_date)['Adj Close']
                
                if data.empty:
                    st.error("No data found for the specified tickers or date range.")
                else:
                    # Handle case where only 1 ticker is downloaded (re-formats DataFrame)
                    if isinstance(data, pd.Series):
                        data = data.to_frame()
                    
                    # Fill missing values if stocks started trading at different times
                    data = data.ffill().dropna()

                    # Equal weight allocation strategy
                    num_stocks = len(tickers)
                    weight = 1.0 / num_stocks
                    
                    # Calculate Normalized / Cumulative Returns (starting at 1.0)
                    normalized_data = data / data.iloc[0]
                    
                    # Calculate Custom Portfolio Growth
                    portfolio_normalized = normalized_data[tickers].sum(axis=1) * weight if num_stocks > 1 else normalized_data[tickers[0]]
                    portfolio_value = portfolio_normalized * initial_investment
                    
                    # Calculate Benchmark Growth
                    benchmark_value = normalized_data[benchmark_ticker] * initial_investment
                    
                    # Combine results into a clean plotting DataFrame
                    df_results = pd.DataFrame({
                        "Custom Portfolio": portfolio_value,
                        "S&P 500 (SPY)": benchmark_value
                    }, index=data.index)

                    # 4. Display Results & Metrics
                    final_port_val = df_results["Custom Portfolio"].iloc[-1]
                    final_bench_val = df_results["S&P 500 (SPY)"].iloc[-1]
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Your Portfolio Final Value", f"${final_port_val:,.2f}", 
                              f"{((final_port_val - initial_investment)/initial_investment)*100:.1f}% Total Return")
                    col2.metric("S&P 500 Final Value", f"${final_bench_val:,.2f}", 
                              f"{((final_bench_val - initial_investment)/initial_investment)*100:.1f}% Total Return")

                    # 5. Render Interactive Performance Chart
                    st.subheader("Performance Trajectory ($)")
                    st.line_chart(df_results)
                    
            except Exception as e:
                st.error(f"An error occurred during data retrieval: {e}")
else:
    st.info("👈 Set your tickers and parameters in the sidebar, then click 'Run Simulation'.")
