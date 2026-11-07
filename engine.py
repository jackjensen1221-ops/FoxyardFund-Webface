"""Quantitative calculation engine for portfolio simulation, multi-asset metrics, factor attribution, and dividend cash income tracking."""
import numpy as np
import pandas as pd

RULES = {
    "prior_year_winner": "Prior Year Winner (Momentum)",
    "prior_year_loser": "Prior Year Loser (Mean Reversion)",
    "mom_6m": "6-Month Momentum",
    "mom_12_1": "12-1M Momentum",
    "custom_basket": "Custom Basket / Fixed Weights",
    "random": "Random Selection (Baseline)"
}

def drawdown(series):
    roll_max = series.cummax()
    return (series - roll_max) / roll_max

def metrics(series):
    if series is None or series.empty or len(series.dropna()) < 2:
        return {}
    s = series.dropna()
    start_val = s.iloc[0]
    end_val = s.iloc[-1]
    days = (s.index[-1] - s.index[0]).days
    years = max(days / 365.25, 0.01)
    cagr = (end_val / start_val) ** (1 / years) - 1 if start_val > 0 else 0.0
    daily_rets = s.pct_change().dropna()
    vol = daily_rets.std() * np.sqrt(252) if not daily_rets.empty else 0.0
    dd = drawdown(s)
    max_dd = dd.min() if not dd.empty else 0.0
    sharpe = (cagr / vol) if vol > 0 else 0.0
    return {
        "Initial value": start_val,
        "Final value": end_val,
        "CAGR": cagr,
        "Volatility": vol,
        "Max drawdown": max_dd,
        "Sharpe (rf=0)": sharpe
    }

def factor_attribution(strat_series, bench_series):
    if strat_series is None or bench_series is None:
        return {"Alpha": 0.0, "Beta": 1.0, "R-squared": 0.0, "Correlation": 1.0}
    df = pd.DataFrame({"strat": strat_series, "bench": bench_series}).dropna()
    if len(df) < 10:
        return {"Alpha": 0.0, "Beta": 1.0, "R-squared": 0.0, "Correlation": 1.0}
    strat_rets = df["strat"].pct_change().dropna()
    bench_rets = df["bench"].pct_change().dropna()
    aligned = pd.DataFrame({"strat": strat_rets, "bench": bench_rets}).dropna()
    if len(aligned) < 5:
        return {"Alpha": 0.0, "Beta": 1.0, "R-squared": 0.0, "Correlation": 1.0}
    cov = np.cov(aligned["strat"], aligned["bench"])
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 1.0
    strat_cagr = (aligned["strat"].mean() * 252)
    bench_cagr = (aligned["bench"].mean() * 252)
    alpha = strat_cagr - (beta * bench_cagr)
    corr = aligned["strat"].corr(aligned["bench"])
    r_squared = corr ** 2 if not np.isnan(corr) else 0.0
    return {"Alpha": alpha, "Beta": beta, "R-squared": r_squared, "Correlation": corr}

def rolling_metrics(eq, be):
    window = 252
    strat_cagr = eq.pct_change(window).dropna()
    bench_cagr = be.pct_change(window).dropna()
    strat_vol = eq.pct_change().rolling(window).std() * np.sqrt(252)
    bench_vol = be.pct_change().rolling(window).std() * np.sqrt(252)
    return {
        "strat_cagr": strat_cagr,
        "bench_cagr": bench_cagr,
        "strat_vol": strat_vol.dropna(),
        "bench_vol": bench_vol.dropna()
    }

def run_custom_fund(prices, weights_dict, start_year, initial=10000.0, annual_contribution=10000.0):
    df = prices.loc[prices.index.year >= start_year].copy()
    if df.empty:
        return None
    valid_weights = {t: w for t, w in weights_dict.items() if t in df.columns and w > 0}
    if not valid_weights:
        return None
    total_w = sum(valid_weights.values())
    norm_weights = {t: w / total_w for t, w in valid_weights.items()}
    dates = df.index
    rets = df[list(norm_weights.keys())].pct_change().fillna(0.0)
    fund_val = float(initial)
    total_invested = float(initial)
    fund_vals = [fund_val]
    prin_vals = [total_invested]
    last_year = dates[0].year
    for i in range(1, len(dates)):
        dt = dates[i]
        curr_year = dt.year
        if curr_year != last_year:
            total_invested += annual_contribution
            fund_val += annual_contribution
            last_year = curr_year
        day_ret = sum(rets.loc[dt, t] * w for t, w in norm_weights.items())
        fund_val = fund_val * (1.0 + day_ret)
        fund_vals.append(fund_val)
        prin_vals.append(total_invested)
    return {
        "equity": pd.Series(fund_vals, index=dates),
        "principal": pd.Series(prin_vals, index=dates),
        "weights": norm_weights
    }

def run_strategy(prices, bench_ticker, start_year, rule="prior_year_winner", top_n=1,
                 stop_pct=15, trailing=False, after_stop="cash", cost_bps=5, tax_rate=0.0,
                 initial=10000.0, annual_contribution=10000.0, custom_weights=None, reinvest_dividends=True):
    if prices.empty:
        return None
    df = prices.loc[prices.index.year >= start_year].copy()
    if df.empty:
        return None
    if bench_ticker in df.columns:
        bench_series = df[bench_ticker]
        stock_df = df.drop(columns=[bench_ticker], errors="ignore")
    else:
        bench_series = df.iloc[:, 0]
        stock_df = df.copy()

    dates = df.index
    n_days = len(dates)
    if n_days < 5:
        return None

    total_invested = float(initial)
    strat_val = float(initial)
    bench_val = float(initial)
    cum_cash_dividends = 0.0
    
    eq_vals = [strat_val]
    bench_vals = [bench_val]
    prin_vals = [total_invested]
    div_cash_vals = [0.0]
    annual_div_payouts = []
    
    last_year = dates[0].year
    curr_yr_divs = 0.0
    
    active_tickers = list(stock_df.columns[:top_n]) if not stock_df.empty else []
    if rule == "custom_basket" and custom_weights:
        active_tickers = [t for t in custom_weights.keys() if t.upper() != "CASH" and t in stock_df.columns]

    stock_rets = stock_df.pct_change().fillna(0.0)
    bench_rets = bench_series.pct_change().fillna(0.0)
    daily_div_rate = 0.000075

    for i in range(1, n_days):
        dt = dates[i]
        curr_year = dt.year
        
        b_ret = bench_rets.loc[dt] if dt in bench_rets.index else 0.0
        if np.isnan(b_ret):
            b_ret = 0.0

        if curr_year != last_year:
            total_invested += annual_contribution
            strat_val += annual_contribution
            bench_val += annual_contribution
            annual_div_payouts.append({"Year": last_year, "Annual Dividend Income ($)": curr_yr_divs})
            curr_yr_divs = 0.0
            last_year = curr_year
            
            if not stock_df.empty and len(stock_df.columns) > 0:
                if rule == "prior_year_winner":
                    past_slice = stock_df.loc[stock_df.index.year < curr_year]
                    if not past_slice.empty:
                        yr_ret = (past_slice.iloc[-1] / past_slice.iloc[0]) - 1
                        active_tickers = yr_ret.nlargest(top_n).index.tolist()
                elif rule == "prior_year_loser":
                    past_slice = stock_df.loc[stock_df.index.year < curr_year]
                    if not past_slice.empty:
                        yr_ret = (past_slice.iloc[-1] / past_slice.iloc[0]) - 1
                        active_tickers = yr_ret.nsmallest(top_n).index.tolist()
                else:
                    active_tickers = stock_df.columns[:top_n].tolist()

        if active_tickers and all(t in stock_rets.columns for t in active_tickers):
            s_ret = stock_rets.loc[dt, active_tickers].mean()
        else:
            s_ret = b_ret
            
        if np.isnan(s_ret):
            s_ret = 0.0
            
        day_div_amount = strat_val * daily_div_rate
        if not reinvest_dividends:
            cum_cash_dividends += day_div_amount
            curr_yr_divs += day_div_amount
            s_ret -= daily_div_rate

        if curr_year != last_year:
            s_ret -= (cost_bps / 10000.0)

        strat_val = strat_val * (1.0 + s_ret)
        bench_val = bench_val * (1.0 + b_ret)

        eq_vals.append(strat_val)
        bench_vals.append(bench_val)
        prin_vals.append(total_invested)
        div_cash_vals.append(cum_cash_dividends)

    annual_div_payouts.append({"Year": last_year, "Annual Dividend Income ($)": curr_yr_divs})

    equity_curve = pd.Series(eq_vals, index=dates)
    bench_scaled = pd.Series(bench_vals, index=dates)
    principal_curve = pd.Series(prin_vals, index=dates)
    dividend_curve = pd.Series(div_cash_vals, index=dates)

    years = sorted(list(set(dates.year)))
    yrs_data = []
    for yr in years:
        yr_slice = equity_curve[equity_curve.index.year == yr]
        if len(yr_slice) > 1:
            ret = (yr_slice.iloc[-1] / yr_slice.iloc[0]) - 1
            bench_yr = bench_scaled[bench_scaled.index.year == yr]
            b_ret = (bench_yr.iloc[-1] / bench_yr.iloc[0]) - 1 if len(bench_yr) > 1 else 0.0
            yrs_data.append({"Year": yr, "Strategy": ret, "Benchmark": b_ret, "Tax paid": 0.0})
            
    yrs_df = pd.DataFrame(yrs_data)
    
    asset_metrics_list = []
    for t in stock_df.columns[:10]:
        t_series = stock_df[t].dropna()
        if not t_series.empty:
            t_fin = initial * (t_series.iloc[-1] / t_series.iloc[0])
            t_cagr = (t_series.iloc[-1] / t_series.iloc[0]) ** (1 / max((t_series.index[-1] - t_series.index[0]).days / 365.25, 0.01)) - 1
            asset_metrics_list.append({"Asset": t, "Final Value ($)": t_fin, "Annualized CAGR": t_cagr})
            
    asset_metrics = pd.DataFrame(asset_metrics_list) if asset_metrics_list else pd.DataFrame(columns=["Asset", "Final Value ($)", "Annualized CAGR"])

    return {
        "equity": equity_curve,
        "bench": bench_scaled,
        "years": yrs_df,
        "asset_breakdown": stock_df,
        "principal": principal_curve,
        "asset_metrics": asset_metrics,
        "dividend_curve": dividend_curve,
        "annual_dividends": pd.DataFrame(annual_div_payouts),
        "total_cash_dividends": cum_cash_dividends
    }
