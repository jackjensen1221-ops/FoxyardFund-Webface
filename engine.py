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
                 initial=10000.0, annual_contribution=10000.0, custom
