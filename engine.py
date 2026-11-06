"""Backtest engine: pure pandas, no Streamlit, so it is easy to test and reuse."""
import numpy as np
import pandas as pd

RULES = {
    "prior_year_winner": "Last year's winners",
    "prior_year_loser": "Last year's losers (contrarian)",
    "mom_6m": "6-month momentum",
    "mom_12_1": "12-month momentum (skip last month)",
    "custom_basket": "Custom Stock Basket / Mutual Fund",
    "random": "Random picks (control)",
}


def _signal(rule, prices, stocks, yr, y, rng):
    """Score every stock using ONLY data from before the first trading day of year y."""
    first_day = prices.index[yr == y][0]
    if rule in ("prior_year_winner", "prior_year_loser"):
        prev = prices.loc[yr == y - 1, stocks]
        if prev.empty:
            return None
        return prev.iloc[-1] / prev.iloc[0] - 1
    if rule in ("mom_6m", "mom_12_1"):
        months, skip = (6, 0) if rule == "mom_6m" else (12, 1)
        start = first_day - pd.DateOffset(months=months)
        end = first_day - pd.DateOffset(months=skip)
        w = prices.loc[(prices.index >= start) & (prices.index < end), stocks]
        if w.empty or w.index[0] > start + pd.Timedelta(days=10):
            return None  # not enough history yet
        return w.iloc[-1] / w.iloc[0] - 1
    if rule == "random":
        return pd.Series(rng.random(len(stocks)), index=stocks)
    if rule == "custom_basket":
        return pd.Series(1.0, index=stocks)
    raise ValueError(f"Unknown rule: {rule}")


def run_strategy(prices, bench, start_year, rule="prior_year_winner", top_n=1,
                 stop_pct=0.0, trailing=False, after_stop="cash",
                 cost_bps=0.0, tax_rate=0.0, initial=10_000, 
                 annual_contribution=0.0, custom_weights=None, seed=0):
    """
    Every year, on the first trading day, buy `top_n` stocks picked by `rule`
    (equal weight), or use `custom_weights` if provided. Optionally exit a position 
    if it closes `stop_pct`% below its entry.
    """
    stocks = [c for c in prices.columns if c != bench]
    yr = prices.index.year
    rng = np.random.default_rng(seed)
    cost = cost_bps / 10_000
    capital, carry = float(initial), 0.0
    segs, rows = [], []
    last_year = yr[-1]

    for y in range(start_year, last_year + 1):
        cur = prices[yr == y]
        if cur.empty:
            continue
            
        # Handle Custom Mutual Fund / Portfolio vs Rule-based selection
        if rule == "custom_basket" and custom_weights:
            picks = [t for t in custom_weights.keys() if t in cur.columns]
            if not picks:
                continue
            weights = pd.Series({t: custom_weights[t] for t in picks})
            if weights.sum() > 0:
                weights = weights / weights.sum() # Normalize to 100%
            else:
                weights = pd.Series(1.0 / len(picks), index=picks)
        else:
            sig = _signal(rule, prices, stocks, yr, y, rng
