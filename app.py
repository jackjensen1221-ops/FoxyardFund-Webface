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
            sig = _signal(rule, prices, stocks, yr, y, rng)
            if sig is None or sig.empty:
                continue
            sig = sig.sort_values(ascending=(rule == "prior_year_loser"))
            picks = list(sig.index[:top_n])
            weights = pd.Series(1.0 / len(picks), index=picks)

        bench_path = cur[bench] / cur[bench].iloc[0]
        vals, stopped = [], []
        
        for t in picks:
            raw = (cur[t] / cur[t].iloc[0]).ffill()
            path = raw * (1 - cost)  # buy cost
            if stop_pct > 0:
                ref = raw.cummax() if trailing else 1.0
                hit = raw <= ref * (1 - stop_pct / 100)
                if hit.any():
                    d = hit.idxmax()
                    val = path[d] * (1 - cost)  # sell cost
                    path = path.copy()
                    if after_stop == "benchmark":
                        path.loc[d:] = val * (1 - cost) * (bench_path.loc[d:] / bench_path[d])
                    else:
                        path.loc[d:] = val
                    stopped.append(f"{t} ({d:%b %d})")
            vals.append(path * weights[t])

        if not vals:
            continue

        port = pd.concat(vals, axis=1).sum(axis=1) * capital
        tax_paid = 0.0
        
        if y < last_year:  # year-end rotation realizes gains
            end_val = port.iloc[-1] * (1 - cost)
            taxable = (end_val - capital) + carry
            if taxable > 0:
                tax_paid, carry = tax_rate * taxable, 0.0
            else:
                carry = taxable
            # Apply annual contribution at year boundary
            port.iloc[-1] = (end_val - tax_paid) + annual_contribution

        segs.append(port)
        rows.append({
            "Year": y,
            "Bought": ", ".join(picks),
            "Signal": float(sig.iloc[:len(picks)].mean()) if rule != "custom_basket" else 0.0,
            "Stopped out": ", ".join(stopped) if stopped else "-",
            "Tax paid": tax_paid,
            "Strategy": port.iloc[-1] / (capital + (annual_contribution if y > start_year else 0)) - 1,
            "Benchmark": bench_path.iloc[-1] - 1,
        })
        capital = port.iloc[-1]

    if not segs:
        return None
    equity = pd.concat(segs)
    b = prices[bench].reindex(equity.index).ffill()
    return {"equity": equity, "bench": b / b.iloc[0] * initial, "years": pd.DataFrame(rows)}


def metrics(equity):
    r = equity.pct_change().dropna()
    yrs = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    vol = r.std() * np.sqrt(252)
    return {
        "Final value": equity.iloc[-1],
        "CAGR": (equity.iloc[-1] / equity.iloc[0]) ** (1 / yrs) - 1,
        "Volatility": vol,
        "Sharpe (rf=0)": (r.mean() * 252) / vol if vol > 0 else np.nan,
        "Max drawdown": drawdown(equity).min(),
    }


def drawdown(equity):
    return equity / equity.cummax() - 1


def start_year_sensitivity(prices, bench, first=2005, **kw):
    """Re-run the strategy from every possible start year, each ending at the latest date."""
    rows = []
    for sy in range(first, prices.index[-1].year - 1):
        r = run_strategy(prices, bench, sy, **kw)
        if r is None:
            continue
        ms, mb = metrics(r["equity"]), metrics(r["bench"])
        rows.append({
            "Start year": sy,
            "Strategy CAGR": ms["CAGR"], "Benchmark CAGR": mb["CAGR"],
            "Excess": ms["CAGR"] - mb["CAGR"],
            "Strategy final": ms["Final value"], "Benchmark final": mb["Final value"],
            "Max drawdown": ms["Max drawdown"],
        })
    return pd.DataFrame(rows)
