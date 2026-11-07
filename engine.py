"""Backtest engine: pure pandas, no Streamlit, supporting multi-asset tracking and cash holdings."""
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
            return None
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
    try:
        stocks = [c for c in prices.columns if c != bench]
        yr = prices.index.year
        rng = np.random.default_rng(seed)
        cost = cost_bps / 10_000
        capital, carry = float(initial), 0.0
        segs, rows, asset_segs = [], [], []
        last_year = yr[-1]

        for y in range(start_year, last_year + 1):
            cur = prices[yr == y]
            if cur.empty:
                continue
                
            weights = pd.Series(dtype=float)
            if rule == "custom_basket" and custom_weights is not None:
                valid_weights = {t: w for t, w in custom_weights.items() if t in cur.columns or t.upper() == "CASH"}
                if not valid_weights:
                    picks = []
                    weights = pd.Series(dtype=float)
                else:
                    total_w = sum(valid_weights.values())
                    if total_w > 0:
                        weights = pd.Series({t: w / total_w for t, w in valid_weights.items()})
                    else:
                        weights = pd.Series(1.0 / len(valid_weights), index=list(valid_weights.keys()))
                    picks = [t for t in weights.index if t.upper() != "CASH"]
            else:
                sig = _signal(rule, prices, stocks, yr, y, rng)
                if sig is None or sig.empty:
                    continue
                sig = sig.sort_values(ascending=(rule == "prior_year_loser"))
                picks = list(sig.index[:top_n])
                weights = pd.Series(1.0 / len(picks), index=picks)

            bench_path = cur[bench] / cur[bench].iloc[0]
            vals, stopped = [], []
            individual_paths = {}
            
            # Handle pure cash allocation if no stocks picked or 100% cash weight
            cash_weight = weights.get("CASH", 0.0) if "CASH" in weights else 0.0
            stock_weight_mult = 1.0 - cash_weight

            for t in picks:
                if t not in cur.columns:
                    continue
                raw = (cur[t] / cur[t].iloc[0]).ffill()
                path = raw * (1 - cost)
                if stop_pct > 0:
                    ref = raw.cummax() if trailing else 1.0
                    hit = raw <= ref * (1 - stop_pct / 100)
                    if hit.any():
                        d = hit.idxmax()
                        val = path[d] * (1 - cost)
                        path = path.copy()
                        if after_stop == "benchmark":
                            path.loc[d:] = val * (1 - cost) * (bench_path.loc[d:] / bench_path[d])
                        else:
                            path.loc[d:] = val
                        stopped.append(f"{t} ({d:%b %d})")
                w_val = weights.get(t, 1.0 / len(picks) if len(picks) > 0 else 0.0) * stock_weight_mult
                weighted_path = path * w_val
                vals.append(weighted_path)
                individual_paths[t] = weighted_path * capital

            # Add Cash component if applicable
            if cash_weight > 0:
                cash_path = pd.Series(1.0, index=cur.index) * cash_weight
                vals.append(cash_path)
                individual_paths["CASH"] = cash_path * capital

            if not vals:
                # 100% cash fallback if everything else fails
                port = pd.Series(1.0, index=cur.index) * capital
                individual_paths["CASH"] = port
            else:
                port = pd.concat(vals, axis=1).sum(axis=1) * capital

            tax_paid = 0.0
            if y < last_year:
                end_val = port.iloc[-1] * (1 - cost)
                taxable = (end_val - capital) + carry
                if taxable > 0:
                    tax_paid, carry = tax_rate * taxable, 0.0
                else:
                    carry = taxable
                port.iloc[-1] = (end_val - tax_paid) + annual_contribution

            segs.append(port)
            asset_segs.append(pd.DataFrame(individual_paths))
            
            rows.append({
                "Year": y,
                "Bought": ", ".join(picks) + (f" + {cash_weight*100:.0f}% Cash" if cash_weight > 0 else ""),
                "Signal": float(sig.iloc[:len(picks)].mean()) if rule != "custom_basket" and len(picks) > 0 else 0.0,
                "Stopped out": ", ".join(stopped) if stopped else "-",
                "Tax paid": tax_paid,
                "Strategy": port.iloc[-1] / (capital + (annual_contribution if y > start_year else 0)) - 1,
                "Benchmark": bench_path.iloc[-1] - 1,
            })
            capital = port.iloc[-1]

        if not segs:
            return None
        equity = pd.concat(segs)
        asset_breakdown = pd.concat(asset_segs).reindex(equity.index).ffill()
        b = prices[bench].reindex(equity.index).ffill()
        return {
            "equity": equity, 
            "bench": b / b.iloc[0] * initial, 
            "years": pd.DataFrame(rows),
            "asset_breakdown": asset_breakdown
        }
    except Exception as e:
        print(f"Strategy execution error: {e}")
        return None


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
