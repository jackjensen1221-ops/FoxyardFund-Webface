"""Backtest engine: pure pandas, no Streamlit, so it is easy to test and reuse."""
import numpy as np
import pandas as pd


def run_strategy(prices, bench, start_year, top_n=1, stop_pct=0.0,
                 trailing=False, after_stop="cash", initial=10_000):
    """
    Each year, on the first trading day, buy the `top_n` best performers of the
    PREVIOUS calendar year (equal weight). Optionally exit a position if it
    closes `stop_pct`% below its entry price (or below its running peak if
    `trailing`). Proceeds then sit in cash or move into the benchmark until
    the next January rebalance.

    prices: DataFrame of adjusted closes (columns = tickers incl. benchmark)
    Returns dict(equity, bench, years) or None if nothing could be simulated.
    """
    stocks = [c for c in prices.columns if c != bench]
    capital, segs, rows = float(initial), [], []

    for y in range(start_year, prices.index[-1].year + 1):
        prev = prices[prices.index.year == y - 1][stocks]
        cur = prices[prices.index.year == y]
        if prev.empty or cur.empty:
            continue

        first, last = prev.iloc[0], prev.iloc[-1]
        ok = first.notna() & last.notna() & cur[stocks].iloc[0].notna()
        ret = (last[ok] / first[ok] - 1).sort_values(ascending=False)
        picks = list(ret.index[:top_n])
        if not picks:
            continue

        bench_path = cur[bench] / cur[bench].iloc[0]
        vals, stopped = [], []
        for t in picks:
            path = (cur[t] / cur[t].iloc[0]).ffill()
            if stop_pct > 0:
                ref = path.cummax() if trailing else 1.0
                hit = path <= ref * (1 - stop_pct / 100)
                if hit.any():
                    d = hit.idxmax()
                    exit_val = path[d]
                    path = path.copy()
                    if after_stop == "benchmark":
                        path.loc[d:] = exit_val * (bench_path.loc[d:] / bench_path[d])
                    else:
                        path.loc[d:] = exit_val
                    stopped.append(f"{t} ({d:%b %d})")
            vals.append(path)

        port = pd.concat(vals, axis=1).mean(axis=1) * capital
        segs.append(port)
        rows.append({
            "Year": y,
            "Bought": ", ".join(picks),
            "Prior-yr return": float(ret.iloc[:len(picks)].mean()),
            "Stopped out": ", ".join(stopped) if stopped else "-",
            "Strategy": port.iloc[-1] / capital - 1,
            "Benchmark": bench_path.iloc[-1] - 1,
        })
        capital = port.iloc[-1]

    if not segs:
        return None
    equity = pd.concat(segs)
    b = prices[bench].reindex(equity.index).ffill()
    return {"equity": equity, "bench": b / b.iloc[0] * initial,
            "years": pd.DataFrame(rows)}


def metrics(equity):
    r = equity.pct_change().dropna()
    yrs = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    vol = r.std() * np.sqrt(252)
    dd = equity / equity.cummax() - 1
    return {
        "Final value": equity.iloc[-1],
        "CAGR": (equity.iloc[-1] / equity.iloc[0]) ** (1 / yrs) - 1,
        "Volatility": vol,
        "Sharpe (rf=0)": (r.mean() * 252) / vol if vol > 0 else np.nan,
        "Max drawdown": dd.min(),
    }


def drawdown(equity):
    return equity / equity.cummax() - 1
