"""Market-moving days: a curated list of well-known events plus data-driven biggest moves."""
import numpy as np
import pandas as pd

EVENTS = [
    ("2007-02-27", "Shanghai selloff spreads worldwide"),
    ("2008-09-15", "Lehman Brothers files for bankruptcy"),
    ("2008-09-29", "House rejects the bank bailout (TARP)"),
    ("2008-10-13", "Global bank-rescue rally"),
    ("2010-05-06", "Flash Crash"),
    ("2011-08-08", "First trading day after the US credit downgrade"),
    ("2015-08-24", "China devaluation 'Black Monday'"),
    ("2016-06-24", "Brexit vote result"),
    ("2016-11-09", "US presidential election result"),
    ("2018-02-05", "Volatility spike ('Volmageddon')"),
    ("2018-12-24", "Christmas Eve selloff"),
    ("2020-03-09", "COVID plus oil price war crash"),
    ("2020-03-12", "Pandemic declared, Europe travel ban"),
    ("2020-03-16", "COVID crash"),
    ("2020-03-24", "Stimulus rally"),
    ("2020-11-09", "Vaccine efficacy news"),
    ("2021-01-27", "GameStop trading frenzy"),
    ("2022-02-24", "Russia invades Ukraine"),
    ("2022-09-13", "Hot inflation report"),
    ("2022-11-10", "Cooler inflation report rally"),
    ("2023-03-10", "Silicon Valley Bank collapses"),
    ("2024-08-05", "Yen carry-trade unwind"),
    ("2025-04-03", "Tariff 'Liberation Day' selloff"),
    ("2025-04-09", "Tariff pause rally"),
]
NAMES = {pd.Timestamp(d): n for d, n in EVENTS}


def _daily(s):
    return s.pct_change(fill_method=None)


def event_frame(spy, twr, start, end):
    r, sr = _daily(spy), _daily(twr)
    rows = [{"Date": ts, "Event": name, "S&P 500": r.loc[ts], "Strategy": sr.get(ts, np.nan)}
            for ts, name in NAMES.items() if start <= ts <= end and ts in r.index]
    return pd.DataFrame(rows)


def major_days(spy, twr, bench_twr, start, end, n=6):
    """Worst and best S&P 500 days in the window, with how your strategy and benchmark did that day."""
    r = _daily(spy).dropna()
    r = r[(r.index >= start) & (r.index <= end)]
    sr, br = _daily(twr), _daily(bench_twr)

    def build(s):
        return pd.DataFrame({
            "Date": [d.strftime("%b %d, %Y") for d in s.index],
            "What happened": [NAMES.get(d, "—") for d in s.index],
            "S&P 500": s.values,
            "Strategy": [sr.get(d, np.nan) for d in s.index],
            "Benchmark": [br.get(d, np.nan) for d in s.index],
        })
    return build(r.nsmallest(n)), build(r.nlargest(n))


def missed_best_days(spy, start, end, ns=(0, 5, 10, 20)):
    r = _daily(spy).dropna()
    r = r[(r.index >= start) & (r.index <= end)]
    return {n: float((1 + r.drop(r.nlargest(n).index)).prod() - 1) for n in ns}
