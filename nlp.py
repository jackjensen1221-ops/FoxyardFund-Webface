"""Turn a plain-English 'what if' question into backtest settings using Claude."""
import re

from catalog import BENCHMARKS

MODEL = "claude-haiku-5-5"   # cheap and fast; swap for "claude-sonnet-5-5" if you want more nuance

RULES = ["prior_year_winner", "prior_year_loser", "mom_6m", "mom_12_1"]

SYSTEM = """You configure a stock backtesting tool. Convert the user's question into settings by calling set_backtest.
What the tool can do: once a year (first trading day of January) buy the top N stocks chosen by a rule, equal weight.
Rules: prior_year_winner (best last calendar year), prior_year_loser (worst last year), mom_6m, mom_12_1.
Universes: large_cap (30 big stocks), sp500, nyse (most-traded NYSE stocks), custom (user named tickers).
Benchmarks: sp500, nasdaq100, small_caps, dow, total_market, sixty_forty, bonds, gold, equal_weight, hedge_fund (QAI), long_short, managed_futures, market_neutral, merger_arb, berkshire. If the user names another ticker, use benchmark_ticker instead.
Optional stop-loss as a PERCENT drop from entry (or from the peak if trailing), checked on daily closes; after a sell the money sits in cash or moves to the benchmark until next January. Optional trading costs and a tax rate on realized gains.
Only set fields the user actually mentioned or clearly implied; omit everything else.
If the user gives a stop-loss as a dollar amount or says 'X' without a number, do not invent one: omit stop_pct and say so in 'unsupported'.
Anything the tool cannot do (intraday data, shorting, leverage, options, rebalancing more often than yearly, non-stock assets) goes in 'unsupported', in one short sentence.
'explanation' restates in 1-2 plain sentences exactly what will be simulated."""

TOOL = {
    "name": "set_backtest",
    "description": "Set the backtest parameters.",
    "input_schema": {
        "type": "object",
        "properties": {
            "explanation": {"type": "string"},
            "unsupported": {"type": "string", "description": "Empty if everything is supported."},
            "rule": {"type": "string", "enum": RULES},
            "top_n": {"type": "integer", "description": "How many stocks to buy (1-10)."},
            "stop_pct": {"type": "number", "description": "Stop-loss percent (0-50). 0 means none."},
            "trailing": {"type": "boolean"},
            "after_stop": {"type": "string", "enum": ["cash", "benchmark"]},
            "start_year": {"type": "integer", "description": "First year to buy (2005-2025)."},
            "cost_bps": {"type": "number", "description": "Cost per trade in basis points."},
            "tax_pct": {"type": "number", "description": "Tax rate on realized gains, percent."},
            "universe": {"type": "string", "enum": ["large_cap", "sp500", "nyse", "custom"]},
            "tickers": {"type": "array", "items": {"type": "string"}, "description": "Only if user named specific stocks."},
            "benchmark": {"type": "string", "enum": [k for k in BENCHMARKS if k != "custom"]},
            "benchmark_ticker": {"type": "string", "description": "Only for a benchmark not in the list."},
        },
        "required": ["explanation"],
    },
}


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def clean(raw: dict) -> dict:
    """Validate and clamp whatever the model returned; never trust it blindly."""
    w = {}
    if raw.get("rule") in RULES:
        w["rule"] = raw["rule"]
    if isinstance(raw.get("top_n"), (int, float)):
        w["top_n"] = int(_clamp(round(raw["top_n"]), 1, 10))
    if isinstance(raw.get("stop_pct"), (int, float)):
        w["stop_pct"] = int(_clamp(round(raw["stop_pct"]), 0, 50))
    if isinstance(raw.get("trailing"), bool):
        w["trailing"] = raw["trailing"]
    if raw.get("after_stop") in ("cash", "benchmark"):
        w["after_stop"] = raw["after_stop"]
    if isinstance(raw.get("start_year"), (int, float)):
        w["start_year"] = int(_clamp(round(raw["start_year"]), 2005, 2025))
    if isinstance(raw.get("cost_bps"), (int, float)):
        w["cost_bps"] = int(_clamp(round(raw["cost_bps"]), 0, 100))
    if isinstance(raw.get("tax_pct"), (int, float)):
        w["tax_pct"] = int(_clamp(round(raw["tax_pct"]), 0, 60))
    if raw.get("universe") in ("large_cap", "sp500", "nyse", "custom"):
        w["universe"] = raw["universe"]
    tickers = [t.strip().upper().replace(".", "-") for t in raw.get("tickers", []) if isinstance(t, str)]
    tickers = [t for t in tickers if re.fullmatch(r"[A-Z0-9\-]{1,10}", t)][:40]
    if tickers:
        w["custom_tickers"] = ", ".join(tickers)
        w["universe"] = "custom"
    if raw.get("benchmark") in BENCHMARKS and raw["benchmark"] != "custom":
        w["bench_key"] = raw["benchmark"]
    else:
        b = str(raw.get("benchmark_ticker", "")).strip().upper()
        if re.fullmatch(r"[A-Z0-9\-]{1,10}", b):
            w["bench_key"], w["custom_bench"] = "custom", b
    return {"widgets": w,
            "explanation": str(raw.get("explanation", "")).strip(),
            "unsupported": str(raw.get("unsupported", "")).strip()}


def parse_question(text: str, api_key: str) -> dict:
    import anthropic  # imported here so the app still loads if the package is missing
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=MODEL, max_tokens=700, system=SYSTEM, tools=[TOOL],
        tool_choice={"type": "tool", "name": "set_backtest"},
        messages=[{"role": "user", "content": text[:1000]}],
    )
    block = next(b for b in msg.content if b.type == "tool_use")
    return clean(block.input)
