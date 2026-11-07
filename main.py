from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Keep the original repository root on the import path so this API reuses
# your existing engine.py rather than duplicating its calculations.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

try:
    import engine
except Exception as exc:
    raise RuntimeError(
        "Could not import the existing root-level engine.py. "
        "Run this backend from inside your cloned portfolio-backtest-simulator repository."
    ) from exc

app = FastAPI(
    title="Investment Intelligence API",
    version="0.1.0",
    description="API layer for the existing portfolio backtesting engine.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class FundBacktestRequest(BaseModel):
    weights: Dict[str, float] = Field(
        examples=[{"VOO": 0.50, "QQQM": 0.30, "BND": 0.20}]
    )
    start_year: int = Field(default=2015, ge=2005, le=2100)
    initial_investment: float = Field(default=10000, gt=0)
    annual_contribution: float = Field(default=0, ge=0)


def download_adjusted_prices(tickers: List[str], start_year: int) -> pd.DataFrame:
    """Download adjusted historical prices for a small user-selected basket."""
    clean = sorted({t.strip().upper() for t in tickers if t.strip()})
    if not clean or len(clean) > 30:
        raise HTTPException(400, "Provide between 1 and 30 valid ticker symbols.")

    try:
        raw = yf.download(
            clean,
            start=f"{start_year}-01-01",
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="column",
        )
    except Exception as exc:
        raise HTTPException(502, f"Market data download failed: {exc}") from exc

    if raw.empty:
        raise HTTPException(404, "No historical price data was returned.")

    # yfinance returns MultiIndex columns for multi-ticker downloads.
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise HTTPException(502, "Price provider did not return adjusted close data.")
        prices = raw["Close"].copy()
    else:
        if "Close" not in raw.columns:
            raise HTTPException(502, "Price provider did not return adjusted close data.")
        prices = raw[["Close"]].rename(columns={"Close": clean[0]})

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=clean[0])

    prices = prices.dropna(how="all").ffill().dropna(axis=1, how="all")
    if prices.empty:
        raise HTTPException(404, "No usable historical prices were found.")
    return prices


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "investment-intelligence-api"}


@app.get("/api/markets/{ticker}")
def get_market_history(ticker: str, start_year: int = 2015):
    if not ticker.replace("-", "").replace(".", "").isalnum():
        raise HTTPException(400, "Invalid ticker.")
    prices = download_adjusted_prices([ticker], start_year)
    series = prices.iloc[:, 0].dropna()
    return {
        "ticker": ticker.upper(),
        "currency": "USD",
        "data": [
            {"date": idx.strftime("%Y-%m-%d"), "adjusted_close": round(float(value), 4)}
            for idx, value in series.items()
        ],
        "source": "Yahoo Finance via yfinance; verify licensing and data rights before commercial use.",
    }


@app.post("/api/backtest/custom-fund")
def backtest_custom_fund(request: FundBacktestRequest):
    weights = {k.strip().upper(): float(v) for k, v in request.weights.items()}
    if any(v < 0 for v in weights.values()):
        raise HTTPException(400, "Weights cannot be negative.")
    if not weights or sum(weights.values()) <= 0:
        raise HTTPException(400, "At least one positive portfolio weight is required.")
    if len(weights) > 30:
        raise HTTPException(400, "A maximum of 30 holdings is supported per request.")

    prices = download_adjusted_prices(list(weights), request.start_year)
    try:
        result = engine.run_custom_fund(
            prices=prices,
            weights_dict=weights,
            start_year=request.start_year,
            initial=request.initial_investment,
            annual_contribution=request.annual_contribution,
        )
    except Exception as exc:
        raise HTTPException(500, f"Backtest engine error: {exc}") from exc

    if result is None or result.get("equity") is None:
        raise HTTPException(
            422,
            "The existing engine could not build this portfolio. Check ticker availability and date range.",
        )

    equity = result["equity"].dropna()
    principal = result["principal"].reindex(equity.index).ffill()
    metrics = engine.metrics(equity)

    return {
        "weights": result["weights"],
        "metrics": {k: float(v) for k, v in metrics.items()},
        "final_value": float(equity.iloc[-1]),
        "total_contributed": float(principal.iloc[-1]),
        "history": [
            {
                "date": idx.strftime("%Y-%m-%d"),
                "portfolio_value": round(float(value), 2),
                "contributed": round(float(principal.loc[idx]), 2),
            }
            for idx, value in equity.items()
        ],
        "methodology": {
            "engine": "Existing root-level engine.run_custom_fund",
            "price_data": "Yahoo Finance auto-adjusted historical prices",
            "limitations": [
                "This endpoint reuses the current simulator's contribution and weighting logic.",
                "The current custom-fund function does not perform periodic portfolio rebalancing.",
                "The current Sharpe metric assumes a zero risk-free rate.",
                "Historical results are not forecasts or guarantees.",
                "Review transaction costs, taxes, dividends, survivorship bias and data licensing before client use."
            ],
        },
    }
