# Investment Intelligence — Integration Starter

This starter adds a Next.js dark institutional interface and a FastAPI layer that calls the existing root-level `engine.py` in your `portfolio-backtest-simulator` repository.

## Important

- This is a development starter, not a finished or client-ready financial platform.
- It does not modify your existing `app.py`, `engine.py`, `data.py`, or other original files.
- The backend imports your existing `engine.py` from the repository root.
- The custom-fund endpoint downloads adjusted historical prices with `yfinance` and calls `engine.run_custom_fund()` and `engine.metrics()`.
- Your current engine's custom-fund method does not periodically rebalance the holdings. The endpoint reports this limitation.
- Yahoo Finance/yfinance data access and redistribution rights must be reviewed before commercial use.
- No live trading, client authentication, AI recommendations, news feed, or real-time market feed is implemented yet.

## 1. Clone your existing project and create a development branch

```bash
git clone https://github.com/jackjensen1221-ops/portfolio-backtest-simulator.git
cd portfolio-backtest-simulator
git switch -c investment-platform-development
```

## 2. Copy this starter into the repository

Unzip the starter package. Copy its `backend/` and `frontend/` directories into the root of the cloned repository.

Your repository should look like:

```text
portfolio-backtest-simulator/
├── app.py                 # your existing Streamlit application
├── engine.py              # your existing quantitative engine
├── data.py                # your existing data module
├── catalog.py
├── events.py
├── nlp.py
├── theme.py
├── backend/
│   ├── main.py
│   └── requirements.txt
└── frontend/
    ├── app/
    ├── package.json
    └── ...
```

Do not delete or replace your existing files.

## 3. Start the Python API

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

If your existing `requirements.txt` is missing a dependency required by `engine.py`, install it and add it to the appropriate requirements file.

Check the API:
- http://127.0.0.1:8000/api/health
- http://127.0.0.1:8000/docs

## 4. Start the frontend

Open a second Terminal window, go to the repository root, then:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000.

## 5. Test the backtest

The starter page defaults to:

- VOO: 50
- QQQM: 30
- BND: 20
- Starting year: 2015
- Initial investment: $10,000
- Annual contribution: $0

Change the holdings or values and select **Run historical backtest**.

Weights are relative and normalized by the existing engine. The backend will return an error if the provider does not return usable price data for a selected ticker.

## 6. Commit to the development branch

When the starter is working:

```bash
git status
git add backend frontend README.md
git commit -m "Add investment intelligence web and API starter"
git push -u origin investment-platform-development
```

## Next engineering milestones

1. Add automated tests around the existing engine before changing its financial logic.
2. Refactor `data.py` so its data access functions do not depend on Streamlit UI/cache functions.
3. Add provider abstraction, request caching, rate limits and explicit market-data timestamps.
4. Improve backtesting: periodic rebalancing, cash-flow timing, dividends, costs, taxes, benchmark alignment, risk-free-rate-aware Sharpe, and bias controls.
5. Add persistent users, roles, portfolios and audit history.
6. Add sourced news, FRED macro data and an AI assistant that calls deterministic analysis tools.
7. Add a paper-trading ledger and simulated order workflow.
8. Complete security, privacy, legal and financial compliance review before any client launch.

## Environment variables

For local development, the frontend can use the default API URL. To override it, create `frontend/.env.local`:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

Never put private API keys in variables prefixed with `NEXT_PUBLIC_`.
