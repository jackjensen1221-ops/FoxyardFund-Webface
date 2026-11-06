"""Pure constants: which universes and benchmarks the app offers."""

UNIVERSES = {
    "large_cap": "Large-cap sample (30)",
    "sp500": "S&P 500 (current members)",
    "nyse": "NYSE (most liquid)",
    "custom": "Custom tickers",
}

# kind: ticker = one fund/stock | blend = daily-rebalanced mix | equal_weight = average of the universe
BENCHMARKS = {
    "sp500": dict(label="S&P 500 (SPY)", kind="ticker", ticker="SPY",
                  note="Large US companies. The default yardstick."),
    "nasdaq100": dict(label="Nasdaq 100 (QQQ)", kind="ticker", ticker="QQQ",
                      note="100 largest Nasdaq companies, tech-heavy."),
    "small_caps": dict(label="Russell 2000 small caps (IWM)", kind="ticker", ticker="IWM",
                       note="US small companies."),
    "dow": dict(label="Dow Jones (DIA)", kind="ticker", ticker="DIA", note="30 large US companies."),
    "total_market": dict(label="Total US market (VTI)", kind="ticker", ticker="VTI",
                         note="Nearly every listed US stock."),
    "sixty_forty": dict(label="60/40 stocks and bonds", kind="blend", parts={"SPY": 0.6, "AGG": 0.4},
                        note="60% SPY, 40% AGG (US bonds), rebalanced daily. A classic balanced portfolio."),
    "bonds": dict(label="US bonds (AGG)", kind="ticker", ticker="AGG", note="Investment-grade US bonds."),
    "gold": dict(label="Gold (GLD)", kind="ticker", ticker="GLD", note="Data starts Nov 2004."),
    "equal_weight": dict(label="Equal-weight universe", kind="equal_weight",
                         note="Buy every stock in your universe equally. Shows whether picking beats just owning them all. "
                              "Inherits the universe's survivorship bias."),
    "hedge_fund": dict(label="Hedge-fund replication (QAI)", kind="ticker", ticker="QAI",
                       note="ETF that mimics a multi-strategy hedge-fund index. True hedge-fund indices (HFRI etc.) are paid data. "
                            "Starts 2009."),
    "long_short": dict(label="Long/short equity (FTLS)", kind="ticker", ticker="FTLS",
                       note="Hedge-fund-style long/short stock ETF. Starts 2014."),
    "managed_futures": dict(label="Managed futures (DBMF)", kind="ticker", ticker="DBMF",
                            note="Replicates managed-futures (CTA) hedge funds. Starts 2019."),
    "market_neutral": dict(label="Market-neutral (BTAL)", kind="ticker", ticker="BTAL",
                           note="Long low-beta, short high-beta stocks. Starts 2011."),
    "merger_arb": dict(label="Merger arbitrage (MNA)", kind="ticker", ticker="MNA",
                       note="Event-driven hedge-fund strategy. Starts 2009."),
    "berkshire": dict(label="Berkshire Hathaway (BRK-B)", kind="ticker", ticker="BRK-B",
                      note="Buffett's holding company."),
    "custom": dict(label="Custom ticker…", kind="ticker", ticker=None, note="Any Yahoo Finance ticker."),
}
