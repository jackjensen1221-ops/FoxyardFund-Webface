"""Data loading: universes, prices, benchmarks. All network access lives here."""
import re
from io import StringIO

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

from catalog import BENCHMARKS

LOAD_FROM = "2004-01-01"
LARGE_CAPS = ("AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO JPM V UNH XOM JNJ WMT PG "
              "MA HD COST NFLX AMD CRM ORCL ADBE KO PEP LLY ABBV MRK CVX BAC").split()
JUNK = r"\b(?:warrants?|rights?|units?|preferred|depositary|notes?|debentures?|funds?|acquisition)\b"


def _fetch_text(url):
    return requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text


@st.cache_data(ttl=86_400, show_spinner=False)
def sp500_tickers():
    html = _fetch_text("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    return [s.replace(".", "-") for s in pd.read_html(StringIO(html))[0]["Symbol"]]


@st.cache_data(ttl=86_400, show_spinner=False)
def nyse_symbols():
    """Common stocks listed on the NYSE, from the official Nasdaq Trader symbol directory."""
    txt = _fetch_text("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt")
    df = pd.read_csv(StringIO(txt), sep="|", dtype=str, on_bad_lines="skip")
    df = df[(df["Exchange"] == "N") & (df["ETF"] == "N") & (df["Test Issue"] == "N")]
    bad_name = df["Security Name"].str.contains(JUNK, case=False, regex=True, na=False)
    bad_sym = df["ACT Symbol"].str.contains(r"[$+=~]", regex=True, na=False)
    return sorted({s.replace(".", "-") for s in df.loc[~bad_name & ~bad_sym, "ACT Symbol"]})


@st.cache_data(ttl=86_400, show_spinner=False)
def liquid_nyse(n: int):
    """Top-n NYSE stocks by median dollar volume over the last 3 months (price >= $5)."""
    syms, parts = nyse_symbols(), []
    for i in range(0, len(syms), 250):
        try:
            d = yf.download(syms[i:i + 250], period="3mo", auto_adjust=True, progress=False, threads=True)
            close, vol = d["Close"], d["Volume"]
            if isinstance(close, pd.Series):
                continue
            parts.append(pd.DataFrame({"dv": (close * vol).median(), "px": close.ffill().iloc[-1]}))
        except Exception:
            continue
    if not parts:
        return []
    s = pd.concat(parts).dropna()
    return list(s[s["px"] >= 5].sort_values("dv", ascending=False).index[:n])


def get_universe(key, custom_text, nyse_n):
    if key == "sp500":
        return sp500_tickers()
    if key == "nyse":
        return liquid_nyse(nyse_n)
    if key == "custom":
        return [t.strip().upper() for t in custom_text.split(",") if t.strip()]
    return LARGE_CAPS


@st.cache_data(ttl=3_600, show_spinner=False)
def load_chunk(tickers: tuple):
    d = yf.download(list(tickers), start=LOAD_FROM, auto_adjust=True, progress=False, threads=True)
    close = d["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    return close.dropna(how="all").dropna(axis=1, how="all")


def load_many(tickers, text="Downloading price history"):
    """Download in batches of 100 so one bad batch can't sink everything, with a progress bar."""
    tickers = sorted(set(tickers))
    chunks = [tuple(tickers[i:i + 100]) for i in range(0, len(tickers), 100)]
    bar, frames = st.progress(0.0, text=text), []
    for i, c in enumerate(chunks):
        try:
            frames.append(load_chunk(c))
        except Exception:
            pass
        bar.progress((i + 1) / len(chunks), text=f"{text} ({i + 1}/{len(chunks)})")
    bar.empty()
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def build_benchmark(key, custom_ticker, stock_prices):
    """Return (label, price Series or None)."""
    spec = BENCHMARKS[key]
    label = spec["label"]
    if key == "equal_weight":
        r = stock_prices.pct_change(fill_method=None).mean(axis=1).fillna(0)
        return label, (1 + r).cumprod()
    if spec["kind"] == "blend":
        px = load_chunk(tuple(sorted(spec["parts"])))
        if set(spec["parts"]) - set(px.columns):
            return label, None
        r = sum(w * px[t].pct_change(fill_method=None) for t, w in spec["parts"].items())
        return label, (1 + r.fillna(0)).cumprod()
    ticker = (custom_ticker or "").strip().upper() if key == "custom" else spec["ticker"]
    if not re.fullmatch(r"[A-Z0-9\-\.\^]{1,12}", ticker or ""):
        return label, None
    label = ticker if key == "custom" else label
    px = load_chunk((ticker,))
    return label, (px[ticker] if ticker in px.columns else None)
