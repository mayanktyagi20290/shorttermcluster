"""
5-13-21 EMA + RSI50 + Elder Force Index confluence scanner.
Fetches daily NSE data via yfinance (server-side, no CORS issues) and
writes data.json for the dashboard to read.

Usage:
    python scan.py                       (uses WATCHLIST below)
    python scan.py RELIANCE TCS INFY     (overrides with given symbols)
"""

import sys
import json
import datetime
import pandas as pd
import yfinance as yf

# Nifty 50 constituents (NSE symbols, no .NS suffix — added automatically).
# Index is rebalanced twice a year (end Jan / end Jul) — update this list
# after a rebalance if a stock has changed.
WATCHLIST = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL",
    "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL",
    "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HINDALCO",
    "HINDUNILVR", "ICICIBANK", "ITC", "INFY", "INDIGO",
    "JSWSTEEL", "JIOFIN", "KOTAKBANK", "LT", "M&M",
    "MARUTI", "MAXHEALTH", "NTPC", "NESTLEIND", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SHRIRAMFIN", "SBIN",
    "SUNPHARMA", "TCS", "TATACONSUM", "TMPV", "TATASTEEL",
    "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
]


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    out = 100 - (100 / (1 + rs))
    return out.fillna(100)


def elder_force_index(close: pd.Series, volume: pd.Series, period: int = 13) -> pd.Series:
    raw = (close.diff()) * volume
    return ema(raw.fillna(0), period)


def classify(v5, v13, v21, r, efi):
    bull_stack = v5 > v13 > v21
    bear_stack = v5 < v13 < v21
    rsi_bull, rsi_bear = r > 50, r < 50
    efi_bull, efi_bear = efi > 0, efi < 0

    if v5 > v21 and v5 > v13:
        strong = bull_stack and rsi_bull and efi_bull
        return ("Strong Buy" if strong else "Buy (confirmed)"), "buy", 100
    if v5 > v13:
        return "Buy (anticipatory)", "buy", 40
    if v5 < v21 and v5 < v13:
        strong = bear_stack and rsi_bear and efi_bear
        return ("Strong Sell" if strong else "Sell (confirmed)"), "sell", 100
    if v5 < v13:
        return "Sell (anticipatory)", "sell", 60
    return "Neutral", None, 0


def compute_row(symbol: str, close: pd.Series, volume: pd.Series):
    close = close.dropna()
    volume = volume.reindex(close.index)
    if len(close) < 25:
        raise ValueError(f"not enough history ({len(close)} bars)")

    e5, e13, e21 = ema(close, 5), ema(close, 13), ema(close, 21)
    r = rsi(close, 14)
    efi = elder_force_index(close, volume, 13)

    v5, v13, v21 = float(e5.iloc[-1]), float(e13.iloc[-1]), float(e21.iloc[-1])
    vr, vefi = float(r.iloc[-1]), float(efi.iloc[-1])
    ltp = float(close.iloc[-1])

    signal, direction, pct = classify(v5, v13, v21, vr, vefi)

    return {
        "symbol": symbol,
        "ltp": round(ltp, 2),
        "ema5": round(v5, 2),
        "ema13": round(v13, 2),
        "ema21": round(v21, 2),
        "rsi": round(vr, 1),
        "efi": round(vefi, 0),
        "signal": signal,
        "direction": direction,
        "pct": pct,
    }


def main():
    symbols = sys.argv[1:] if len(sys.argv) > 1 else WATCHLIST
    tickers = [f"{s}.NS" for s in symbols]

    # Single batched call for all tickers — much faster and gentler on
    # Yahoo's rate limits than fetching one symbol at a time.
    raw = yf.download(
        tickers=tickers,
        period="6mo",
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )

    results, errors = [], []
    for sym, ticker in zip(symbols, tickers):
        try:
            if len(tickers) == 1:
                df = raw
            else:
                if ticker not in raw.columns.get_level_values(0):
                    raise ValueError("no data returned")
                df = raw[ticker]
            if df is None or df.empty:
                raise ValueError("empty dataframe")
            results.append(compute_row(sym, df["Close"], df["Volume"]))
        except Exception as e:
            errors.append({"symbol": sym, "error": str(e)})

    out = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "results": results,
        "errors": errors,
    }
    with open("data.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"Scanned {len(results)} ok, {len(errors)} failed. Wrote data.json")


if __name__ == "__main__":
    main()
