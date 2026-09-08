"""
Two-cluster NSE scanner:
  - Short-term: 5/13/21 EMA + RSI50 + Elder Force Index (13) — for traders
  - Long-term: 20/50/100/200 SMA + RSI14 + MACD(12,26,9) + Volume vs 20d avg — for investors
Both computed from the same daily NSE data (fetched server-side via yfinance,
so no CORS issues). Writes data.json for the dashboard to read.

Usage:
    python scan.py                       (uses WATCHLIST below)
    python scan.py RELIANCE TCS INFY     (overrides with given symbols)
"""

import sys
import json
import datetime
import pandas as pd
import yfinance as yf

# Nifty Smallcap 100 constituents (NSE symbols, no .NS suffix — added
# automatically). A few remaining low-confidence tickers (Pine Labs,
# Cohance Lifesciences, JSW Cement, Swan Corp, Piramal Finance) may still
# need correcting — if a symbol shows up in the dashboard's error rows,
# look up its correct NSE symbol (nseindia.com search) and fix it below.
WATCHLIST = [
    "IDBI", "MEESHO", "ASTERDM", "WELCORP", "RBLBANK",
    "PIRAMALFIN", "HINDCOPPER", "SONACOMS", "GLAND", "AEGISLOG",
    "NAVINFLUOR", "POONAWALLA", "NH", "ANANDRATHI", "PWL",
    "DELHIVERY", "WOCKPHARMA", "HSCL", "KARURVYSYA", "SAILIFE",
    "NUVAMA", "STARHEALTH", "TATATECH", "MANAPPURAM", "LALPATHLAB",
    "NETWEB", "MRPL", "PNBHOUSING", "IKS", "CDSL",
    "NEULANDLAB", "REDINGTON", "GRSE", "CHOLAHLDNG", "PPLPHARMA",
    "URBANCO", "IIFL", "IFCI", "ANGELONE", "CGCL",
    "BANDHANBNK", "AMBER", "ITI", "DATAPATTNS", "KAYNES",
    "CUB", "NBCC", "FORCEMOT", "CREDITACC", "JYOTICNC",
    "AFFLE", "BRIGADE", "ANANTRAJ", "IGL", "SAGILITY",
    "TENNIND", "RAMCOCEM", "GESHIP", "CESC", "HBLENGINE",
    "PINELABS", "CASTROLIND", "FSL", "CAMS", "TRITURBINE",
    "SARDAEN", "GMDCLTD", "AARTIIND", "OLAELEC", "COHANCE",
    "DEEPAKFERT", "DEVYANI", "BEML", "JSWCEMENT", "CHAMBLFERT",
    "GPIL", "KFINTECH", "PGEL", "SYNGENE", "TATACHEM",
    "ARE&M", "FIVESTAR", "CROMPTON", "NATCOPHARM", "JBMA",
    "APTUS", "INOXWIND", "JMFINANCIL", "SIGNATURE", "IRCON",
    "KEC", "AFCONS", "ZENSARTECH", "WHIRLPOOL", "BLS",
    "FIRSTCRY", "RPOWER", "SWANCORP",
]

HISTORY_PERIOD = "2y"  # need 200+ daily bars for the long-term cluster


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


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


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


# ---------- short-term (5/13/21 EMA + RSI50 + EFI) ----------

def classify_short(v5, v13, v21, r, efi):
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


def compute_short(close: pd.Series, volume: pd.Series):
    close = close.dropna()
    volume = volume.reindex(close.index)
    if len(close) < 25:
        raise ValueError(f"not enough history ({len(close)} bars)")

    e5, e13, e21 = ema(close, 5), ema(close, 13), ema(close, 21)
    r = rsi(close, 14)
    efi = elder_force_index(close, volume, 13)

    v5, v13, v21 = float(e5.iloc[-1]), float(e13.iloc[-1]), float(e21.iloc[-1])
    vr, vefi = float(r.iloc[-1]), float(efi.iloc[-1])

    signal, direction, pct = classify_short(v5, v13, v21, vr, vefi)

    return {
        "ema5": round(v5, 2), "ema13": round(v13, 2), "ema21": round(v21, 2),
        "rsi": round(vr, 1), "efi": round(vefi, 0),
        "signal": signal, "direction": direction, "pct": pct,
    }


# ---------- long-term (20/50/100/200 SMA + RSI14 + MACD + Volume) ----------

def classify_long(price, s20, s50, s100, s200, r, macd_hist, vol_ratio):
    above20, above50, above100, above200 = price > s20, price > s50, price > s100, price > s200
    below20, below50, below100, below200 = price < s20, price < s50, price < s100, price < s200
    bullish_macd, bearish_macd = macd_hist > 0, macd_hist < 0
    rsi_bull = r > 50
    vol_confirm = vol_ratio > 1

    if above20 and above50 and above100 and above200 and rsi_bull and bullish_macd and vol_confirm:
        return "Strong Bullish", "buy", "Stage 4 — above 20/50/100/200 SMA, RSI/MACD/volume confirmed"
    if below20 and below50 and below100 and below200 and bearish_macd:
        return "Strong Bearish", "sell", "Below 20/50/100/200 SMA, bearish MACD"
    if above50 and above100 and above200 and bullish_macd:
        stage = "Stage 3 — above 50/100/200 SMA" if above100 else "Stage 2 — above 50 SMA"
        return "Bullish", "buy", stage
    if below50 and below100 and bearish_macd:
        return "Bearish", "sell", "Below 50/100 SMA, bearish MACD"
    return "Neutral / Wait", None, "Mixed signals — no clear trend confirmation"


def compute_long(close: pd.Series, volume: pd.Series):
    close = close.dropna()
    volume = volume.reindex(close.index)
    if len(close) < 210:
        raise ValueError(f"not enough history for 200 SMA ({len(close)} bars)")

    s20, s50, s100, s200 = sma(close, 20), sma(close, 50), sma(close, 100), sma(close, 200)
    r = rsi(close, 14)
    line, sig, hist = macd(close)
    vol_avg20 = volume.rolling(20).mean()

    price = float(close.iloc[-1])
    v20, v50, v100, v200 = float(s20.iloc[-1]), float(s50.iloc[-1]), float(s100.iloc[-1]), float(s200.iloc[-1])
    vr = float(r.iloc[-1])
    vhist = float(hist.iloc[-1])
    vol_ratio = float(volume.iloc[-1] / vol_avg20.iloc[-1]) if vol_avg20.iloc[-1] else 1.0

    signal, direction, stage = classify_long(price, v20, v50, v100, v200, vr, vhist, vol_ratio)

    return {
        "sma20": round(v20, 2), "sma50": round(v50, 2), "sma100": round(v100, 2), "sma200": round(v200, 2),
        "rsi": round(vr, 1), "macd_hist": round(vhist, 2), "vol_ratio": round(vol_ratio, 2),
        "signal": signal, "direction": direction, "stage": stage,
    }


def scan_symbol(symbol: str, df: pd.DataFrame):
    close, volume = df["Close"], df["Volume"]
    ltp = round(float(close.dropna().iloc[-1]), 2)
    out = {"symbol": symbol, "ltp": ltp}
    out["short"] = compute_short(close, volume)
    try:
        out["long"] = compute_long(close, volume)
    except ValueError as e:
        out["long"] = None
        out["long_error"] = str(e)
    return out


def main():
    symbols = sys.argv[1:] if len(sys.argv) > 1 else WATCHLIST
    tickers = [f"{s}.NS" for s in symbols]

    raw = yf.download(
        tickers=tickers,
        period=HISTORY_PERIOD,
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
            results.append(scan_symbol(sym, df))
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
