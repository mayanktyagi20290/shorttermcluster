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
