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
# automatically). Best-effort list — a few recently listed/renamed companies
# (Meesho, Physicswallah, Urban Company, Pine Labs, Cohance Lifesciences,
# JSW Cement, Swan Corp, Tenneco Clean Air, HBL Engineering, Amara Raja,
# Piramal Finance) may have a slightly different ticker than guessed here.
# If a symbol shows up in the dashboard's error rows, look up its correct
# NSE symbol (nseindia.com search) and fix it below.
WATCHLIST = [
    "IDBI", "MEESHO", "ASTERDM", "WELCORP", "RBLBANK",
    "PIRAMALFIN", "HINDCOPPER", "SONACOMS", "GLAND", "AEGISLOG",
    "NAVINFLUOR", "POONAWALLA", "NH", "ANANDRATHI", "PHYSICSWALA",
    "DELHIVERY", "WOCKPHARMA", "HIMADRI", "KARURVYSYA", "SAILIFE",
    "NUVAMA", "STARHEALTH", "TATATECH", "MANAPPURAM", "LALPATHLAB",
    "NETWEB", "MRPL", "PNBHOUSING", "IKS", "CDSL",
    "NEULANDLAB", "REDINGTON", "GRSE", "CHOLAHLDNG", "PPLPHARMA",
    "URBANCOMP", "IIFL", "IFCI", "ANGELONE", "CGCL",
    "BANDHANBNK", "AMBER", "ITI", "DATAPATTNS", "KAYNES",
    "CUB", "NBCC", "FORCEMOT", "CREDITACC", "JYOTICNC",
    "AFFLE", "BRIGADE", "ANANTRAJ", "IGL", "SAGILITY",
    "TENNECO", "RAMCOCEM", "GESHIP", "CESC", "HBLPOWER",
    "PINELABS", "CASTROLIND", "FSL", "CAMS", "TRITURBINE",
    "SARDAEN", "GMDCLTD", "AARTIIND", "OLAELEC", "COHANCE",
    "DEEPAKFERT", "DEVYANI", "BEML",
