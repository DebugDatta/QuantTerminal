"""
Universe Definition Module for Indian Equities.
Defines Mega/Large, Mid-Cap, and Small-Cap ticker universes with sector metadata and benchmark indices.
"""

import pandas as pd
from typing import Dict, List

# 1. Mega / Large Cap Universe (39 tickers)
MEGA_LARGE_TICKERS: List[Dict[str, str]] = [
    {"ticker": "RELIANCE.NS", "initial_cap_group": "Mega/Large", "sector": "Energy"},
    {"ticker": "HDFCBANK.NS", "initial_cap_group": "Mega/Large", "sector": "Financials"},
    {"ticker": "ICICIBANK.NS", "initial_cap_group": "Mega/Large", "sector": "Financials"},
    {"ticker": "SBIN.NS", "initial_cap_group": "Mega/Large", "sector": "Financials"},
    {"ticker": "BHARTIARTL.NS", "initial_cap_group": "Mega/Large", "sector": "Telecom"},
    {"ticker": "TCS.NS", "initial_cap_group": "Mega/Large", "sector": "IT"},
    {"ticker": "INFY.NS", "initial_cap_group": "Mega/Large", "sector": "IT"},
    {"ticker": "ITC.NS", "initial_cap_group": "Mega/Large", "sector": "Consumer"},
    {"ticker": "LT.NS", "initial_cap_group": "Mega/Large", "sector": "Industrials"},
    {"ticker": "HINDUNILVR.NS", "initial_cap_group": "Mega/Large", "sector": "Consumer"},
    {"ticker": "BAJFINANCE.NS", "initial_cap_group": "Mega/Large", "sector": "Financials"},
    {"ticker": "AXISBANK.NS", "initial_cap_group": "Mega/Large", "sector": "Financials"},
    {"ticker": "KOTAKBANK.NS", "initial_cap_group": "Mega/Large", "sector": "Financials"},
    {"ticker": "M&M.NS", "initial_cap_group": "Mega/Large", "sector": "Auto"},
    {"ticker": "MARUTI.NS", "initial_cap_group": "Mega/Large", "sector": "Auto"},
    {"ticker": "SUNPHARMA.NS", "initial_cap_group": "Mega/Large", "sector": "Healthcare"},
    {"ticker": "HCLTECH.NS", "initial_cap_group": "Mega/Large", "sector": "IT"},
    {"ticker": "NTPC.NS", "initial_cap_group": "Mega/Large", "sector": "Utilities"},
    {"ticker": "POWERGRID.NS", "initial_cap_group": "Mega/Large", "sector": "Utilities"},
    {"ticker": "TITAN.NS", "initial_cap_group": "Mega/Large", "sector": "Consumer"},
    {"ticker": "TATASTEEL.NS", "initial_cap_group": "Mega/Large", "sector": "Metals"},
    {"ticker": "HINDALCO.NS", "initial_cap_group": "Mega/Large", "sector": "Metals"},
    {"ticker": "JSWSTEEL.NS", "initial_cap_group": "Mega/Large", "sector": "Metals"},
    {"ticker": "ULTRACEMCO.NS", "initial_cap_group": "Mega/Large", "sector": "Materials"},
    {"ticker": "ADANIENT.NS", "initial_cap_group": "Mega/Large", "sector": "Industrials"},
    {"ticker": "ADANIPORTS.NS", "initial_cap_group": "Mega/Large", "sector": "Industrials"},
    {"ticker": "ADANIGREEN.NS", "initial_cap_group": "Mega/Large", "sector": "Utilities"},
    {"ticker": "ONGC.NS", "initial_cap_group": "Mega/Large", "sector": "Energy"},
    {"ticker": "COALINDIA.NS", "initial_cap_group": "Mega/Large", "sector": "Energy"},
    {"ticker": "IOC.NS", "initial_cap_group": "Mega/Large", "sector": "Energy"},
    {"ticker": "NESTLEIND.NS", "initial_cap_group": "Mega/Large", "sector": "Consumer"},
    {"ticker": "ASIANPAINT.NS", "initial_cap_group": "Mega/Large", "sector": "Consumer"},
    {"ticker": "BAJAJFINSV.NS", "initial_cap_group": "Mega/Large", "sector": "Financials"},
    {"ticker": "WIPRO.NS", "initial_cap_group": "Mega/Large", "sector": "IT"},
    {"ticker": "TECHM.NS", "initial_cap_group": "Mega/Large", "sector": "IT"},
    {"ticker": "TMPV.NS", "initial_cap_group": "Mega/Large", "sector": "Auto"},
    {"ticker": "EICHERMOT.NS", "initial_cap_group": "Mega/Large", "sector": "Auto"},
    {"ticker": "DRREDDY.NS", "initial_cap_group": "Mega/Large", "sector": "Healthcare"},
    {"ticker": "CIPLA.NS", "initial_cap_group": "Mega/Large", "sector": "Healthcare"},
]

# 2. Mid-cap / Diversified Universe (50 tickers)
MID_DIVERSIFIED_TICKERS: List[Dict[str, str]] = [
    {"ticker": "BEL.NS", "initial_cap_group": "Mid", "sector": "Defence"},
    {"ticker": "HAL.NS", "initial_cap_group": "Mid", "sector": "Defence"},
    {"ticker": "BHEL.NS", "initial_cap_group": "Mid", "sector": "Industrials"},
    {"ticker": "TRENT.NS", "initial_cap_group": "Mid", "sector": "Consumer"},
    {"ticker": "POLYCAB.NS", "initial_cap_group": "Mid", "sector": "Industrials"},
    {"ticker": "DIXON.NS", "initial_cap_group": "Mid", "sector": "Industrials"},
    {"ticker": "PERSISTENT.NS", "initial_cap_group": "Mid", "sector": "IT"},
    {"ticker": "COFORGE.NS", "initial_cap_group": "Mid", "sector": "IT"},
    {"ticker": "MPHASIS.NS", "initial_cap_group": "Mid", "sector": "IT"},
    {"ticker": "LTIM.NS", "initial_cap_group": "Mid", "sector": "IT"},
    {"ticker": "CUMMINSIND.NS", "initial_cap_group": "Mid", "sector": "Industrials"},
    {"ticker": "ABB.NS", "initial_cap_group": "Mid", "sector": "Industrials"},
    {"ticker": "SIEMENS.NS", "initial_cap_group": "Mid", "sector": "Industrials"},
    {"ticker": "CGPOWER.NS", "initial_cap_group": "Mid", "sector": "Industrials"},
    {"ticker": "THERMAX.NS", "initial_cap_group": "Mid", "sector": "Industrials"},
    {"ticker": "ASTRAL.NS", "initial_cap_group": "Mid", "sector": "Industrials"},
    {"ticker": "VOLTAS.NS", "initial_cap_group": "Mid", "sector": "Consumer"},
    {"ticker": "KEI.NS", "initial_cap_group": "Mid", "sector": "Industrials"},
    {"ticker": "HAVELLS.NS", "initial_cap_group": "Mid", "sector": "Consumer"},
    {"ticker": "KAYNES.NS", "initial_cap_group": "Mid", "sector": "Industrials"},
    {"ticker": "MAXHEALTH.NS", "initial_cap_group": "Mid", "sector": "Healthcare"},
    {"ticker": "AUROPHARMA.NS", "initial_cap_group": "Mid", "sector": "Healthcare"},
    {"ticker": "TORNTPHARM.NS", "initial_cap_group": "Mid", "sector": "Healthcare"},
    {"ticker": "ALKEM.NS", "initial_cap_group": "Mid", "sector": "Healthcare"},
    {"ticker": "LUPIN.NS", "initial_cap_group": "Mid", "sector": "Healthcare"},
    {"ticker": "BIOCON.NS", "initial_cap_group": "Mid", "sector": "Healthcare"},
    {"ticker": "LAURUSLABS.NS", "initial_cap_group": "Mid", "sector": "Healthcare"},
    {"ticker": "FORTIS.NS", "initial_cap_group": "Mid", "sector": "Healthcare"},
    {"ticker": "MANKIND.NS", "initial_cap_group": "Mid", "sector": "Healthcare"},
    {"ticker": "ZYDUSLIFE.NS", "initial_cap_group": "Mid", "sector": "Healthcare"},
    {"ticker": "FEDERALBNK.NS", "initial_cap_group": "Mid", "sector": "Financials"},
    {"ticker": "IDFCFIRSTB.NS", "initial_cap_group": "Mid", "sector": "Financials"},
    {"ticker": "INDUSINDBK.NS", "initial_cap_group": "Mid", "sector": "Financials"},
    {"ticker": "BANKBARODA.NS", "initial_cap_group": "Mid", "sector": "Financials"},
    {"ticker": "CANBK.NS", "initial_cap_group": "Mid", "sector": "Financials"},
    {"ticker": "REC.NS", "initial_cap_group": "Mid", "sector": "Financials"},
    {"ticker": "PFC.NS", "initial_cap_group": "Mid", "sector": "Financials"},
    {"ticker": "CHOLAFIN.NS", "initial_cap_group": "Mid", "sector": "Financials"},
    {"ticker": "MUTHOOTFIN.NS", "initial_cap_group": "Mid", "sector": "Financials"},
    {"ticker": "LICHSGFIN.NS", "initial_cap_group": "Mid", "sector": "Financials"},
    {"ticker": "INDHOTEL.NS", "initial_cap_group": "Mid", "sector": "Services"},
    {"ticker": "IRCTC.NS", "initial_cap_group": "Mid", "sector": "Services"},
    {"ticker": "RVNL.NS", "initial_cap_group": "Mid", "sector": "Infrastructure"},
    {"ticker": "MAZDOCK.NS", "initial_cap_group": "Mid", "sector": "Defence"},
    {"ticker": "BEML.NS", "initial_cap_group": "Mid", "sector": "Defence"},
    {"ticker": "CONCOR.NS", "initial_cap_group": "Mid", "sector": "Services"},
    {"ticker": "NMDC.NS", "initial_cap_group": "Mid", "sector": "Metals"},
    {"ticker": "HINDCOPPER.NS", "initial_cap_group": "Mid", "sector": "Metals"},
    {"ticker": "SAIL.NS", "initial_cap_group": "Mid", "sector": "Metals"},
    {"ticker": "JINDALSTEL.NS", "initial_cap_group": "Mid", "sector": "Metals"},
]

# 3. Small / Emerging Universe (40 tickers)
SMALL_EMERGING_TICKERS: List[Dict[str, str]] = [
    {"ticker": "CDSL.NS", "initial_cap_group": "Small", "sector": "Financials"},
    {"ticker": "BSE.NS", "initial_cap_group": "Small", "sector": "Financials"},
    {"ticker": "KALYANKJIL.NS", "initial_cap_group": "Small", "sector": "Consumer"},
    {"ticker": "CREDITACC.NS", "initial_cap_group": "Small", "sector": "Financials"},
    {"ticker": "ANGELONE.NS", "initial_cap_group": "Small", "sector": "Financials"},
    {"ticker": "MCX.NS", "initial_cap_group": "Small", "sector": "Financials"},
    {"ticker": "CAMS.NS", "initial_cap_group": "Small", "sector": "Financials"},
    {"ticker": "NUVAMA.NS", "initial_cap_group": "Small", "sector": "Financials"},
    {"ticker": "SONACOMS.NS", "initial_cap_group": "Small", "sector": "Auto"},
    {"ticker": "CRAFTSMAN.NS", "initial_cap_group": "Small", "sector": "Auto"},
    {"ticker": "ENDURANCE.NS", "initial_cap_group": "Small", "sector": "Auto"},
    {"ticker": "SUPRAJIT.NS", "initial_cap_group": "Small", "sector": "Auto"},
    {"ticker": "JBMA.NS", "initial_cap_group": "Small", "sector": "Auto"},
    {"ticker": "OLECTRA.NS", "initial_cap_group": "Small", "sector": "Auto"},
    {"ticker": "DATAPATTNS.NS", "initial_cap_group": "Small", "sector": "Defence"},
    {"ticker": "IDEAFORGE.NS", "initial_cap_group": "Small", "sector": "Defence"},
    {"ticker": "KPITTECH.NS", "initial_cap_group": "Small", "sector": "IT"},
    {"ticker": "BSOFT.NS", "initial_cap_group": "Small", "sector": "IT"},
    {"ticker": "AFFLE.NS", "initial_cap_group": "Small", "sector": "IT"},
    {"ticker": "NEWGEN.NS", "initial_cap_group": "Small", "sector": "IT"},
    {"ticker": "TANLA.NS", "initial_cap_group": "Small", "sector": "IT"},
    {"ticker": "ROUTE.NS", "initial_cap_group": "Small", "sector": "IT"},
    {"ticker": "CYIENT.NS", "initial_cap_group": "Small", "sector": "IT"},
    {"ticker": "AAVAS.NS", "initial_cap_group": "Small", "sector": "Financials"},
    {"ticker": "KARURVYSYA.NS", "initial_cap_group": "Small", "sector": "Financials"},
    {"ticker": "RBLBANK.NS", "initial_cap_group": "Small", "sector": "Financials"},
    {"ticker": "CESC.NS", "initial_cap_group": "Small", "sector": "Utilities"},
    {"ticker": "SJVN.NS", "initial_cap_group": "Small", "sector": "Utilities"},
    {"ticker": "IREDA.NS", "initial_cap_group": "Small", "sector": "Financials"},
    {"ticker": "KPI.NS", "initial_cap_group": "Small", "sector": "Utilities"},
    {"ticker": "APLAPOLLO.NS", "initial_cap_group": "Small", "sector": "Metals"},
    {"ticker": "WELCORP.NS", "initial_cap_group": "Small", "sector": "Metals"},
    {"ticker": "JSL.NS", "initial_cap_group": "Small", "sector": "Metals"},
    {"ticker": "JINDALSAW.NS", "initial_cap_group": "Small", "sector": "Metals"},
    {"ticker": "GRAPHITE.NS", "initial_cap_group": "Small", "sector": "Industrials"},
    {"ticker": "RATNAMANI.NS", "initial_cap_group": "Small", "sector": "Metals"},
    {"ticker": "ASTRAMICRO.NS", "initial_cap_group": "Small", "sector": "Defence"},
    {"ticker": "GRSE.NS", "initial_cap_group": "Small", "sector": "Defence"},
    {"ticker": "COCHINSHIP.NS", "initial_cap_group": "Small", "sector": "Defence"},
    {"ticker": "APARINDS.NS", "initial_cap_group": "Small", "sector": "Industrials"},
]

# Benchmark Indices
MARKET_BENCHMARK = "^NSEI"  # NIFTY 50

# Sector Indices map (Yahoo Finance symbols on NSE)
SECTOR_INDICES: Dict[str, str] = {
    "Financials": "^NSEBANK",
    "IT": "^CNXIT",
    "Energy": "^CNXENERGY",
    "Auto": "^CNXAUTO",
    "Healthcare": "^CNXPHARMA",
    "Metals": "^CNXMETAL",
    "Consumer": "^CNXFMCG",
    "Infrastructure": "^CNXINFRA",
    "Utilities": "^CNXENERGY",     # Best proxy where utility index is consolidated
    "Industrials": "^CNXINFRA",    # Best proxy for capital goods / infra
    "Defence": "^CNXINFRA",        # Best proxy for heavy engineering / defence
    "Materials": "^CNXMETAL",      # Best proxy for commodities / materials
    "Services": "^CNXINFRA",
    "Telecom": "^CNXINFRA"
}


def get_universe_df() -> pd.DataFrame:
    """Returns the unified universe DataFrame with ticker, initial_cap_group, and sector."""
    all_stocks = MEGA_LARGE_TICKERS + MID_DIVERSIFIED_TICKERS + SMALL_EMERGING_TICKERS
    df = pd.DataFrame(all_stocks)
    return df
