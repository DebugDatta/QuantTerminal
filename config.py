"""Central configuration: exchange options, benchmarks, currency symbols, presets."""

EXCHANGE_OPTIONS = ["Auto", "NSE", "BSE", "GLOBAL"]

BENCHMARKS = {
    "India": {
        "Nifty 50": "^NSEI",
        "Sensex": "^BSESN",
        "Bank Nifty": "^NSEBANK",
    },
    "Global": {
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC",
        "None": None,
    },
}

CURRENCY_SYMBOLS = {
    "INR": "\u20b9",
    "USD": "$",
    "EUR": "\u20ac",
    "GBP": "\u00a3",
    "JPY": "\u00a5",
    "SGD": "S$",
}

INDIAN_PRESETS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "ITC.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "HCLTECH.NS",
    "SUNPHARMA.NS", "TATAMOTORS.NS", "WIPRO.NS", "ULTRACEMCO.NS", "ONGC.NS",
]

MARKET_CAP_THRESHOLDS_INR = {
    "Large Cap": 7.5e11,
    "Mid Cap": 2e11,
    "Small Cap": 1e10,
}

MARKET_CAP_THRESHOLDS_USD = {
    "Large Cap": 1e11,
    "Mid Cap": 2e10,
    "Small Cap": 2e9,
}

DEFAULT_RISK_FREE_RATE = 0.05
TRADING_DAYS_PER_YEAR = 252
