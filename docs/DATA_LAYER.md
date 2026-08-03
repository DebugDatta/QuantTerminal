# Data Layer

## Source

Yahoo Finance via the `yfinance` Python library.

Data fields downloaded:
| Field | Description |
|---|---|
| Open | Opening price |
| High | Highest price |
| Low | Lowest price |
| Close | Adjusted close price |
| Volume | Trading volume |

## Indian Market Support

Indian equities on Yahoo Finance use exchange-specific suffixes:

| Exchange | Suffix | Benchmark Index | Example |
|---|---|---|---|
| **NSE** (National Stock Exchange) | `.NS` | `^NSEI` (Nifty 50) | `RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS` |
| **BSE** (Bombay Stock Exchange) | `.BO` | `^BSESN` (Sensex) | `RELIANCE.BO`, `500325.BO` |

### Ticker Resolution (`resolve_ticker`)

The app auto-detects the correct suffix, or you can set it explicitly:

| Exchange Setting | Behavior |
|---|---|
| **Auto** (default) | Try `.NS` → if no data → try raw → if no data → try `.BO` |
| **NSE** | Append `.NS` to the symbol |
| **BSE** | Append `.BO` to the symbol |
| **Global** | Use symbol as-is (no suffix) |

```python
resolve_ticker("RELIANCE", "auto")   # → RELIANCE.NS
resolve_ticker("TCS", "NSE")         # → TCS.NS
resolve_ticker("AAPL", "global")     # → AAPL
resolve_ticker("^NSEI", "auto")      # → ^NSEI (indices kept as-is)
```

### Currency Detection

`detect_currency(ticker)` reads `yf.Ticker(ticker).info.get("currency")` for display formatting only.

| Detected | Symbol | Examples |
|---|---|---|
| INR | ₹ | RELIANCE.NS, TCS.NS, HDFCBANK.NS |
| USD | $ | AAPL, MSFT, SPY |
| EUR | € | SAP.DE, AIR.PA |
| GBP | £ | BP.L, HSBA.L |

> Note: Currency is cosmetic for single-asset analytics. For multi-asset portfolio PnL, FX conversion is applied (see below).

### FX Rate Lookup (`get_fx_rate`)

```python
get_fx_rate(from_ccy, to_ccy, date=None)
```

Pulls exchange rates via yfinance currency pair tickers (e.g., `USDINR=X` for USD → INR).

**Behavior:**
- If `date` is None, returns the latest close rate
- If `date` is specified, returns the historical close rate on that date
- If `from_ccy == to_ccy`, returns 1.0
- Pair ticker format: `{from_ccy}{to_ccy}=X` (e.g., `EURINR=X`, `GBPUSD=X`)

**Base Currency Configuration** (`config.py`):
```python
BASE_CURRENCY = "INR"   # All portfolio values converted to this currency
SUPPORTED_CURRENCIES = ["INR", "USD", "EUR", "GBP", "JPY", "SGD"]
```

### Indian Benchmark Indices

| Index | Yahoo Ticker |
|---|---|
| Nifty 50 | `^NSEI` |
| Sensex | `^BSESN` |
| Bank Nifty | `^NSEBANK` |
| Nifty Next 50 | `^NSMIDCP` |

### Indian ETF Examples

| ETF | Yahoo Ticker |
|---|---|
| Nippon India ETF Nifty 50 BeES | `NIFTYBEES.NS` |
| Nippon India ETF Gold BeES | `GOLDBEES.NS` |
| Nippon India ETF Junior BeES | `JUNIORBEES.NS` |

## Loader API (`data/loader.py`)

### `load_data(tickers, start, end, exchange="auto", progress_callback=None)`

Downloads OHLCV data for one or more tickers.

**Parameters:**
- `tickers` — str or list of str. Single ticker or list (e.g., `"AAPL"` or `["AAPL", "MSFT", "SPY"]`)
- `start` — str or datetime. Start date (e.g., `"2020-01-01"`)
- `end` — str or datetime. End date (default: today)
- `exchange` — str: `"auto"`, `"NSE"`, `"BSE"`, or `"global"` (default: `"auto"`)
- `progress_callback` — callable, optional. Called with (current, total) for progress tracking

**Returns:**
- Single ticker: `pd.DataFrame` with columns `[Open, High, Low, Close, Volume]`
- Multiple tickers: `pd.DataFrame` with `pd.MultiIndex` columns `[(ticker, field), ...]`

**Error handling:**
- Invalid tickers are logged and excluded from results
- If no valid tickers remain, raises `ValueError`
- Network errors retry up to 3 times with exponential backoff

### `search_tickers(query, market="all")`

Searches Yahoo Finance for matching tickers.

**Parameters:**
- `query` — str. Search term (e.g., `"Apple"`, `"Reliance"`)
- `market` — str: `"all"`, `"india"` (filters .NS/.BO), `"global"` (excludes .NS/.BO)

**Returns:**
- List of dicts: `[{"symbol": "...", "name": "...", "type": "...", "exchange": "..."}]`

## Caching (`data/cache.py`)

### `cached_load(tickers, start, end, ttl_hours=24)`

Wrapper around `load_data` with disk caching.

**How it works:**
1. Hash of `(tickers, start, end)` → cache key
2. If cached file exists and is within TTL, load from disk
3. Otherwise, call `load_data`, save to disk, return

**Cache location:** `~/.quantterminal_cache/`

### `clear_cache()`

Deletes all cached data files.

### `cache_info()`

Returns dict with: `{cached_tickers, cached_dates, cache_size_mb}`

## Resampling (`data/resample.py`)

### `resample_ohlcv(df, freq)`

Resamples OHLCV data to a different frequency.

**Parameters:**
- `df` — OHLCV DataFrame
- `freq` — str: `"W"` (weekly), `"M"` (monthly), `"Q"` (quarterly), `"Y"` (annual)

**Resampling rules:**
- Open → first value
- High → max
- Low → min
- Close → last value
- Volume → sum

## Supported Ticker Formats

| Type | Global Example | Indian Example |
|---|---|---|
| Large Cap Stocks | AAPL, MSFT | RELIANCE.NS, TCS.NS |
| Mid/Small Cap | ROKU, SQ | TITAN.NS, BAJFINANCE.NS |
| Indices | ^GSPC, ^DJI | ^NSEI, ^BSESN |
| ETFs | QQQ, VTI | NIFTYBEES.NS, GOLDBEES.NS |
| Forex | EURUSD=X | USDINR=X |
| Crypto | BTC-USD | — |
| Commodities | CL=F | — |
| International | 0700.HK, BP.L | — |

## Data Quality Notes

- Yahoo Finance provides **adjusted close** prices (dividend/split-adjusted)
- Volume is raw (non-adjusted)
- Indian market data may have NaN rows on NSE holidays (Republic Day, Diwali, etc.). Use `drop_holiday_nans()` to clean.
- Crypto and forex data may have gaps on weekends
- Some international/Indian tickers may have limited history
- Rate limits: Yahoo may throttle bulk requests. Use multi-ticker downloads and caching.
- **Survivorship bias**: Yahoo Finance only serves currently-listed tickers. Historical backtests on universes filtered by today's constituents will exclude delisted/bankrupt tickers, inflating apparent returns. This is inherent to the data source and cannot be fully eliminated.
- **Point-in-time guard**: Fundamental and snapshot values are *as-of-download*, not a point-in-time database (see `BIAS_MITIGATION.md` §B1). They must **never** be used to construct historical signals; they are restricted to live screeners and valuation overlays.
- **Backfill / universe reconstitution**: index/factor membership is taken from today's constituents; new entrants' history is treated as present (backfill) — label such studies "current-constituents only."
- **Selection bias**: the snapshot universe is biased toward liquid, large-cap, exchange-listed names; disclose this in screener results.
- **Adjusted close vs H/L misalignment**: yfinance back-adjusts Close for dividends but does **not** back-adjust High, Low, or Open. Indicators mixing Close with H/L (ATR, Bollinger Bands, Keltner Channels, OBV, CMF, ADL) may show small artifacts around ex-dividend dates. The effect is typically negligible for daily-frequency analysis.

## Calendar Alignment

Multi-asset operations (correlation, covariance, portfolio optimization, pairs) use **inner-join** on trading days:
- Any date where any ticker has NaN Close is excluded
- This avoids artifacts from forward-filling holidays or non-trading days
- Mixed India+Global universes will have fewer available trading days than either market alone

## Reliability

yfinance uses an unofficial Yahoo Finance endpoint that may break or be rate-limited without notice. If a fetch fails:

| Cache Age | Behavior |
|---|---|
| < 1 trading day | Silently serve cache (fresh) |
| 1–5 trading days | Serve cache + **yellow staleness banner** |
| > 5 trading days | Serve cache + **red staleness banner** + warning icon on charts |
| > 30 trading days | Block model fitting (GARCH, HMM, ARIMA) — return error: "Data too stale for reliable estimation" |

If no cache exists, a graceful error message is displayed (no crashes).

---

## Fundamentals & Snapshot Layer

Two modules supply non-OHLCV data (fundamentals, valuations, technical presets) with a strict no-hardcode, no-silent-blank policy.

### `data/fundamentals.py` — Safe yfinance accessors

Wraps **every** yfinance surface so a failure on any single field never crashes the app.

| yfinance surface | Loader |
|---|---|
| `.info`, `.fast_info` | `get_info(ticker)` |
| `.income_stmt`, `.quarterly_income_stmt` | `get_income_stmt(ticker)` |
| `.balance_sheet`, `.quarterly_balance_sheet` | `get_balance_sheet(ticker)` |
| `.cashflow`, `.quarterly_cashflow` | `get_cashflow(ticker)` |
| `.dividends`, `.splits`, `.actions` | `get_actions(ticker)` |
| `.earnings_dates`, `.calendar` | `get_calendar(ticker)` |
| `.recommendations`, `.recommendation_summary` | `get_recommendations(ticker)` |
| `.analyst_price_targets` | `get_targets(ticker)` |
| `.news` | `get_news(ticker)` |

Common signature:

```python
get_field(ticker, field, sink=SnapshotData, default=None)
    -> value | Unavailable(field, reason)
```

**Access order (per field):**
1. Try live yfinance (`try/except` wrapping `yf.YFError`, `KeyError`, `TypeError`, connection errors).
2. If absent/invalid, look up the same field in the snapshot dataset (see below).
3. If still absent, return the structured sentinel `Unavailable(field, reason)`.

**Provenance flag:** every resolved field carries a provenance dict `{"source": "yfinance"|"snapshot", "point_in_time": false}`. `point_in_time` is `false` for all fundamental/snapshot values, so backtests reject any historical use of them (see `docs/BIAS_MITIGATION.md` §B1).

**Sentinel:** `Unavailable` is a namedtuple `(field, reason)`. The UI renders it as a labeled message — **"Data not available for \<ticker\>"** — never a blank cell or `NaN`.

**No hardcoding:** The symbol ↔ suffix mapping, thresholds, and schemas are read from `config.py` and the dataset metadata, not literals. yfinance values are read via `.get(key)`-style access with safe numeric coercion — never positional indexing.

### `data/snapshot.py` — Local snapshot datasets

Loads the precomputed NSE/BSE snapshot files held under `data/snapshots/` (the full Indian equity universe). These make the screener and composite scoring **fast and complete** without a per-ticker bulk call (which would hit rate limits). `model_comparison_summary.csv` is a model-result artifact and is **not** part of the snapshot layer.

| Dataset | Content (representative fields) |
|---|---|
| **issuers** | Security code, NSE/BSE symbol, security id, status, group, face value, ISIN, industry, sector |
| **main** | Sector/Industry, market cap, current price, quarterly/annual revenue & net profit, margins, growth, cash-flow |
| **financials** | Revenue, profits, margins, growth (YoY/QoQ), cash-flow detail |
| **valuations** | P/E (TTM/forward/3Y/5Y), PEG, P/B, EPS, ROE, ROA, Piotroski score, sector/industry ratios |
| **technicals** | Precomputed RSI, MACD, SMA/EMA (5..200), ATR, ADX, Beta, ROC, MFI, momentum scores |
| **performance** | Relative returns vs Nifty 50 / Sensex / sector / industry across week..10Y |

### File inventory (`data/snapshots/`)

| File | Sheet | Rows | Cols | Snapshot kind |
|---|---|---|---|---|
| `equity_financials.xlsx` | Data Downloader | 5630 | 36 | financials |
| `equity_performance.xlsx` | Data Downloader | 5630 | 37 | performance |
| `equity_technicals.xlsx` | Data Downloader | 5630 | 35 | technicals |
| `equity_valuations.xlsx` | Data Downloader | 5630 | 37 | valuations |
| `equity_issuers.xlsx` | Data Downloader | 5191 | 11 | issuer |
| `equity_main.xlsx` | Sheet1 | 5630 | 135 | main (merged) |
| `equity_main1.csv` | — | 5630 | 114 | main (CSV fallback) |
| `equity_issuers1.csv` | — | 4193 | 14 | issuer (CSV fallback) |

> `equity_main.xlsx` is a **merged** view (financials + performance + technicals + valuations concatenated with `.1/.2/.3` duplicate-column suffixes). Treat it as the merged source of truth after **de-duplicating** those suffix columns; the individual files are lighter per-kind sources. `equity_main1.csv` has stray `Unnamed:` columns and a partial `sector_name` split — prefer the xlsx where detail matters.

#### Join keys & ticker mapping (critical)

- **Primary cross-file key** is `Stock Code` (in the `equity_*.xlsx` files) **⇄ `Security Id`** (in `equity_issuers.xlsx`). Both hold the NSE **symbol** (e.g., `INFY`, `TCS`); overlap is ~4,830 of ~5,630 names.
- **Do NOT join on** `Security Code` (issuers) — that is a *numeric* BSE/NSE code with **no overlap** with `Stock Code`.
- Resolve to a live yfinance ticker:
  - **NSE:** `Stock Code + ".NS"` → `INFY.NS`. `Stock Code` is the most complete symbol column (5,392/5,630 non-null; the `NSE Code` column is only ~2,884 and is redundant).
  - **BSE:** `BSE Code + ".BO"` (numeric scrip code, e.g., `500209.BO`) — 4,943 present.
  - Coverage: 1 row is missing both NSE and BSE symbols; ISIN is null for ~184 rows.

`map_snapshot_to_tickers()` builds the mapping from `Stock Code` (NSE) or `BSE Code` (BSE) using the join above.

```python
load_snapshot(kind) -> pd.DataFrame            # issuer / main / financials / valuations / technicals / performance
snapshot_field(ticker, field) -> value | None
map_snapshot_to_tickers(exchange="NSE") -> dict  # dataset security code / symbol -> resolved yfinance ticker
```

**Live-enrichment (`snapshot_field` + `get_field`):** Bulk screens read the snapshot only (fast). When a selected row needs a fresher or missing value, `get_field` runs the yfinance-first access chain for that single ticker (never the whole universe).

**Data-source toggle** (`config.py`):

```python
DATA_SOURCE = "snapshot"        # "snapshot" | "snapshot+live" | "live"
SCREENER_RULES_ONLY_FROM = "dataset"   # screener rules read from dataset, not hardcoded
```

### Snapshot staleness baseline

The snapshot is **point-in-time**, not live. Its effective date is bounded by the result metadata:
- `Latest financial result` ≤ **2025-06-30** (max value date in `equity_financials.xlsx` / `equity_main.xlsx`).
- `Result Announced Date` ≤ **2025-08-26**.

Consequently, every snapshot-derived fundamental/valuation field should render as 🟡 **stale** until the snapshot is refreshed, per the availability-badge rules below and `docs/MODEL_CONFIDENCE.md`.

### Availability badges

Every field rendered from this layer carries an availability badge (see `docs/MODEL_CONFIDENCE.md`):

| Source | Badge |
|---|---|
| Live yfinance value | 🟢 |
| Snapshot fallback value | 🟡 (stale — current snapshot is as-of Aug 2025; refresh to update) |
| Unavailable | 🔴 + "Data not available for \<ticker\>" |
