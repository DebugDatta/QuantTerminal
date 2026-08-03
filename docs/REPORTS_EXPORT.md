# Reports & Export

Three export formats: CSV, Excel, PDF. All data derived from computed analytics (no raw external data).

> **Dependencies**: Chart embedding in Excel and PDF requires `kaleido` (Plotly → static PNG) and `Pillow` (image handling). If these are not installed, exports fall back to tables-only (no charts), and a warning is displayed.
>
> **Currency in reports**: Values are displayed with the detected currency symbol (₹ for Indian NSE/BSE tickers, $ for US, € for EU, etc.). In Excel/PDF reports, the currency symbol is included in the column headers (e.g., "Close (₹)", "PnL ($)").

---

## CSV Export (`reports/csv.py`)

### `export_csv(data, filename)`

**Parameters:**
| Name | Type | Description |
|---|---|---|
| data | pd.DataFrame or dict of DataFrames | Data to export |
| filename | str | Output filename (.csv) |

- Single DataFrame → single CSV file
- Dict of DataFrames → multiple CSV files, filenames derived from keys
- All CSVs saved to a timestamped folder in `exports/`

### Supported Export Data
| Report Section | Data Content |
|---|---|
| Historical Prices | OHLCV table |
| Returns | Daily/weekly/monthly returns table |
| Technical Indicators | Indicator values table |
| Statistical Tests | Test results table |
| Risk Metrics | Risk metrics table |
| Portfolio Weights | Weight allocations |
| Trade Log | All executed trades |
| Backtest Metrics | Performance summary |

---

## Excel Export (`reports/excel.py`)

### `export_excel(report_data, filename)`

Generates multi-sheet Excel workbook using `openpyxl`.

**Parameters:**
| Name | Type | Description |
|---|---|---|
| report_data | dict | Section name → DataFrame |
| filename | str | Output filename (.xlsx) |

### Sheet Structure
| Sheet | Content |
|---|---|
| Executive Summary | Key metrics, total return, Sharpe, Max DD |
| Statistics | Summary statistics, test results, volatility estimates, GARCH coefficients |
| Returns | Return tables, CAGR |
| Technical Analysis | Indicator values, signals |
| Risk | Risk metrics, VaR, drawdown |
| Portfolio | Weights, risk contribution |
| Backtest | Trade log, performance metrics, equity curve |
| Forecasting | Forecast values, confidence intervals |
| ML Results | ML model metrics, feature importance |
| Regimes | Regime labels, state probabilities |
| Simulation | Monte Carlo percentiles, terminal distribution |
| Factors | Factor returns, IC time series |
| Pairs | Cointegration results, pair rankings |
| Charts | Embedded chart images |

### Formatting
- Headers: bold, frozen
- Numbers: percentage format for returns, 2 decimal places for prices
- Conditional formatting: positive returns green, negative red
- Auto-adjusted column widths
- **Confidence badges** shown next to model outputs (GARCH, HMM, ARIMA, ML, cointegration, PCA). In Excel, rendered as colored fill instead of emoji:
  - 🟢 **High** → Green fill (#C6EFCE)
  - 🟡 **Medium** → Yellow fill (#FFEB9C)
  - 🔴 **Low** → Red fill (#FFC7CE)

---

## PDF Export (`reports/pdf.py`)

### `export_pdf(report_data, filename, include_sections=None)`

Generates PDF report using `reportlab`.

**Parameters:**
| Name | Type | Description |
|---|---|---|
| report_data | dict | Section name → (DataFrame, Chart) |
| filename | str | Output filename (.pdf) |
| include_sections | list | Subset of sections to include |

### Sections

#### Executive Summary
- Ticker, date range
- Total Return, CAGR, Sharpe, Sortino, Max Drawdown
- Current vs benchmark comparison

#### Statistics Section
- Summary statistics table
- Stationarity test results
- Normality test results
- Return distribution description
- Volatility estimates table (6 estimators)
- GARCH model coefficients and diagnostics

#### Technical Analysis Section
- Indicator values table (last N periods)
- Signal summary
- Chart: price with indicator overlay

#### Risk Section
- Risk metrics table
- VaR/CVaR summary
- Chart: drawdown curve, underwater plot

#### Portfolio Section
- Weight allocation table
- Risk contribution table
- Chart: allocation pie, efficient frontier

#### Strategy Performance Section
- Backtest metrics table
- Trade log summary
- Charts: equity curve, monthly returns heatmap

#### Forecasting Section
- Forecast values table with confidence intervals
- Residual diagnostics
- Charts: forecast with CI bands, residual ACF

#### Machine Learning Section
- Model metrics table (R², RMSE, MAE)
- Feature importance chart
- Charts: predicted vs actual scatter, residuals

#### Regime Detection Section
- Regime summary per segment
- Transition matrix (HMM)
- Chart: regime timeline on price

#### Monte Carlo Section
- Path percentiles table
- Terminal distribution statistics
- Charts: fan chart, terminal histogram

#### Factor Research Section
- Factor returns table (long/short/spread)
- IC time series summary
- Charts: cumulative factor returns, IC

#### Statistical Arbitrage Section
- Top pair rankings table
- Cointegration test results
- Charts: spread with z-score bands

### PDF Layout
- Title page with project name, ticker, date range
- Each section starts on a new page
- Tables with alternating row colors
- Charts embedded as PNG images
- Page numbers, header with ticker name
- Footer with generation date and data source

---

## Export Workflow

```
User selects sections to include
         │
         ▼
Generate all selected analytics
         │
         ▼
    ┌──── EXPORT FORMAT (single-select) ────┐
    │                                        │
    ▼                                        ▼
┌─────────┐                           ┌────────────┐
│   CSV   │    ──────────────────►    │  Excel or  │
│ (files) │                           │   PDF      │
└─────────┘                           │ (single    │
                                      │  file)     │
                                      └────────────┘
         │                                        │
         ▼                                        ▼
  Folder per section                       Single output file
  (exports/csv/...)                        (exports/excel/... or exports/pdf/...)
         │                                        │
         └────────────────┬───────────────────────┘
                          ▼
                    Download file
```

---

## Export Location

All exports are saved to the `exports/` directory:
```
exports/
├── csv/
│   └── quantterminal_20260101_120000/
│       ├── prices.csv
│       ├── returns.csv
│       └── ...
├── excel/
│   └── quantterminal_report_20260101_120000.xlsx
└── pdf/
    └── quantterminal_report_20260101_120000.pdf
```
