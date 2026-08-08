# Task Division & Project Plan

QuantTerminal — build-from-docs execution plan. Three equal workstreams, one shared page team.

## Team

| Person | Background | Role |
|---|---|---|
| **Michael Fernandez** | MSc BDA | Engine + advanced ML, lead/architect, integration owner |
| **Pranav Sahai** | 2nd-year BSc Statistics | Statistics, risk, quant research |
| **Aashima Grover** | 2nd-year BSc IT | Technicals, visualization, portfolio, reporting |

## Balance

| | Module files | Pages |
|---|---|---|
| Michael | 24 / 71 | 6 |
| Pranav | 23 / 71 | 6 |
| Aashima | 24 / 71 | 6 |

All function signatures are pinned in `docs/ARCHITECTURE.md` (directory tree) and `docs/DATA_LAYER.md` (loader/cache/snapshot API). Build against those contracts — no design changes.

---

## Per-Person Deliverables

### Michael — Engine + Advanced ML (24 files, 6 pages)

| Module | Files | From docs |
|---|---|---|
| Foundation | `data/loader.py`, `data/cache.py`, `data/resample.py`, `config.py`, `requirements.txt`, `app.py`, `utils/decorators.py` | ARCHITECTURE, DATA_LAYER |
| Strategies | `strategies/base.py`, `strategies/builtin.py`, `strategies/signals.py` | STRATEGIES_BACKTESTING |
| Backtesting | `backtesting/engine.py`, `backtesting/metrics.py`, `backtesting/trades.py`, `backtesting/optimization.py` | STRATEGIES_BACKTESTING |
| Forecasting | `forecasting/arima.py`, `forecasting/exponential.py`, DL forecasters (RNN/LSTM/GRU) | FORECASTING_ML |
| Machine Learning | `machine_learning/models.py`, `machine_learning/features.py`, `machine_learning/evaluation.py` | FORECASTING_ML |
| Regime | `regime/hmm.py`, `regime/gmm.py`, `regime/change_point.py` | REGIME_SIMULATION |
| Simulation | `simulation/gbm.py`, `simulation/bootstrap.py`, `simulation/portfolio_sim.py` | REGIME_SIMULATION |

Pages: 12 Backtesting, 13 Forecasting, 14 Machine Learning, 15 Regime Detection, 16 Monte Carlo, 18 Settings.

Bias controls to honor: B6 (backtest realism), B7 (overfitting discipline), B9 (leakage-proof ML), B14 (DL guards), B15 (RL guards).

### Pranav — Statistics & Risk (23 files, 6 pages)

| Module | Files | From docs |
|---|---|---|
| Core | `core/returns.py`, `core/metrics.py`, `core/drawdown.py` | ARCHITECTURE |
| Statistics | `statistics/summary.py`, `stationarity.py`, `diagnostics.py`, `distributions.py`, `correlation.py`, `pca.py`, `clustering.py`, `timeseries.py` | STATISTICAL_MODELS |
| Volatility | `volatility/estimators.py`, `volatility/garch.py` | RISK_ANALYTICS |
| Risk | `risk/metrics.py`, `risk/rolling.py` | RISK_ANALYTICS |
| Factor | `factor/factors.py`, `factor/scores.py` | FACTOR_RESEARCH |
| StatArb | `statarb/pairs.py`, `statarb/cointegration.py`, `statarb/spread.py` | STATISTICAL_MODELS |
| Reports | `reports/csv.py`, `reports/excel.py`, `reports/pdf.py` | REPORTS_EXPORT |

Pages: 3 Return Analytics, 5 Statistical Analysis, 6 Volatility Lab, 7 Risk Analytics, 9 Factor Research, 10 Statistical Arbitrage.

Bias controls to honor: B2 (robust SEs), B4 (spurious-regression gate), B5 (risk inference), B10 (drift mgmt).

### Aashima — Technicals, Visuals, Portfolio (24 files, 6 pages)

| Module | Files | From docs |
|---|---|---|
| Technical | `technical/trend.py`, `momentum.py`, `volatility.py`, `volume.py`, `strength.py`, `signals.py` | TECHNICAL_INDICATORS |
| Plots (shared layer) | `plots/candlestick.py`, `indicators.py`, `distributions.py`, `returns.py`, `risk.py`, `portfolio.py`, `correlation.py`, `timeseries.py`, `clustering.py`, `simulation.py` | ARCHITECTURE |
| Portfolio | `portfolio/builder.py`, `portfolio/risk_contribution.py` | PORTFOLIO_OPTIMIZATION |
| Optimization | `optimization/mean_variance.py`, `risk_parity.py`, `hrp.py`, `frontier.py` | PORTFOLIO_OPTIMIZATION |
| Utils | `utils/helpers.py` | ARCHITECTURE |

Pages: 1 Dashboard, 2 Market Explorer, 4 Technical Analysis, 8 Portfolio Lab, 11 Strategy Lab, 17 Reports.

Bias controls to honor: B3 (VIF), B8 (parameter & structural), B13 (interpretability/reproducibility).

---

## Step-by-Step Timeline (30 days)

### Phase 0 — Kickoff & conventions (½ day · all three)
- [ ] Create `venv`, `pip install -r requirements.txt`, verify `yfinance` fetch for `RELIANCE.NS` + `AAPL`.
- [ ] Branches: `michael-engine`, `pranav-stats`, `aashima-tech` off `main` (main protected).
- [ ] Freeze contracts: walk `docs/ARCHITECTURE.md` + `docs/DATA_LAYER.md`.
- [ ] Conventions: `@st.cache_data`, MultiIndex `(ticker, field)`, `render_page()` template, no-hardcode policy, availability badges 🟢/🟡/🔴, "historical, not forecast" banner.
- **Exit:** interface cheat-sheet committed; everyone can load OHLCV.

### Phase 1 — Foundation sprint (Days 1–3 · Michael + Pranav in parallel)
- Michael: `data/`, `config.py`, `requirements.txt`, `app.py` skeleton, `utils/`.
- Pranav (starts immediately, no deps): `core/` (returns, metrics, drawdown).
- Aashima: `technical/` trend + momentum (independent of foundation).
- **Exit:** `load_data` + core metrics round-trip verified NSE + global; app shell renders.

### Phase 2 — Module builds (Days 4–14 · three parallel tracks)
Suggested order (dependency-ordered):

| Track M — Michael | Track P — Pranav | Track A — Aashima |
|---|---|---|
| `strategies/` | `statistics/` | `technical/` (full) |
| `backtesting/` engine+metrics+trades | `volatility/` | `plots/` base (candlestick, indicators, returns) |
| `backtesting/optimization.py` | `risk/` | `portfolio/` + `optimization/` |
| `forecasting/` (ARIMA + DL) | `factor/` | `plots/` full |
| `machine_learning/` | `statarb/` | `utils/helpers.py` |
| `regime/`, `simulation/` | `reports/` | |

Each module done when: functions match docs, edge cases handled (holiday NaNs, staleness banners, `Unavailable` sentinel), quick self-test passes.
- **Exit:** all 68 module files importable; one demo script per track.

### Phase 3 — Cross-track sync (Days 15–16 · all three)
- Backtest smoke test: SMA Cross on `RELIANCE.NS` through `backtesting/engine.py` (validates Michael's engine + Pranav's `core/`).
- Page-11 interface check: Aashima confirms `strategies/` API.
- Plots sign-off: Aashima's `plots/` verified against all module return types.
- **Exit:** integration script passes; no interface mismatches.

### Phase 4 — Shared page team (Days 17–23 · 6 pages each)
| Michael | Pranav | Aashima |
|---|---|---|
| 12 Backtesting | 3 Return Analytics | 1 Dashboard |
| 13 Forecasting | 5 Statistical Analysis | 2 Market Explorer |
| 14 ML | 6 Volatility Lab | 4 Technical Analysis |
| 15 Regime | 7 Risk Analytics | 8 Portfolio Lab |
| 16 Monte Carlo | 9 Factor Research | 11 Strategy Lab |
| 18 Settings | 10 StatArb | 17 Reports |

Cross-reviews: each person reviews 2 pages by other owners vs `docs/STREAMLIT_PAGES.md`.
- **Exit:** all 18 pages render with live NSE data + availability badges.

### Phase 5 — Integration & bias audit (Days 24–27 · Michael leads)
- Merge branches; fix cross-module issues; complete `app.py` routing.
- Walk the Part C ownership matrix (`docs/BIAS_MITIGATION.md` line 286): verify each control → module → page (B1 point-in-time guard, B4 gate, B6 realism, B7 discipline, B9 leakage, etc.).
- Staleness banner + cache TTL verified; full smoke test of all 18 pages.
- **Exit:** app runs end-to-end, no silent errors, bias checklist ticked.

### Phase 6 — Tests, docs, demo (Days 28–30 · all three)
- Add `pytest` suite (foundation/core, stats, backtest, ML) with synthetic + real data.
- Update `README.md`; this file is the living plan/checklist.
- Demo walkthrough, final review, commit to `main`.

**Milestones:** M0 end Day 3 · M1 end Day 14 · M2 end Day 23 · M3 end Day 30.

---

## Dependency Map (who waits on whom)

```
data/ + core/ + config.py  (Michael, Pranav)   ← everything depends on these
   │
   ├─ technical/ + plots/ + utils/  (Aashima)   ← plots/ is the shared page glue
   ├─ statistics/ + volatility/ + risk/ (Pranav)
   ├─ factor/ + statarb/ + reports/ (Pranav)     ← depends on statistics/, core/
   ├─ strategies/ + backtesting/ (Michael)       ← depends on technical/, core/
   └─ forecasting/ + ml/ + regime/ + simulation/ (Michael) ← depends on data/
        │
        ▼
   Streamlit pages (all three)  ← consume modules + plots/ via render_page() template
```

## Shared Boundaries to Coordinate
- **Page 1 Dashboard**: Aashima's UI, consumes Pranav's `core/` metrics.
- **Page 11 Strategy Lab**: Aashima's UI, consumes Michael's `strategies/` interface.
- **`plots/`**: Aashima's module, consumed by every page — sign-off in Phase 3.

## Bias-Audit Checklist (Part C — `docs/BIAS_MITIGATION.md` line 286)
- [ ] B1 point-in-time guard: `data/fundamentals.py`, `data/snapshot.py` → pages 1, 2, 9
- [ ] B2 robust SEs: `factor/scores.py`, `statistics/*`, `machine_learning/evaluation.py` → 5, 9, 14
- [ ] B3 VIF: `machine_learning/features.py`, `factor/factors.py` → 9, 14
- [ ] B4 spurious-regression gate: `statistics/stationarity.py`, `statarb/cointegration.py` → 5, 10
- [ ] B5 risk inference: `risk/metrics.py`, `simulation/*` → 7, 16
- [ ] B6 backtest realism: `backtesting/engine.py`, `data/resample.py` → 12, 2
- [ ] B7 overfitting discipline: `backtesting/optimization.py`, forecasting DL → 12, 13, 14
- [ ] B8 parameter & structural: `config.py`, `ui/`, `reports/` → 1, 17, 18
- [ ] B9 leakage-proof ML: `data/snapshot.py`, `machine_learning/*` → 13, 14
- [ ] B10 drift mgmt: `data/loader.py` → 13, 14
- [ ] B11 confirmation/metrics: `machine_learning/evaluation.py` → 14, 9
- [ ] B12 data quality: `data/fundamentals.py`, `data/resample.py`, `data/loader.py` → all
- [ ] B13 interpretability: forecasting DL, `reports/modelscards.py` → 13, 17
- [ ] B14 DL guards → 13
- [ ] B15 RL guards → 16, 12
- [ ] B16 ensemble discipline → 14
- [ ] B17 research integrity → 9, 18
- [ ] B18 alt-data care → 2, 9
- [ ] B19 microstructure → 7, 11
