# Bias Mitigation

The authoritative reference for how the platform guards against the systematic biases and model risks that distort quantitative research. It extends the controls already described in the statistical, risk, factor, backtesting, and data-layer docs.

> **Design ethos:** Nothing is hardcoded. All thresholds, screen rules, health-score weights, and indicator presets are read from `config.py` / dataset files — which simultaneously constrains the surface for judgment-driven hacking. Results are always a **historical backtest / measurement, not a forecast** and carry that banner in the UI and Reports.

---

## Part A — Bias ↔ Mitigation Matrix

| # | Bias | Category | Where it can bite | Existing control | Strengthened / New control |
|---|---|---|---|---|---|
| 1 | Survivorship bias | Data | Backtests/factors on present-day constituents exclude delisted/bankrupt names | DATA_LAYER reliability note | Universe disclosure banner; factor results labeled "current constituents only" |
| 2 | Look-ahead bias | Data | Signals using data not knowable at `t` | `Close(t) → Open(t+1)` convention; factor `shift(1)` | Point-in-time guard on all fundamental/snapshot fields (§B1) |
| 3 | Data-snooping | Data | Repeated testing until "something works" | Deflated Sharpe; sensitivity heatmap; walk-forward | 3-way train/val/test; report # trials (§B7) |
| 4 | Selection bias | Data | Only liquid, large-cap names tested/screened | — | Screener universe disclosure + liquidity floor (§B6) |
| 5 | Backfill bias | Data | New entrants retroactively added to index history | — | Universe reconstitution / backfill caveat (§B1) |
| 6 | Time-period / regime bias | Data | Results regime-dependent | Walk-forward; regime conditioning | Report estimation window; regime splits (§B7) |
| 7 | Factor survivorship | Data | Failing factors dropped from study sets | — | Factor PnL robustness + survivor flag (§B7) |
| 8 | Overfitting / curve-fitting | Overfitting | Too many params → fits noise | Sensitivity heatmap | 3-way split + VIF (§B3, §B7) |
| 9 | In-sample vs OOS bias | Overfitting | Reporting fit as predictive performance | Default OOS Sharpe | Always report IS & OOS together (§B7) |
| 10 | Multiple testing | Overfitting | Best of many runs reported | Deflated Sharpe (`n_combos≥10`) | Disclose # trials; flag >50 (§B7) |
| 11 | P-hacking | Overfitting | Tuning until significant | Config-driven thresholds | Methodology lock: thresholds from config/files only (§B8) |
| 12 | Model risk | Overfitting | Wrong distribution/functional form | — | Out-of-sample validation + badge at-risk (§B7) |
| 13 | Parameter instability | Overfitting | Fixed params over shifting regime | Rolling beta/sharpe/vol | param-stability report (§B7) |
| 14 | Regime-change blindness | Overfitting | Model worked for old regime | Regime conditioning | walk-forward + regime splits (§B7) |
| 15 | Spurious regression | Statistical | Two non-stationary series → high R² | — | Stationarity/cointegration gate (§B4) |
| 16 | Non-stationarity ignored | Statistical | OLS/VAR on unit-root data | ADF/KPSS/ZA tests | Auto-difference gate (§B4) |
| 17 | Autocorrelation ignored | Statistical | Underestimated SEs (stale/asynchronous pricing) | Ljung-Box | HAC/Newey-West SEs (§B2) |
| 18 | Heteroskedasticity ignored | Statistical | Vol clustering → wrong inference | GARCH | HC1 robust SEs (§B2) |
| 19 | Endogeneity | Statistical | Regressor correlated with error | — | Caveat for event/factor regressions (§B8) |
| 20 | Simultaneity | Statistical | Price & volume jointly determined | — | Structural caveat (§B8) |
| 21 | Multicollinearity | Statistical | Correlated regressors inflate variance | — | VIF ≥ 10 drop in features/factors (§B3) |
| 22 | Errors-in-variables | Statistical | Proxy index → beta attenuation | — | Robust/demeaned beta; report proxy (§B2) |
| 23 | Small-sample bias | Statistical | CLT/MLE on short series | Min-obs floors; confidence badges | Badge floor (MODEL_CONFIDENCE) |
| 24 | Ecological bias | Statistical | Inferring asset from index aggregates | — | Analysis-level note (§B8) |
| 25 | Normality / Gaussian bias | Distributional | Fat tails mistaken as normal | JB test; student-`t` GARCH; parametric VaR | Report all three VaR types (§B5) |
| 26 | Fat-tail underestimation | Distributional | Understates extreme EV | Tail Ratio | Stressed-scenario table (§B5) |
| 27 | Vol clustering ignored | Distributional | Treats vols as i.i.d. | GARCH/EWMA | EWMA/GARCH always available (§B5) |
| 28 | Correlation instability | Distributional | Corr → 1 in crises | — | Kendall τ + conditional tail correlation (§B5) |
| 29 | VaR-model bias | Distributional | VaR ignores tail severity, not subadditive | CVaR | Anchor on CVaR/ES; show both (§B5) |
| 30 | Procyclicality | Distributional | Recent-vol understates calm/overstates shock | rolling vol | EWMA caveat + stress levels (§B5) |
| 31 | Cost/slippage omission | Backtest | Profitable BT → losing live | commission=0.001, slippage=0.001 | Spread/market-impact estimate (§B6) |
| 32 | Liquidity bias | Backtest | Assume tradable at last size | — | RVOL/min-volume floor (§B6) |
| 33 | Point-in-time data | Backtest | Estimates, membership as-of-history | — | PIT guard (§B1) |
| 34 | Regime overfitting | Backtest | Fits one period | Walk-forward | compare across windows (§B6, B7) |
| 35 | Capacity constraints | Backtest | Assumes strategy scales linearly with AUM | — | capacity + market-impact decay note (§B6) |
| 36 | Short-sale constraint | Backtest | Frictionless shorting assumed | SEBI retail long-only default | US margin/borrowing caveat (§B6) |
| 37 | Anchoring | Behavioral | Over-rely on initial estimate | — | config-driven thresholds; window-default disclosure |
| 38 | Confirmation | Behavioral | Find evidence for a model | — | IS+OOS always shown (§B7) |
| 39 | Hindsight | Behavioral | Past more predictable than it was | Backtest is historical | "results historical, not forecast" banner |
| 40 | Overconfidence | Behavioral | Underestimating parameter uncertainty | confidence badges | param-stability report (§B7) |
| 41 | Recency bias | Behavioral | Overweight recent regime | Rolling windows | show window used (§B8) |
| 42 | Narrative fallacy | Behavioral | Story fit to noise | — | §B8 note |
| 43 | Currency/inflation bias | Structural | Nominal vs real, multi-ccy | FX conversion; cosmetic currency | real-return note (§B8) |
| 44 | Benchmark bias | Structural | Benchmark flatters strategy | risk-free/bench param | expose + multi-benchmark (§B6) |
| 45 | Fee/expense omission | Structural | Gross vs net | commission/slippage | net-of-cost default (§B6) |
| 46 | Rebalancing bias | Structural | Costless, instantaneous | — | rebalance-cost & frequency disclosure (§B6) |
| 47 | Look-back window bias | Structural | 1Y vs 5Y materially different | Rolling windows | window sensitivity report (§B8) |
| 48 | Base-rate neglect | Structural | Rare-event EV underweighted | — | rare-event EV note (§B5, B8) |

---

## Part A.2 — Extended bias matrix (ML · DL · RL · Research · Alt-data · Microstructure)

Rows reference the Part B controls below. `dup` marks a bias already covered elsewhere.

### §8 Data leakage (model)
| Bias | Where it can bite | Control |
|---|---|---|
| Target / label leakage | A feature derived (directly or indirectly) from the target, so the model "cheats" in train but fails live | B9 (feature governance) |
| Temporal k-fold shuffle | Random k-fold lets the model train on "future" to predict "past" | B9: walk-forward + **purged/embargoed** |
| Train–test contamination | Scaling/imput/feature-selection across the whole set leaks test stats | B9: fit transforms on **train only** |
| Preprocessing leakage | Normalizing with whole-series mean/std, not the training window | B9 |
| Duplicate/overlapping samples | Overlapping rolling windows shared across train/test inflate accuracy | B9: window de-dup / embargo |

### §9 Dataset shift / distribution
| Bias | Where it can bite | Control |
|---|---|---|
| Covariate shift | Feature distribution changes train→deployment | B10: drift detection + retrain trigger |
| Concept drift | Feature→target relationship changes (market structure evolves) | B10: ADWIN / retrain / decay |
| Non-stationary features | Raw price/volume levels instead of returns/stationary diffs | B10: enforce stationary transforms |
| Class imbalance | Crash/default labels — model predicts the majority class | B10: resample / class-weight / PR-focus |

### §10 Feature engineering & selection
| Bias | Where it can bite | Control |
|---|---|---|
| Look-ahead in features | Rolling average includes current/future bar; non-point-in-time fundamentals | B1 (PIT), B9 |
| Selection-on-full-sample | Choosing "significant" features on the whole dataset | B9: nested selection on train only |
| Multicollinearity | Many indicators = transforms of the same price | B3 (VIF) |
| Curse of dimensionality | Too many features vs independent observations | B9/B11: regularization |
| Proxy / surrogate labels | e.g., "big drop" as proxy for "crash" | B12: label provenance note |

### §11 Model validation
| Bias | Where it can bite | Control |
|---|---|---|
| Hyperparameter-scale on test | Repeatedly checking test turns it into validation | B11: seal test; tune on val only |
| Multi-model selection | Reporting only the best of many runs | B11: PSR / Deflated Sharpe |
| Random seed / init sensitivity | Unstable results across seeds/backends | B11: seed band mean±stdev |
| Cross-sectional leakage | One stock's future leaking to another via shared normalization | B11: purged / group-wise scaling |

### §12 Data quality
| Bias | Where it can bite | Control |
|---|---|---|
| Noise-to-signal mismatch | Model fits noise | B12: OOS reality check |
| Imputation bias | Naive `ffill`/mean-fill injects look-ahead | B12: train-only imputation policy |
| Outlier winsorization | Clipping crash/earnings tail points (often informative) | B12: capped, non-silent |
| Corporate-action errors | Unadjusted splits/dividends create jumps | B12: adjustment flag |
| Survivorship in labels | Training only on "surviving" companies | B12: label-universe note |

### §13 Interpretability & deployment
| Bias | Where it can bite | Control |
|---|---|---|
| Black-box overreliance | Deep/ensemble models masking regime fragility | B13: introspection / checks |
| Feature-importance instability | SHAP varies across resamples but shown causal/stable | B13: resample brackets |
| Reproducibility bias | Not robust to lib/seed/vendor changes | B13: version pinning |
| Backtest-to-live gap | Live alpha decay / paper-vs-live divergence | B13, dup B6: disclaimer + live-considerations |

### §14 Deep learning
| Bias | Where it can bite | Control |
|---|---|---|
| Look-ahead in architecture | BiRNN / non-causal conv / attention sees future | B14: causal masking |
| Vanishing / exploding gradients | LSTM/RNN long sequences diverge or drop signal | B14: gating/clip/stable init |
| BatchNorm across time | Batch statistics leak / shift across time | B14: causal batch construction |
| Overparameterization | Params >> effective independent obs | B14: regularization / dropout / early-stop |
| Seed / backend non-determinism | Materially different results run-to-run | B11/B14: seed band |
| Positional leakage | Unmasked self-attention attends future tokens | B14: causal attention mask |
| Autoencoder/mode collapse | Latents reconstruct noise / single dominant mode | B14: reconstruction breakdown-aware |
| Synthetic/generated data | fake paths miss tails/vol clustering/regimes | B14: stress-check vs real |
| Transfer-mismatch | Pretrain one market, fine-tune another | B14: DGP-comparability gate |

### §15 Reinforcement learning
| Bias | Where it can bite | Control |
|---|---|---|
| Non-stationary MDP | Markets adversarial / non-stationary | B15 |
| Reward hacking | raw PnL objective → degenerate leverage/tail | B15: regularized objective |
| Off-policy evaluation | Evaluating on a different-policy history | B15: importance weighting / guardrails |
| Sim-to-real gap | Simulator misses impact/latency | B15: conservative sim + guardrail |
| Live exploration loss | Real-money losses during exploration | B15: safe simulator only |

### §16 Ensemble & meta-modeling
| Bias | Where it can bite | Control |
|---|---|---|
| Diversity illusion | Correlated models on the same leaky features | B16: diversity check |
| Stacking leakage | OOF folds leak into the meta-learner | B16: true out-of-fold discipline |
| Averaging masks bias | Bagging hides a systematically biased learner | B16: base-learner bias audit |

### §17 Research-process
| Bias | Where it can bite | Control |
|---|---|---|
| Publication bias | Only profitable strategies shared; failures vanish | B17: full-study reporting |
| Factor zoo / cross-paper | False discovery across the whole field | B17: cross-function multiple-testing |
| Replication crisis | Factors fail OOS in other markets | B17: OOS validation |
| HARKing | Presenting data-driven finding as a priori hypothesis | B17: pre-registration / config |
| Citation / authority bias | Trusting "standard" without work in own data | B17: re-validate assumptions |

### §18 Alternative-data
| Bias | Where it can bite | Control |
|---|---|---|
| Vendor survivorship / backfill | Alt-data history is non-random | B18: survivor/backfill note |
| Sentiment / NLP label | Generic lexicon mislabels financial language | B18: domain lexicon |
| Coverage bias | Skews toward large / retail-heavy names | B18: coverage note |
| PIT of snapshots | Fundamental snapshot is as-of-download, not historical | B9 / §B1 |

### §19 Market microstructure (high-frequency)
| Bias | Where it can bite | Control |
|---|---|---|
| Bid-ask bounce | Trade-to-trade "returns" oscillate around the mid | B19: mid-quote filter |
| Microstructure noise | Realized-volatility estimators biased upward | B19: two-scale RV |
| Asynchronous / stale price | Correlation bias trending down (Epps effect) | B19: sync / lag / refresh-time |

---

## Part B — Concrete Control Specifications

### B1 — Point-in-Time Guard (look-ahead, backfill, point-in-time)
- Fundamental and snapshot fields are *as-of-download*, **not** a point-in-time database. They record current values, not historical membership as it then existed.
- **Rule:** Such fields are **forbidden as inputs to backtest or factor signal construction** — they must not act on a date other than the value date. They may feed the **live screener** and a **valuation overlay (value-at-date)**.
- Implemented via a data provenance flag per field (`{"source": "yfinance"|"snapshot", "portal": "live"|"historical_ok"}`). A backtest raising a per-ticker historical demand for a non-`historical_ok` field must reject it with a labeled message (see `data/fundamentals.py`).
- Universes are built from **today's constituents**; any index/factor study using them is labeled "current-constituents only, backfill/survivorship applies."

### B2 — Robust Standard Errors
- Factor IC headline uses **ICIR-implied t** with robust SE. Compute `t = ICIR · sqrt(n_periods)` under **Newey-West (HAC)** with `lags = floor(4·(n/100)^(2/9))`.
- Regression/β/α inference uses heteroskedasticity-robust covariance (HC1) by default.
- Purpose: autocorrelated, heteroskedastic returns → correct SEs, no spurious significance (#17, #18).

### B3 — Multicollinearity (VIF)
- Before factor/ML fitting, run Variance Inflation Factor on the feature/cross-sectional set.
- Drop features with `VIF ≥ 10`; flag `VIF ≥ 5`.
- Prune overlapping momentum/volatility factor specifications (e.g., don't stack ROC(5) & ROC(20)).
- Synergy: also limits # parameters so reduces overfitting (#21, #8).

### B4 — Spurious-Regression Gate
- Level/OLS regressions require **stationarity** (ADF, p<0.05) of both series; else **difference** first.
- Level relationships require **cointegration** (Engle-Granger / Johansen) rather than simple `R²`.
- Do not report `R²` alone for trending series (spurious regression) — report the gate outcome.

### B5 — Economic / Risk Inference Rules
- **VaR:** always report **Historical VaR, Parametric VaR, and CVaR/Expected Shortfall** together. Present CVaR as the primary tail measure (VaR ignores tail severity and is not subadditive). Emphasize ES for heavy left-hand tails.
- **Procyclicality:** recent-volatile models (EWMA, rolling) understate risk in calm and overstate after a shock. Show both a **current-window** measure and a **stressed** (e.g., long-horizon or worst-regime) measure.
- **Tail correlation:** display Kendall τ and a conditional tail (crisis) correlation table in addition to Pearson; correlations spike toward 1 in crises.
- **Rare events:** Monte Carlo shows simulation from GBM central-case and **bootstrap** — the distributional tails are not robust to rare extreme EV.

### B6 — Backtest Realism
- **Liquidity floor:** strategies and Factor backtests filter/min-size on minimum volume (and optionally RVOL); the platform should not assume you can trade illiquid names at last quoted in size.
- **Capacity:** assumptions do not scale linearly with AUM — note market-impact decay for large notional (future capacity model).
- **Cost/impact:** commission + slippage are parameters; a spread/market-impact estimate should be added to net the backtest. **Net-of-cost is the default** for metrics; gross is labeled "gross".
- **Rebalancing:** report the assumed rebalance frequency and per-rebalance cost; do not assume costless/instantaneous.
- **Benchmark:** expose the chosen benchmark (config list: Nifty 50, Sensex, S&P 500, NASDAQ) and offer comparisons across multiple to avoid benchmark that flatters the strategy.
- **Short-sale:** India retail long-only default (SEBI); global runs with a US margin/borrow-cost caveat.

### B7 — Overfitting Discipline
- Require a **3-way chronological split** (train / validation / test) for any tuned model (see `FORECASTING_ML.md`).
- **Report IS and OOS metrics together** — never present in-sample as predictive.
- **Disclose the number of trials/parameter combinations**; if `> 50`, show the deflated Sharpe and flag high multiple-testing risk.
- **Walk-forward** is the default OOS regime; show per-window parameter stability.

### B8 — Judgment & Structural Notes
- All tunable thresholds/weights live in `config.py` / dataset files (not code literals) — this both fulfils the "nothing hardcoded" constraint and narrows deliberate p-hacking.
- Where rolling windows are used, **state the window** (recency is visible), and where practical show a parameter-sensitivity report (1Y vs 5Y).
- For currency: portfolio PnL converts to BASE_CURRENCY via FX; compare **real** (inflation-adjusted) returns for cross-decade work.
- Analysis caveats for endogeneity/simultaneity/ecological inference are surfaced as notes beside event-study and index-level outputs.
- Every result screen embeds the disclaimer: **"This is a historical measurement, not a forecast or a recommendation."**

### B9 — Leakage-Proof ML Pipeline
- **Split by time, never by row.** Walk-forward with **purged + embargoed** folds (drop a gap after train boundaries so overlapping/lagged features cannot leak).
- **Fit-on-train only:** min/max scaling, imputation parameters, PCA/winsor thresholds and feature selection are all fitted/derived on the training window and *applied* (not refit) to validation/test.
- Feature provenance flag blocks PIT-forbidden fundamentals from signal construction (see B1).
- Duplicate / heavy-overlap window removal so train and test share no engineered matrix.

### B10 — Dataset-Shift / Drift Management
- Compute a **drift metric** (PSI/KL or statistical test) between training feature stats and the most recent window.
- **Retrain trigger** when drift exceeds `config.py` threshold; surface an "out-of-sample regime change" flag on the forecast/confidence screen.
- Enforce stationary inputs (returns/diffs) except where a level model is explicitly chosen and documented.

### B11 — Validation & Metric Discipline
- **3-way chronological split**: train / validation / test. Hyperparameters and early-stopping tune only on validation; **test is sealed** until final evaluation.
- **Seed band:** run the fitted estimator across `N >= 5` seeds / initialisations (and CPU/GPU), report `mean +/- st` on OOS metrics, not a single run.
- Report **PSR / Deflated Sharpe** when many seeds/models are compared (multiple-testing guard).
- Any evaluation module returns PR curve / class-aware metrics for imbalanced labels, not just accuracy.

### B12 — Data-Quality Grades
- Read-through grade per sample: yfinance/snapshot loaders attach a quality flag (missing %, corporate-action-adjusted, imputation policy).
- **Imputation policy**: impute from train statistics only; never from full series; record the method.
- **Outlier handling capped & labelled**: winsor limits shown, never silently dropped, crash/earnings tails preserved or flagged.
- Corporate action: every OHLCV series carries an `adjusted` boolean (see `data/resample.py`); mixed-`adjusted` comparisons rejected.

### B13 — Interpretability & Reproducibility
- Attach a **decomposed / attention or SHAP** summary to any model figure, with **resample brackets** (mean ± CI of importance across bootstrap/seed).
- **Environment pinning**: `requirements.txt`, random seeds, and DL-determinism flags; store a `model_card` per artifact (architecture, hyperparams, data version, seed band, train window).
- Every backtest carries **implied-friction & benchmark** and re-confirms the **"historical, not forecast"** banner (B8).

### B14 — Deep-Learning Guards
- **Causal design**: causal masking for CNN / conv / self-attention so no information from a current+future bar flows into a past prediction. Only OHLCV known at forecast time.
- **Scale-stable layers**: controlled initialization, gradient clipping, and an explicit guard on batch statistics applied across time.
- **Regularization first**: dropout/early-stop/l2 before trusting an "overfit" gain. Check overparameterization against effective obs.
- **Seed / DL-ML non-determinism report** (B11 mean±st).
- Synthetic/generated data is always stress-checked against real distribution (tails, auto-correlation, vol clustering).

### B15 — RL / Policy Guards
- Train only in a **sandbox / paper simulator**; keep PnL objectives regularized so a naive optimizer cannot "hack" for reward (e.g., penalty on tail risk and drawdown, not raw PnL alone); never deploy real money on an un-explored policy.
- Report policy changes and expose a **kill-switch / guardrail**; every live-step claim is surfaced with a confidence badge (B8).

### B16 — Ensemble Discipline
- **Diversity check** for base learners (correlation / feature-disagreement) — don't average near-identical models.
- **True out-of-fold** for metalearners — OOF prediction supplied via the K-fold of base training, not leaky in-sample ones.
- Audit each base learner's bias separately (a differential underperformer is flagged, not hidden by the blend).

### B17 — Research-Process Integrity
- **Pre-register** factor/strategy intent and config (in `config.py` / dataset file) before running; note deviations.
- **Full-study reporting**: profits and failures together; M-TTS (Multiple-Testing-To-Significance) for the whole family of tested functions.
- Require an **OOS validation** (different market/horizon) before a research finding is called robust.

### B18 — Alternative-Data Care
- Log the **survivor / backfill** history of any third-party or derived series; never backtest on a field whose history only exists because the entity survived.
- NLP/sentiment uses a **financial-domain lexicon / labelling** reviewed against a human sample; bias via coverage (large-cap, retail-heavy) noted.
- PIT and snapshot month selected transparently (B1/B9).

### B19 — Market-Microstructure / High-Frequency
- Where tick/order book available: filter **microstructure noise** — use **two-scale / kernel RV**, mid-to-mid vs trade returns, to deflate Epps-type noise.
- Correct for **asynchronous / stale** prints (lag-and-lookback / refresh-time adjustment) before linking multiple instruments or FX pairs.
- At low frequency (this repo default) mark the **bid/ask vs close** caveat; never treat the derived midpoint spread as freely realizable—apply explicit friction (§B6).

---

## Part C — Ownership (control → module → page)

| Control | Module | Streamlit page |
|---|---|---|
| B1 Point-in-time guard | `data/fundamentals.py`, `data/snapshot.py` | 1 Dashboard, 2 Market Explorer, 9 Factor |
| B2 Robust SEs | `factor/scores.py`, `statistics/*`, `machine_learning/evaluation.py` | 5 Statistical, 9 Factor, 14 ML |
| B3 VIF | `machine_learning/features.py`, `factor/factors.py` | 9 Factor, 14 ML |
| B4 Spurious-regression gate | `statistics/stationarity.py`, `statarb/cointegration.py` | 5 Statistical, 10 StatArb |
| B5 Risk inference | `risk/metrics.py`, `simulation/*` | 7 Risk, 16 Monte Carlo |
| B6 Backtest realism | `backtesting/engine.py`, `data/resample.py` | 12 Backtest, 2 Explorer |
| B7 Overfitting discipline | `backtesting/optimization.py`, `forecasting/nn.py` | 12, 13, 14 |
| B8 Parameter & structural | `config.py`, `ui/`, `reports/` | 1, 17, 18 |
| B9 Leakage-proof ML | `data/snapshot.py`, `machine_learning/*`, `forecasting/nn.py` | 13 Forecast, 14 ML |
| B10 Drift mgmt | `data/loader.py`, `statistics/tests.py` | 13, 14 |
| B11 Confirmation / metrics | `machine_learning/evaluation.py`, `factor/ic.py` | 14 ML, 9 Factor |
| B12 Data quality | `data/fundamentals.py`, `data/resample.py`, `data/loader.py` | all |
| B13 Interpretability / reproducibility | `forecasting/nn.py`, `ui/`, `reports/modelscards.py` | 13, 17 |
| B14 DL guards | `forecasting/dl.py`, `machine_learning/trainer.py` | 13 Forecast |
| B15 RL guards | `reinforcement/*`, `simulation/*`, `backtesting/engine.py` | 16 Monte-Carlo, 12 |
| B16 Ensemble discipline | `machine_learning/ensemble.py`, `backtesting/optimization.py` | 14 ML |
| B17 Research integrity | `config.py`, `factor_research/*`, `reports/` | 9 Factor, 18 |
| B18 Alt-data care | `data/snapshot.py`, `factor_research/*` | 2, 9 |
| B19 Microstructure | `risk/metrics.py`, `statistics/distribution.py`, `data/tick*` | 7 Risk, 11 Microstructure |