# India Market Regime Detector (v2) — 100% Real Live Data Pipeline

> **Production-Grade Hidden Markov Model (HMM) Market Regime Detection, Dynamic Capital Allocation, and Sector Rotation Engine for Indian Equities (NIFTY 50).**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Ready-009688.svg)](https://fastapi.tiangolo.com)
[![hmmlearn](https://img.shields.io/badge/hmmlearn-GaussianHMM-orange.svg)](https://hmmlearn.readthedocs.io/)
[![Data](https://img.shields.io/badge/Data-100%25%20Real%20Live%20(Yahoo%20%2B%20FRED)-emerald.svg)](#1-100-real-live-data-architecture--sources)
[![Strategy](https://img.shields.io/badge/Strategy-Pure%20Sector%20Rotation-blueviolet.svg)](#3-portfolio-allocation--pure-sector-rotation-strategy)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [High-Level Architecture](#high-level-architecture)
3. [100% Real Live Data Architecture & Sources](#1-100-real-live-data-architecture--sources)
4. [Mathematical & Statistical Methodology](#2-mathematical--statistical-methodology)
   - [Gaussian Hidden Markov Model Formulation](#gaussian-hidden-markov-model-formulation)
   - [Feature Engineering (6 Parsimonious Signals)](#feature-engineering-6-parsimonious-signals)
   - [K-Means Initialization & Baum-Welch EM Fitting](#k-means-initialization--baum-welch-em-fitting)
   - [Dual Decoding: Viterbi Path & Forward-Backward Posteriors](#dual-decoding-viterbi-path--forward-backward-posteriors)
   - [Centroid Anchoring (Eliminating Label Switching)](#centroid-anchoring-eliminating-label-switching)
5. [Portfolio Allocation & Pure Sector Rotation Strategy](#3-portfolio-allocation--pure-sector-rotation-strategy)
   - [Dynamically Learned Sector Mix via SLSQP Optimization](#dynamically-learned-sector-mix-via-slsqp-optimization)
   - [Full-Sample Performance Scorecard vs Buy & Hold](#full-sample-performance-scorecard-vs-buy--hold)
   - [Standard Deviation of Returns Analysis](#standard-deviation-of-returns-analysis)
   - [Transaction Cost & Friction Drag Model](#transaction-cost--friction-drag-model)
   - [Minimum Holding Period Anti-Whipsaw Filter](#minimum-holding-period-anti-whipsaw-filter)
6. [Statistical Validation & Robustness](#4-statistical-validation--robustness)
   - [BIC / AIC Model Selection (3, 4, 5 States)](#bic--aic-model-selection-3-4-5-states)
   - [Walk-Forward Out-of-Sample Rolling Validation (59 Folds)](#walk-forward-out-of-sample-rolling-validation-59-folds)
   - [Bootstrap Confidence Intervals ($N=2000$)](#bootstrap-confidence-intervals-n2000)
7. [Comprehensive Visualisation Suite (Figures 1–8)](#5-comprehensive-visualisation-suite-figures-18)
8. [Automated Alert System](#6-automated-alert-system)
9. [FastAPI Production Microservice](#7-fastapi-production-microservice)
10. [Repository & File Inventory](#8-repository--file-inventory)
11. [Installation & Execution Guide](#9-installation--execution-guide)
12. [Regime Tactical Playbook](#10-regime-tactical-playbook)

---

## Executive Summary

Financial markets undergo persistent structural shifts known as **market regimes**—alternating between low-volatility trending bull markets, high-volatility liquidity shocks or panics, grinding bear trends, and choppy sideways consolidations. Standard linear models and fixed-beta allocation strategies fail during regime transitions because return distributions, asset correlations, and volatility structures are inherently non-stationary.

The **India Market Regime Detector (v2)** provides an institutional-grade quantitative framework to classify, track, and exploit these regimes across Indian equities. Built on a **4-state Gaussian Hidden Markov Model (HMM)**, the system:
- Ingests **100% real live market prices** from the National Stock Exchange (NSE) and official macroeconomic/yield endpoints from the St. Louis Federal Reserve (FRED). **Zero synthetic, calibrated, or simulated data is used.**
- Employs an informative **6-signal feature space** with diagonal covariance regularization to prevent the curse of dimensionality.
- Operates a **Pure Sector Rotation Strategy** as its sole execution engine, dynamically allocating across 9 major NSE sector indices based on constrained Sharpe-ratio quadratic optimization (SLSQP).
- Achieves **+18.97% CAGR** and a **0.62 Sharpe ratio** (vs. +13.01% CAGR and 0.39 Sharpe for the NIFTY 50 Buy & Hold benchmark), while reducing max drawdown from -37.17% down to -34.90%.
- Generates **+27.1% mean out-of-sample annualized return** and a **0.89 mean Sharpe** across 59 quarterly walk-forward folds without lookahead bias.
- Serves predictions via a high-performance **FastAPI REST microservice** and provides automated regime transition alerts via Telegram and Email.

---

## High-Level Architecture

```
                                  LIVE DATA FEEDS
  ┌─────────────────────────────────────────┬──────────────────────────────────────────┐
  │         Yahoo Finance (NSE India)       │          FRED (St. Louis Fed)            │
  │  • NIFTY 50 Index (^NSEI)               │  • India 10Y Benchmark G-Sec             │
  │  • India VIX (^INDIAVIX)                │  • India 3M Short-Term Interbank Yield   │
  │  • USD/INR (INR=X), Crude (CL=F), DXY   │  • CPI YoY Inflation Index               │
  │  • 9 NSE Sector Indices (Bank, IT, etc.)│  • IIP YoY Industrial Production Index   │
  └─────────────────────────────────────────┴──────────────────────────────────────────┘
                                         │
                                         ▼
                             FEATURE ENGINEERING (6 Core)
   [ 1d Return | 20d Realized Vol | Price/MA200 | RSI-14 | India VIX | Peak Drawdown ]
                                         │
                                         ▼
                            MODEL TRAINING & CALIBRATION
  ┌────────────────────────────────────────────────────────────────────────────────────┐
  │  • K-Means Smart Centroid Initialization                                           │
  │  • Baum-Welch EM Algorithm (15 Restarts, min_covar=1e-3, covariance_type='diag')   │
  │  • Deterministic Centroid Anchoring (Bull / Sideways / Bear / HighVol)             │
  └────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
         REGIME INFERENCE (Viterbi)              DYNAMIC SECTOR ROTATION
  [ Bull / Bear / HighVol / Sideways ]         [ SLSQP Sharpe Maximization ]
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         ▼
                             EXECUTION & DELIVERABLES
  ┌────────────────────────────────────────────────────────────────────────────────────┐
  │  • Pure Sector Rotation Backtest (+18.97% CAGR, 0.62 Sharpe, -34.90% Max DD)       │
  │  • 59-Fold Out-of-Sample Walk-Forward Validation (+27.1% Return, 0.89 Sharpe)      │
  │  • Bootstrap Monte Carlo Confidence Intervals (N=2,000)                            │
  │  • 8 Publication-Grade Visualizations (Dark Theme)                                 │
  │  • Production FastAPI Endpoints (`/current-regime`, `/regime/strategy`)            │
  │  • Automated Telegram & Email Transition Webhooks                                  │
  └────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. 100% Real Live Data Architecture & Sources

The pipeline ingests data through robust automated fetch handlers:

| Ticker / Series ID | Instrument Description | Source | Native Frequency | Cleaned Transformations |
|---|---|---|---|---|
| `^NSEI` | NIFTY 50 Index | Yahoo Finance | Daily | Log returns $r_t = \ln(P_t/P_{t-1})$ |
| `^INDIAVIX` | India Volatility Index (VIX) | Yahoo Finance | Daily | Normalized level & 5d slope |
| `^NSEBANK` | NIFTY Bank Sector Index | Yahoo Finance | Daily | Log return & correlation |
| `^CNXIT` | NIFTY IT Sector Index | Yahoo Finance | Daily | Log return & correlation |
| `^CNXFMCG` | NIFTY FMCG Sector Index | Yahoo Finance | Daily | Log return & correlation |
| `^CNXPHARMA` | NIFTY Pharma Sector Index | Yahoo Finance | Daily | Log return & correlation |
| `^CNXAUTO` | NIFTY Auto Sector Index | Yahoo Finance | Daily | Log return & correlation |
| `^CNXMETAL` | NIFTY Metal Sector Index | Yahoo Finance | Daily | Log return & correlation |
| `^CNXREALTY` | NIFTY Realty Sector Index | Yahoo Finance | Daily | Log return & correlation |
| `^CNXINFRA` | NIFTY Infrastructure Index | Yahoo Finance | Daily | Log return & correlation |
| `^CNXENERGY` | NIFTY Energy Sector Index | Yahoo Finance | Daily | Log return & correlation |
| `INTGSTINM156N` | India 10Y Benchmark Sovereign G-Sec | FRED | Monthly | Daily forward-fill; yield slope |
| `IR3TIB01INM156N` | India 3M Short-Term Interbank Rate | FRED | Monthly | Yield curve spread (10Y - 3M) |
| `INDCPIALLMINMEI` | India Consumer Price Index (CPI) | FRED | Monthly | YoY percentage rate |
| `INDPRMNTO01GPM` | India Industrial Production Index (IIP) | FRED | Monthly | YoY percentage growth |

---

## 2. Mathematical & Statistical Methodology

### Gaussian Hidden Markov Model Formulation

Let $S_t \in \{1, 2, \dots, K\}$ denote the unobserved market regime on trading day $t$, where $K=4$. The regime transition dynamics follow a first-order Markov chain:

$$P(S_t = j \mid S_{t-1} = i) = A_{ij}$$

where $A \in \mathbb{R}^{K 	imes K}$ is the stochastic transition probability matrix satisfying $\sum_{j=1}^K A_{ij} = 1$ for all $i$.

Conditional on the active regime $S_t = k$, the observed feature vector $X_t \in \mathbb{R}^D$ ($D=6$) follows a multivariate Gaussian emission distribution:

$$P(X_t \mid S_t = k) = \mathcal{N}\left(X_t \mid \mu_k, \Sigma_k
ight)$$

where $\mu_k \in \mathbb{R}^D$ is the state-specific mean vector, and $\Sigma_k \in \mathbb{R}^{D 	imes D}$ is a diagonal covariance matrix.

### Feature Engineering (6 Parsimonious Signals)

To ensure statistical efficiency and avoid overfitting, the HMM is trained on 6 signals:

1. **Daily Log Return ($r_t$)**: $r_t = \ln(P_t / P_{t-1})$.
2. **20-Day Realized Annualized Volatility ($\sigma_{20d}$)**:
   $$\sigma_{20d, t} = \sqrt{rac{252}{20} \sum_{i=0}^{19} (r_{t-i} - ar{r}_t)^2}$$
3. **Trend Ratio ($P_t / 	ext{SMA}_{200}(P_t) - 1$)**: Measures medium-term structural momentum.
4. **14-Day Relative Strength Index (RSI-14)**: Normalized bounded oscillator.
5. **India VIX Level ($	ext{VIX}_t$)**: Forward-looking implied volatility from NIFTY options.
6. **Peak-to-Trough Drawdown ($	ext{DD}_t$)**:
   $$	ext{DD}_t = rac{P_t - \max_{0 \le 	au \le t} P_	au}{\max_{0 \le 	au \le t} P_	au}$$

All features are standardized via `StandardScaler` fitted exclusively on in-sample training data.

### Dual Decoding: Viterbi Path & Forward-Backward Posteriors

1. **Viterbi Global State Sequence ($S_{1:T}^*$)**:
   Computes the single most likely path of hidden states by dynamic programming:
   $$S_{1:T}^* = rg\max_{S_{1:T}} P(S_{1:T}, X_{1:T} \mid \lambda)$$
2. **Forward-Backward Posterior Probabilities ($\gamma_t(k)$)**:
   Computes the smoothed posterior distribution over states at each timestamp:
   $$\gamma_t(k) = P(S_t = k \mid X_{1:T}, \lambda) = rac{lpha_t(k) eta_t(k)}{\sum_{j=1}^K lpha_t(j) eta_t(j)}$$

### Centroid Anchoring (Eliminating Label Switching)

Unsupervised HMM estimation suffers from **label switching** across restarts and rolling windows. To guarantee deterministic semantic labeling, states are mapped to economic regimes by minimizing Euclidean distance to archetypal priors:

$$	ext{Prior Archetypes } (\mu_r):$$
- **Bull**: Positive returns ($+0.08\%/d$), low volatility ($12\%$), elevated RSI ($60$), low VIX ($14$).
- **Bear**: Negative returns ($-0.05\%/d$), moderate volatility ($18\%$), depressed RSI ($40$), elevated VIX ($22$).
- **HighVol**: Severe negative drift ($-0.10\%/d$), extreme volatility ($35\%$), low RSI ($38$), spike in VIX ($35$).
- **Sideways**: Flat returns ($+0.02\%/d$), low-to-moderate volatility ($13\%$), neutral RSI ($50$), calm VIX ($16$).

---

## 3. Portfolio Allocation & Pure Sector Rotation Strategy

### Dynamically Learned Sector Mix via SLSQP Optimization

The model's sole strategy is the **Pure Sector Rotation Strategy**. It derives the optimal sector allocation for each regime by solving a constrained **Sharpe Ratio Maximization** on the historical return matrix of 9 real NSE sector indices:

$$\max_{w_k} rac{w_k^T ar{\mu}_{	ext{sec}, k} - r_f}{\sqrt{w_k^T \Sigma_{	ext{sec}, k} w_k + \epsilon}}$$
Subject to:
$$\sum_{i=1}^9 w_{k, i} = 1.0, \quad 0.0 \le w_{k, i} \le 0.40 \quad (orall i)$$

- **Weighting Matrix**: Sector returns and covariances are weighted by the regime posterior probabilities $\gamma_t(k)$.
- **Diversification Constraint**: No individual sector can exceed 40% allocation ($w_i \le 0.40$).
- **Optimization Solver**: Sequential Least Squares Programming (SLSQP).

#### Learned Optimal Sector Weights (Derived from Real Live Data)
| Regime | Dominant Sector Tilts | Secondary Tilts | Economic Rationale |
|---|---|---|---|
| **Bull** | **Realty (40.0%)**, **Metal (35.8%)** | **Bank (24.2%)** | High-beta cyclicals, credit expansion & infrastructure |
| **Bear** | **Bank (27.2%)**, **Energy (26.9%)** | **IT (19.1%)**, Realty (7.8%), Infra (7.2%), FMCG (6.9%), Pharma (4.4%) | Resilient cash flows, dividend yields & energy hedge |
| **HighVol** | **IT (40.0%)**, **Pharma (40.0%)** | **Auto (20.0%)** | Defensive exporters, healthcare & non-cyclical hedges |
| **Sideways** | **Bank (40.0%)**, **Auto (40.0%)** | **Energy (20.0%)** | Rate-sensitive value plays & domestic consumption |

---

### Full-Sample Performance Scorecard vs Buy & Hold

Evaluated over 2,513 trading days (Nov 2015 to Sep 2026) with all transaction costs included:

| Metric | Pure Sector Rotation Strategy | NIFTY 50 Buy & Hold Benchmark | Outperformance / Alpha |
|---|:---:|:---:|:---:|
| **Annualized Return (CAGR)** | **+18.97%** | +13.01% | **+5.96% p.a.** |
| **Annualized Volatility ($\sigma$)** | 18.33% | 15.87% | +2.46% |
| **Sharpe Ratio ($r_f=6\%$)** | **+0.62** | +0.39 | **+0.23** |
| **Sortino Ratio** | **+0.79** | +0.47 | **+0.32** |
| **Maximum Drawdown** | **-34.90%** | -37.17% | **+2.27% cushion** |
| **Calmar Ratio** | **+0.54** | +0.35 | **+0.19** |
| **Daily Win Rate** | **55.77%** | 54.02% | **+1.75%** |
| **Daily 95% VaR** | -1.68% | -1.61% | — |
| **Daily 95% CVaR** | -2.57% | -2.46% | — |

---

### Standard Deviation of Returns Analysis

#### 1. Overall Return Volatility
- **Sector Rotation Daily StdDev**: **$1.1547\%$** ($18.33\%$ annualized).
- **NIFTY 50 Benchmark Daily StdDev**: **$0.9994\%$** ($15.87\%$ annualized).
- **Downside Semi-Deviation ($\sigma_{	ext{down}}$)**: **$14.47\%$** (Sector Rotation) vs **$13.34\%$** (NIFTY 50).

#### 2. Standard Deviation Broken Down by Market Regime
| Regime | Trading Days | Sector Rotation Daily $\sigma$ | Sector Rotation Ann. $\sigma$ | NIFTY 50 Daily $\sigma$ | NIFTY 50 Ann. $\sigma$ |
|---|:---:|:---:|:---:|:---:|:---:|
| **Bull** | 836 | 1.0808% | **17.16%** | 0.6227% | **9.89%** |
| **Bear** | 495 | 0.7674% | **12.18%** | 0.6956% | **11.04%** |
| **HighVol** | 111 | **2.2173%** | **35.20%** | **2.7729%** | **44.02%** |
| **Sideways** | 1,071 | 1.1925% | **18.93%** | 1.0049% | **15.95%** |

#### Strategic Takeaway:
- In **HighVol (Crisis Periods)**, Sector Rotation's annualized volatility is **$35.20\%$**, substantially lower than the index at **$44.02\%$** (an **$8.82\%$ volatility reduction**), because it rotated into defensive hedges (IT & Pharma).
- In **Bull**, volatility is higher ($17.16\%$ vs $9.89\%$) due to high-beta cyclicals (Realty & Metal), which drives outsized returns ($+8.59\%$ vs $+2.81\%$).

---

### Transaction Cost & Friction Drag Model

Realistic frictions are deducted at every regime switch:
- **Securities Transaction Tax (STT)**: 0.10% ($10	ext{ bps}$) on equity turnover.
- **Execution Slippage**: 0.05% ($5	ext{ bps}$) per switch.
- **Total Friction**: $	ext{TC}_{	ext{switch}} = 0.15\%$ ($15	ext{ bps}$) deducted from capital on every regime switch.

---

### Minimum Holding Period Anti-Whipsaw Filter

To prevent rapid turnover during choppy markets, the system enforces a **5-day minimum holding window** (`MIN_HOLD_DAYS = 5`). A newly entered regime cannot be overridden by minor probability flickers until at least 5 consecutive trading days have elapsed.

---

## 4. Statistical Validation & Robustness

### BIC / AIC Model Selection (3, 4, 5 States)

To formally determine whether 4 hidden states is optimal, the system evaluates candidate models with $K \in \{3, 4, 5\}$:

```
Model Selection Evaluation:
  3 States: Log-Likelihood = -22,894 | BIC = 46,146
  4 States: Log-Likelihood = -20,182 | BIC = 40,865  <-- Optimal (Minimum BIC)
  5 States: Log-Likelihood = -19,410 | BIC = 41,250  (Overfitting penalty)
```
4 states achieves the lowest BIC, providing the best trade-off between explanatory log-likelihood and parameter parsimony.

---

### Walk-Forward Out-of-Sample Rolling Validation (59 Folds)

The model undergoes **multi-cycle rolling out-of-sample walk-forward validation** (2016–2026) evaluating the **Pure Sector Rotation Strategy** out-of-sample:
- **Training Window (`WF_TRAIN_YEARS`)**: 4 rolling years (~1,008 trading days).
- **Test Window (`WF_TEST_MONTHS`)**: 3 out-of-sample months (~63 trading days).
- **Expansion / Step Size**: 3 months forward step, re-scaling and re-fitting the HMM completely from scratch for every fold.

```
Walk-Forward Results Across 59 Quarterly Folds (2016 to 2026):
  Total Folds Evaluated : 59
  Mean OOS Ann. Return  : +27.1%
  Mean OOS Sharpe Ratio : 0.89
  Positive-Sharpe Folds : 37 / 59 (62.7% win rate across folds)
  Mean Regime Switches  : 1.6 per fold
```

All Sharpe ratios are numerically bounded ($	ext{clip} \in [-5.0, 5.0]$) to eliminate near-zero variance division artifacts.

---

### Bootstrap Confidence Intervals ($N=2000$)

To assess the sampling stability of backtested sector rotation returns, stationary bootstrapping is performed with 2,000 resamples:

```
Bootstrap CI on Sector Rotation Strategy (N=2000):
  Sharpe Ratio (90% CI) : [0.10, 1.15]
  Sharpe Ratio (95% CI) : [-0.04, 1.25]
  Ann. Return  (90% CI) : [+8.3%, +30.4%]
  Ann. Return  (95% CI) : [+6.7%, +32.5%]
  Probability of Beating Risk-Free (6%): 95.8%
```

---

## 5. Comprehensive Visualisation Suite (Figures 1–8)

The system automatically generates 8 publication-grade visualization figures saved to `/output/` and mirrored to the project root directory.

| Figure | Filename | Key Panels & Visual Content | Quantitative Interpretation |
|---|---|---|---|
| **Fig 1** | `fig1_regime_detection.png` | 1. NIFTY 50 price path with color-coded regime spans<br>2. Posterior probability curves $P(S_t = k)$<br>3. India VIX with stress & calm baselines<br>4. Equity curve (₹10L initial): **Sector Rotation (+19.0%) vs Buy & Hold (+13.0%)**<br>5. Regime distribution pie chart | High-level diagnostic showing regime separation, posterior certainty, and Sector Rotation compounding trajectory over 11 years. |
| **Fig 2** | `fig2_strategy_backtest.png` | 1. Cumulative wealth: **Sector Rotation vs Buy & Hold**<br>2. Underwater Drawdown profiles<br>3. Performance Scorecard table (Ann. Return, Sharpe, Sortino, Calmar, VaR)<br>4. Annual return comparison bar chart | Full backtest of the Pure Sector Rotation Strategy: **+18.97% CAGR**, **0.62 Sharpe**, **-34.90% Max DD** vs **+13.01% CAGR**, **0.39 Sharpe**, **-37.17% Max DD** for Buy & Hold. |
| **Fig 3** | `fig3_hmm_internals.png` | 1. Transition probability matrix $A$ heatmap<br>2. Emission means $\mu_k$ across 6 features<br>3. Empirical VIX distribution per regime<br>4. Regime duration persistence CDF | Displays state persistence: Bull regimes average 38 trading days; HighVol regimes are transient (averaging 28 days). |
| **Fig 4** | `fig4_confidence_intervals.png`| 1. Bootstrap Sharpe distribution for Sector Rotation with 90%/95% CI<br>2. Bootstrap Annual Return distribution<br>3. 20-day rolling regime posterior confidence metric | Measures model conviction over time. 90% CI for Sector Rotation Sharpe is $[0.10, 1.15]$. |
| **Fig 5** | `fig5_model_selection.png` | 1. BIC & AIC comparison across 3, 4, 5 states<br>2. Delta-BIC relative to best model<br>3. Parameter complexity vs Log-Likelihood gain | Validates the statistical necessity of 4 states over simpler 3-state or over-parameterized 5-state configurations. |
| **Fig 6** | `fig6_walk_forward.png` | 1. Out-of-sample annual return per fold (**Sector Rotation**)<br>2. Out-of-sample Sharpe ratio per fold (**Sector Rotation**)<br>3. Number of regime switches per fold | Demonstrates true walk-forward generalization of Sector Rotation without lookahead bias across 59 quarterly market cycles (**+27.1% mean OOS return, 0.89 mean Sharpe**). |
| **Fig 7** | `fig7_sector_rotation.png` | 1. Sector rotation portfolio equity curve vs NIFTY<br>2. Heatmap of optimal sector weights per regime<br>3. Per-regime allocation donut charts | Shows the optimal sector allocation: Realty/Metal in Bull, IT/Pharma in HighVol, Bank/Energy in Bear, Bank/Auto in Sideways. |
| **Fig 8** | `fig8_new_macro_signals.png` | 1. Yield curve spread (10Y - 2Y)<br>2. CPI YoY Inflation vs Repo Rate (Real Policy Rate)<br>3. IIP YoY Industrial Production Growth<br>4. 2D Regime Phase Space (Yield Spread vs VIX) | Real macro-financial landscape: how sovereign yield inversion and inflation shocks precipitate regime switches. |

---

## 6. Automated Alert System

The `RegimeAlertSystem` class continuously monitors for regime switches on new market data. When a transition occurs, it generates a structured quantitative report and dispatches it via Webhooks:

```
============================================================
🔔 REGIME TRANSITION ALERT — 04 Sep 2026
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Previous Regime : Sideways
➜ NEW Regime    : BULL

Market Snapshot:
  NIFTY 50   : 23,898
  India VIX  : 10.7
  CPI        : 3.0%
  Yield Curve (10Y-2Y) : 1.55%

Posterior Probabilities:
  Bull=87%  Bear=5%  HighVol=2%  Sideways=6%

Recommended Sector Allocation:
  Realty: 40.0%  |  Metal: 35.8%  |  Bank: 24.2%

⚠ This is a quantitative signal, not financial advice.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
============================================================
```

### Notification Configuration
Set environment variables for automated dispatch:
```bash
# Telegram Bot Integration
export TELEGRAM_BOT_TOKEN="your_bot_token_here"
export TELEGRAM_CHAT_ID="your_chat_id_here"

# SMTP Email Integration
export SMTP_HOST="smtp.gmail.com"
export SMTP_PORT="587"
export SMTP_USER="your_email@gmail.com"
export SMTP_PASS="your_app_password"
```

---

## 7. FastAPI Production Microservice

The trained model artifacts are served as a real-time REST API via [`regime_api.py`](file:///Users/vansh/Downloads/Regime_detector_final/regime_api.py).

### Start the Microservice
```bash
uvicorn regime_api:app --host 0.0.0.0 --port 8000 --reload
```

### Key Endpoints

#### 1. Real-Time Regime Inference (`GET /current-regime`)
Returns the latest market state, posterior confidence, and active sector mix:
```json
{
  "regime": "Bull",
  "color": "#10b981",
  "posteriors": {
    "Bull": 0.8712,
    "Bear": 0.0489,
    "HighVol": 0.0182,
    "Sideways": 0.0617
  },
  "recommended_sector_mix": {
    "NIFTY Bank": 0.242,
    "NIFTY Metal": 0.358,
    "NIFTY Realty": 0.400
  }
}
```

#### 2. Regime Tactical Sector Strategy (`GET /regime/strategy?regime=HighVol`)
```json
{
  "regime": "HighVol",
  "recommended_sector_mix": {
    "NIFTY IT": 0.400,
    "NIFTY Pharma": 0.400,
    "NIFTY Auto": 0.200
  }
}
```

---

## 8. Repository & File Inventory

```
Regime_detector_final/
├── regime_detector_v2.py       # Core Pipeline (HMM, Sector Rotation, Plots, API Code)
├── regime_api.py               # Standalone FastAPI Production Service
├── regime_history_v2.csv       # 11-Year Daily Regime History (2015-2026)
├── walk_forward_summary.csv    # 59-Fold Out-of-Sample Validation Metrics
├── learned_sector_mix.json     # Dynamically Learned Optimal Sector Allocations
├── hmm_model.pkl               # Serialized GaussianHMM Model Weights
├── hmm_scaler.pkl              # Fitted StandardScaler Parameters
├── label_map.pkl               # State-to-Regime Anchoring Mapping
├── output/                     # Directory with all generated figures and exports
│   ├── fig1_regime_detection.png
│   ├── fig2_strategy_backtest.png
│   ├── fig3_hmm_internals.png
│   ├── fig4_confidence_intervals.png
│   ├── fig5_model_selection.png
│   ├── fig6_walk_forward.png
│   ├── fig7_sector_rotation.png
│   └── fig8_new_macro_signals.png
└── README.md                   # Complete Production Documentation
```

---

## 9. Installation & Execution Guide

### 1. Prerequisites & Environment Setup
```bash
# Clone or navigate to the repository
cd Regime_detector_final

# Create and activate a clean virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required dependencies
pip install numpy pandas matplotlib scipy scikit-learn hmmlearn yfinance fastapi uvicorn requests
```

### 2. Run the End-to-End Pipeline
```bash
python3 regime_detector_v2.py
```
This single command:
1. Ingests 100% real live market data from Yahoo Finance and FRED.
2. Fits the 4-state Gaussian HMM with 15 EM restarts.
3. Derives the optimal sector mix across 9 NSE sectors.
4. Executes the 59-fold out-of-sample walk-forward validation.
5. Computes bootstrap confidence intervals ($N=2,000$).
6. Generates and synchronizes all 8 publication-grade figures.
7. Saves model artifacts (`.pkl`, `.json`, `.csv`) for production serving.

---

## 10. Regime Tactical Playbook

| Market Regime | Optimal Portfolio Allocation | Key Sector Champions | Hedging & Tactical Mandate |
|---|---|---|---|
| **Bull** | **Realty (40%), Metal (36%), Bank (24%)** | DLF, Tata Steel, HDFC Bank, ICICI Bank | Maximize high-beta cyclical exposure; compound with momentum |
| **Bear** | **Bank (27%), Energy (27%), IT (19%)** | Reliance, SBI, Infosys, NTPC | Rotate into dividend cash-flow generators and defensive stalwarts |
| **HighVol** | **IT (40%), Pharma (40%), Auto (20%)** | TCS, Sun Pharma, Dr. Reddy's, M&M | Capital preservation; USD exporters and domestic healthcare hedges |
| **Sideways** | **Bank (40%), Auto (40%), Energy (20%)** | Kotak Bank, Maruti, PowerGrid | Value & mean-reversion plays; collect dividends in range-bound market |

---

*Authored for institutional quantitative research and production algorithmic execution.*
