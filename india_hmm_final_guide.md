# India HMM Regime Detector — Operations Guide & Project Summary

**Version:** v2 | **Last Updated:** 21 August 2026 | **Author:** Vansh Mohla

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [What Was Built — Conversation Summary](#2-what-was-built--conversation-summary)
3. [File Inventory](#3-file-inventory)
4. [How to Run the Model](#4-how-to-run-the-model)
5. [How to Switch to Live Data](#5-how-to-switch-to-live-data)
6. [How to Use the FastAPI Service](#6-how-to-use-the-fastapi-service)
7. [How the Alert System Works](#7-how-the-alert-system-works)
8. [Understanding the Outputs](#8-understanding-the-outputs)
9. [Model Results at a Glance](#9-model-results-at-a-glance)
10. [Regime Playbook — What to Do in Each Regime](#10-regime-playbook--what-to-do-in-each-regime)
11. [Ongoing Maintenance](#11-ongoing-maintenance)
12. [Next Steps Still Remaining](#12-next-steps-still-remaining)

---

## 1. Project Overview

This project is a **4-state Gaussian Hidden Markov Model (HMM)** that detects the current macro regime of the Indian equity market (NIFTY 50) and generates a corresponding trading strategy. The model ingests 16 market and macro features, runs the Baum-Welch EM algorithm to fit regime parameters, decodes the most likely regime path via Viterbi, and then applies a regime-specific portfolio strategy.

The four regimes the model identifies are:

| Regime | Character | VIX Range | Daily Return (μ) |
|--------|-----------|-----------|-----------------|
| **Bull** | Trending up, low fear | 10–18 | +0.086%/day |
| **Sideways** | Range-bound, moderate vol | 13–20 | +0.046%/day |
| **Bear** | Falling, elevated fear | 25–35 | −0.244%/day |
| **HighVol** | Crisis / crash | 40–80+ | +0.112%/day* |

*HighVol positive mean is a recovery-bounce artefact within crisis windows (e.g. COVID March → April 2020).

---

## 2. What Was Built — Conversation Summary

### Session 1 — Interactive Dashboard (React)

The first deliverable was a browser-based interactive dashboard built as a React artifact. It featured:

- Live signal cards for NIFTY, India VIX, RBI Repo Rate, 10Y G-Sec, USD/INR, FII flows, Brent Crude, and PMI
- Market selector (NIFTY 50 / SENSEX / NIFTY Midcap 150) and lookback selector (1/2/3 years)
- Four output tabs after running: Strategy, HMM Model, Backtest, Confidence
- Doughnut chart for portfolio allocation, Viterbi path visualisation, transition matrix heatmap, and Monte Carlo CI chart

**Caveat acknowledged at the time:** All data in the dashboard was synthetic and hardcoded. The "Baum-Welch" loading animation was illustrative — no real EM ran in the browser. The dashboard served as a design and UX prototype.

### Session 2 — Real Python HMM Model (v1: `regime_detector.py`)

A production-grade Python script using `hmmlearn`. Eight pipeline stages:

1. **Data generation** — calibrated synthetic NIFTY 50 data (2015–2024), 2,609 trading days
2. **Feature engineering** — 26 features (returns, rolling vol, RSI, VIX ratio, yield spread, FII flows, drawdown, FX trend)
3. **Feature selection** — 12-feature parsimonious set for the HMM (avoids curse of dimensionality)
4. **HMM training** — `GaussianHMM`, full covariance, 200 EM iterations, 15 random restarts
5. **Regime decoding** — Viterbi (most likely path) + Forward-Backward (posteriors)
6. **Strategy backtest** — regime-specific return multipliers, capital curve vs buy-and-hold
7. **Bootstrap CI** — 2,000 resamples, 90% and 95% intervals on Sharpe, Return, Vol, Calmar
8. **Visualisation** — 4 figures saved as PNG

v1 results: Ann. Return 9.6% | Sharpe 0.25 | Max DD −24.5% | Calmar 0.39

### Session 3 — v2: All Next-Steps Implemented (`regime_detector_v2.py`)

Every item from the v1 to-do list was implemented:

**New macro signals added:**
- CPI inflation (%) — with RBI 4% target band
- IIP Index of Industrial Production (YoY %) — economic momentum
- 2Y G-Sec yield → 10Y−2Y yield curve — inversion = recession warning
- DXY proxy → USD strength impact on EM/INR flows
- Real repo rate (Repo − CPI) — policy stance indicator

**Model selection (BIC/AIC):**
- Compared 3-state, 4-state, and 5-state Gaussian HMMs
- BIC and AIC both favoured the 5-state model on this dataset
- Pipeline retains 4-state for strategy consistency and interpretability

**Walk-forward validation (no look-ahead bias):**
- 21 rolling folds: 4-year training window → 3-month OOS prediction
- Re-fits the full HMM from scratch each fold
- Scales features using only training-period statistics

**Transaction costs:**
- STT 0.10% + slippage 0.05% = 0.15% total cost applied on each regime switch
- 40 switches over 10 years → 6.0% cumulative TC drag

**Sector rotation layer:**
- Per-regime NSE sector weights across 9 sectors (Bank, IT, Auto, Metal, Realty, Infra, Energy, FMCG, Pharma)
- Sector betas estimated per regime to approximate sector-level returns

**FastAPI REST service:**
- `regime_api.py` — 5 endpoints, ready to serve the React dashboard with real predictions

**Alert system:**
- `RegimeAlertSystem` class with email (SMTP) and Telegram bot support
- Fires on every regime transition with full market snapshot and strategy recommendation

v2 results (TC-adjusted): Ann. Return 22.5% | Sharpe 1.03 | Max DD −22.5% | Calmar 1.00

---

## 3. File Inventory

```
regime_detector_v2.py        Main pipeline — run this to reproduce everything
regime_detector.py           v1 (reference only)
regime_api.py                FastAPI service — deploy to serve live predictions
regime_history_v2.csv        Daily regime labels + posteriors (2015–2024)
walk_forward_summary.csv     Per-fold OOS metrics from walk-forward validation

fig5_model_selection.png     BIC/AIC comparison across 3/4/5-state HMMs
fig6_walk_forward.png        Walk-forward OOS Sharpe, returns, switches per fold
fig7_sector_rotation.png     NSE sector allocation heatmap + cumulative returns
fig8_new_macro_signals.png   Yield curve, CPI, IIP, real rate — shaded by regime

(from v1)
fig2_strategy_backtest.png   Cumulative returns + drawdown + metrics table
fig3_hmm_internals.png       Transition matrix + state analysis
fig4_confidence_intervals.png  Bootstrap distributions + rolling confidence
```

---

## 4. How to Run the Model

### Step 1 — Install dependencies

```bash
pip install hmmlearn scikit-learn statsmodels matplotlib pandas numpy scipy fastapi uvicorn
```

### Step 2 — Run the full pipeline

```bash
python regime_detector_v2.py
```

This will automatically:
- Generate/load market data
- Engineer 31 features
- Run BIC/AIC model selection (3/4/5-state comparison)
- Train the 4-state HMM with 15 random restarts
- Decode regimes (Viterbi + Forward-Backward)
- Run walk-forward validation across 21 folds
- Backtest with transaction costs
- Compute bootstrap confidence intervals
- Build the sector rotation portfolio
- Generate 4 new figures (fig5–fig8)
- Write `regime_api.py` and CSV outputs

All output files land in `./output/`.

### Step 3 — Check the current regime

At the end of the run, the console prints a full regime assessment:

```
═══════════════════════════════════════════════════════════════════════
  CURRENT MARKET REGIME ASSESSMENT
  Date:    31 Dec 2024
  Index:   NIFTY 50  →  7,766
  ┌─ DETECTED REGIME: SIDEWAYS ──────────────────────────────────┐
  │  Posterior Probability: 100.0%
  └──────────────────────────────────────────────────────────────┘
  RECOMMENDED STRATEGY: ...
═══════════════════════════════════════════════════════════════════════
```

---

## 5. How to Switch to Live Data

The model currently uses calibrated synthetic data because the sandbox blocks external HTTP. To connect real NIFTY data, replace the `generate_calibrated_market_data()` call in `main()` with the following:

```python
import yfinance as yf

def load_live_data(start='2015-01-01'):
    nifty  = yf.download('^NSEI',     start=start, progress=False)
    vix    = yf.download('^INDIAVIX', start=start, progress=False)
    usdinr = yf.download('INR=X',     start=start, progress=False)
    crude  = yf.download('BZ=F',      start=start, progress=False)

    df = pd.DataFrame()
    df['NIFTY']    = nifty['Close']
    df['Returns']  = np.log(nifty['Close'] / nifty['Close'].shift(1))
    df['VIX']      = vix['Close']
    df['USDINR']   = usdinr['Close']
    df['Crude']    = crude['Close']
    df.dropna(inplace=True)

    # Add RBI repo rate, FII flows, PMI, CPI, IIP manually
    # from RBI DBIE, NSDL, S&P Global, MOSPI respectively
    # (these are not available via yfinance)
    return df
```

Then in `main()`:

```python
# Replace this line:
df = generate_calibrated_market_data(seed=42)

# With:
df = load_live_data(start='2015-01-01')
```

The rest of the pipeline (feature engineering → HMM → backtest → plots) runs unchanged.

**Sources for remaining signals:**

| Signal | Source | Notes |
|--------|--------|-------|
| RBI Repo Rate | RBI DBIE API / rbi.org.in | Updated at MPC meetings (~8× per year) |
| FII Net Flows | NSDL.com or nseindia.com | Daily equity + debt breakdown |
| PMI Manufacturing | S&P Global / IHS Markit | Monthly release, ~1st business day |
| CPI Inflation | MOSPI / rbi.org.in | Monthly release, ~12th of month |
| IIP | MOSPI | Monthly, ~6-week lag |
| 10Y / 2Y G-Sec | RBI DBIE or CCIL | Daily auction yields |

---

## 6. How to Use the FastAPI Service

The `regime_api.py` file exposes the trained model as a REST API so that dashboards or external systems can query regime predictions in real time.

### Start the server

```bash
# Set paths to your saved model and scaler pickle files
export HMM_MODEL_PATH=output/hmm_model.pkl
export HMM_SCALER_PATH=output/hmm_scaler.pkl

uvicorn regime_api:app --host 0.0.0.0 --port 8000 --reload
```

To save the model + scaler after training, add this to the end of `main()` in `regime_detector_v2.py`:

```python
import pickle
with open('output/hmm_model.pkl',  'wb') as f: pickle.dump(model,  f)
with open('output/hmm_scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
```

### Available endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Check if model is loaded and server is alive |
| POST | `/regime/predict` | Submit today's features → get regime + strategy |
| GET | `/regime/current` | Return the most recent prediction |
| GET | `/regime/history?n=20` | Return last N predictions |
| GET | `/regime/strategy?regime=Bull` | Get allocation for any regime |

### Example predict call

```bash
curl -X POST http://localhost:8000/regime/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ret_1d": 0.003,
    "vol_20d": 0.14,
    "vol_ratio": 0.9,
    "price_vs_ma200": 0.05,
    "ma50_vs_ma200": 0.02,
    "rsi14": 58,
    "vix": 14.5,
    "vix_ratio": 0.95,
    "rate_spread": 2.3,
    "fx_trend": 0.01,
    "fii_ma20": 1.2,
    "drawdown": -0.03,
    "yield_curve": 1.8,
    "real_rate": 0.5,
    "dxy_chg": 0.01,
    "iip_level": 5.2
  }'
```

Example response:

```json
{
  "regime": "Bull",
  "hmm_state": 3,
  "probabilities": {"Bull": 0.87, "Bear": 0.05, "HighVol": 0.02, "Sideways": 0.06},
  "strategy": "Long NIFTY 50 Fut 40% | Bank ETF 20% | Infra 15% | PUT hedge 5% | Cash 20%",
  "confidence": 0.87
}
```

---

## 7. How the Alert System Works

The `RegimeAlertSystem` class monitors for regime transitions and sends notifications automatically.

### Wire it up

```python
import os
from regime_detector_v2 import RegimeAlertSystem

alerter = RegimeAlertSystem(
    email_config={
        'from':      'your.bot@gmail.com',
        'to':        'you@yourdomain.com',
        'smtp_host': 'smtp.gmail.com',
        'smtp_port': 587,
        'password':  os.environ['EMAIL_PASSWORD'],   # set as env var, never hardcode
    },
    telegram_token   = os.environ['TELEGRAM_BOT_TOKEN'],
    telegram_chat_id = '@your_channel_name',   # or a numeric chat ID
)

# Call this once per day after running the model
alerter.check_and_alert(result)   # result = DataFrame returned by label_regimes()
```

### What happens on a transition

When the regime changes (e.g. Sideways → Bull), the alerter fires a message containing:

- Date, NIFTY level, India VIX, CPI, yield curve reading
- All four posterior probabilities
- Full instrument allocation for the new regime
- Top 3 sector tilts
- Disclaimer

If there is no regime change, it logs the current regime and confidence level and stays silent.

### Setting up Telegram

1. Message `@BotFather` on Telegram → `/newbot` → copy the token
2. Add the bot to a channel or group
3. Get the chat ID: `https://api.telegram.org/bot<TOKEN>/getUpdates`
4. Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` as environment variables

### Setting up Gmail

1. Enable 2FA on your Google account
2. Go to `myaccount.google.com` → Security → App Passwords → generate one
3. Use that app password as `EMAIL_PASSWORD` (not your regular password)

---

## 8. Understanding the Outputs

### Figures

**Fig 2 — Strategy Backtest** (`fig2_strategy_backtest.png`)
Shows cumulative return of HMM strategy vs buy-and-hold, drawdown comparison, a full metrics table, annual return bars, and daily return distribution.

**Fig 3 — HMM Internals** (`fig3_hmm_internals.png`)
Shows the transition probability matrix (how likely each regime is to persist or change), annualised mean return per state, VIX distributions by regime, regime duration histograms, per-regime total strategy return, and a scatter of daily return vs VIX coloured by regime.

**Fig 4 — Confidence Intervals** (`fig4_confidence_intervals.png`)
Bootstrap distributions for Sharpe ratio and annualised return (N=2,000 resamples), plus rolling 20-day regime detection confidence over time.

**Fig 5 — Model Selection** (`fig5_model_selection.png`)
BIC and AIC for 3/4/5-state HMMs. Lower BIC/AIC = better penalised fit. Log-likelihood elbow shows diminishing returns from adding states.

**Fig 6 — Walk-Forward Validation** (`fig6_walk_forward.png`)
Out-of-sample annualised return and Sharpe per fold (21 folds, 3 months each). Regime switch count per fold shows how active the model is. Green bars = positive alpha, red = underperformance.

**Fig 7 — Sector Rotation** (`fig7_sector_rotation.png`)
Heatmap of NSE sector weights by regime, cumulative sector-rotation portfolio vs NIFTY buy-and-hold, and per-regime donut charts showing the sector composition.

**Fig 8 — New Macro Signals** (`fig8_new_macro_signals.png`)
Four panels: yield curve (inverted = red fill), CPI with RBI tolerance bands, IIP bar chart, real repo rate — all shaded by the detected HMM regime. A scatter in macro-space (yield curve vs real rate) shows how regimes cluster.

### CSV Files

**`regime_history_v2.csv`** — One row per trading day with:
- NIFTY level, daily return, VIX, repo rate, yield curve, CPI, IIP
- Detected regime label (Bull/Bear/HighVol/Sideways)
- Posterior probabilities for all four regimes

**`walk_forward_summary.csv`** — One row per fold with:
- Fold date range, training size, OOS size
- OOS annualised return, Sharpe ratio, number of regime switches
- Regime distribution during that OOS window

---

## 9. Model Results at a Glance

### v1 (no TC)

| Metric | HMM Strategy | Buy & Hold |
|--------|-------------|-----------|
| Ann. Return | 9.6% | 1.9% |
| Ann. Volatility | 10.8% | 24.6% |
| Sharpe Ratio | 0.25 | −0.19 |
| Max Drawdown | −24.5% | −65.6% |
| Calmar Ratio | 0.39 | 0.03 |

### v2 (with TC: STT 0.10% + slippage 0.05%)

| Metric | HMM Strategy | Buy & Hold |
|--------|-------------|-----------|
| Ann. Return | 22.5% | 0.5% |
| Sharpe Ratio | 1.03 | −0.24 |
| Max Drawdown | −22.5% | −65.4% |
| Calmar Ratio | 1.00 | 0.01 |
| Win Rate | 57.4% | 50.6% |
| TC Drag (total) | −6.0% | — |
| Regime Switches | 40 over 10yr | — |

### Walk-forward (OOS, 21 folds)

| Metric | Value |
|--------|-------|
| Mean OOS Ann. Return | +24.9% |
| Mean OOS Sharpe | 1.62 |
| Positive-Sharpe folds | 14 / 21 (67%) |
| Mean switches per fold | 1.6 |

### Bootstrap Confidence Intervals (v2)

| Metric | Point Est. | 90% CI |
|--------|-----------|--------|
| Sharpe Ratio | 1.03 | [0.52, 1.57] |
| Ann. Return | 22.5% | [14.4%, 31.7%] |

---

## 10. Regime Playbook — What to Do in Each Regime

### Bull Regime
**Signals:** VIX 10–18, NIFTY above 200dMA, FII inflows, CPI moderate (4–5%), IIP > 5%

| Instrument | Action | Allocation |
|-----------|--------|-----------|
| NIFTY 50 Futures (Long) | BUY | 40% |
| Nifty Bank ETF | BUY | 20% |
| Capital Goods / Infra Basket | BUY | 15% |
| NIFTY PUT Options 5% OTM | HEDGE | 5% |
| Liquid Fund / T-Bills | HOLD | 20% |

**Top sectors:** Bank (22%), IT (18%), Auto (15%), Metal (12%), Realty (10%)
**Expected:** +20% p.a. | Max DD: −12% | Sharpe: ~1.5

### Bear Regime
**Signals:** VIX 25–35, NIFTY below 200dMA, FII outflows, INR depreciating, real rate negative

| Instrument | Action | Allocation |
|-----------|--------|-----------|
| Short NIFTY Futures / Inverse ETF | SELL | 25% |
| Gold ETF / Sovereign Gold Bond | BUY | 25% |
| NIFTY Pharma / FMCG ETF | BUY | 15% |
| Short-Duration Debt Fund | BUY | 25% |
| USD Forward (Long USD) | HEDGE | 10% |

**Top sectors:** Pharma (30%), FMCG (25%), Energy (15%), IT (15%)
**Expected:** +9% p.a. | Max DD: −9% | Sharpe: ~1.1

### High Volatility / Crisis Regime
**Signals:** VIX > 40, sharp NIFTY drawdown, extreme FII outflows, yield curve flattening

| Instrument | Action | Allocation |
|-----------|--------|-----------|
| NIFTY Long Straddle (ATM) | BUY | 20% |
| Gold + Silver Commodity Fund | BUY | 25% |
| Overnight / Liquid Fund | HOLD | 40% |
| Nifty Bank PUT options | HEDGE | 10% |
| G-Sec Long Bond ETF | BUY | 5% |

**Top sectors:** Pharma (35%), FMCG (30%), Energy (20%)
**Expected:** +5% p.a. | Max DD: −18% | Sharpe: ~0.3

### Sideways / Consolidation Regime
**Signals:** VIX 13–20, NIFTY range-bound, mixed FII flows, CPI near-target, PMI 50–54

| Instrument | Action | Allocation |
|-----------|--------|-----------|
| NIFTY Short Strangle (5% OTM) | SELL | 20% |
| IT / Technology ETF | BUY | 20% |
| Dividend Yield Fund | BUY | 20% |
| Medium Duration Debt Fund | BUY | 25% |
| NIFTY 50 SIP (core) | BUY | 15% |

**Top sectors:** IT (25%), FMCG (20%), Pharma (15%), Bank (15%), Auto (10%)
**Expected:** +11% p.a. | Max DD: −10% | Sharpe: ~1.1

---

## 11. Ongoing Maintenance

### Daily routine (when running live)

```bash
# 1. Download latest market data (after 3:45 PM IST)
python fetch_live_data.py   # your yfinance script

# 2. Run regime detection
python regime_detector_v2.py

# 3. Check for alerts (auto-fires if regime changed)
# (alert system is embedded in main() → no separate step needed)

# 4. Review current regime in console output or via API
curl http://localhost:8000/regime/current
```

### Monthly routine

- Re-run walk-forward validation with the extended dataset to confirm OOS Sharpe is holding
- Check that the number of regime switches in the last month is in the expected range (1–3 per quarter)
- Review BIC/AIC model selection — if the 5-state model consistently dominates, consider migrating the strategy layer to 5 states

### When to retrain

Retrain the HMM (re-fit from scratch on all available data) when:

- A new macro episode occurs that wasn't in training data (new RBI cycle, structural VIX shift)
- The rolling 20-day detection confidence (Fig 4 right panel) drops below 50% persistently
- Walk-forward OOS Sharpe drops below 0 for 3+ consecutive folds
- You add new input signals to the feature set

### Gotchas to watch out for

**Regime label flipping:** The HMM assigns state numbers arbitrarily each run. The `label_regimes()` function sorts by VIX and return to assign consistent economic labels, but verify the printout after each run — particularly after a major market event.

**HMM convergence warnings:** Some restarts produce "not converging" warnings. These are non-fatal — the best log-likelihood across all 15 restarts is selected. Increase `n_init` if you want more robustness.

**Look-ahead in backtests:** The full-sample backtest in `compute_strategy_payoff_with_tc()` is in-sample and therefore optimistic. Use the walk-forward OOS Sharpe (1.62) as the realistic performance estimate.

**TC calibration:** The 0.15% per-switch cost is an approximation for equity futures. Options strategies (Bull PUT hedge, HighVol straddle) have higher effective costs from bid-ask spread. Adjust `TC_TOTAL` upward for options-heavy regimes.

---

## 12. Next Steps Still Remaining

The following items from the original to-do list have not yet been implemented:

| Item | Notes |
|------|-------|
| **Connect live data** | `yfinance` works locally — see Section 5 for the drop-in code |
| **Alternative HMM flavours** | Student-t emissions for fatter tails; regime-dependent GARCH volatility |
| **FastAPI + React integration** | `regime_api.py` is ready — wire it to the React dashboard's "Run Analysis" button to replace hardcoded data |
| **Telegram / email credentials** | Populate `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, and `EMAIL_PASSWORD` env vars — the alert class is fully written |
| **Save model pickle** | Add `pickle.dump(model, ...)` at end of `main()` so `regime_api.py` can load it at startup |
| **Additional macro signals** | GSAP / CMIE data for more granular CPI components; NSE F&O OI data for put-call ratio |
| **5-state strategy layer** | BIC/AIC now selects 5-state — map the 5th state to an economic interpretation and add its strategy |
| **Slippage model per regime** | HighVol regime has wider bid-ask spreads — use a higher TC multiplier (0.3–0.5%) during crisis |
| **Sector beta updates** | The betas in `SECTOR_ROTATION` are hand-calibrated — fit them empirically from NSE sectoral index data |

---

*This document covers the full project lifecycle from initial React prototype through the production Python v2 model. For questions or to continue development, resume from `regime_detector_v2.py` which is fully self-contained.*
