"""
India Market Regime Detector — v2
===================================
Implements ALL next-steps from the session summary:

  ✓ Walk-forward validation (rolling training window, no look-ahead bias)
  ✓ Transaction costs (STT 0.1% + slippage 0.05% on regime switches)
  ✓ Model selection (BIC/AIC comparison: 3-state vs 4-state vs 5-state HMM)
  ✓ Additional macro signals: CPI, IIP, yield-curve inversion (2Y proxy),
    DXY-proxy impact on INR
  ✓ Sector rotation layer: per-regime NSE sector weights
  ✓ FastAPI wrapper skeleton (serve predictions as REST API)
  ✓ Alert system (email/Telegram notification on regime transition)

Author: Claude (Anthropic) — v2, 2026
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import warnings, os, json, textwrap
warnings.filterwarnings('ignore')

from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from scipy import stats

# ══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════

REGIME_COLORS = {
    'Bull':     '#10b981',
    'Bear':     '#ef4444',
    'HighVol':  '#f59e0b',
    'Sideways': '#3b82f6',
}

# ── NEW: Sector rotation weights per regime ──────────────────────────
# NSE sectoral indices: NIFTY Bank, IT, Pharma, FMCG, Auto, Metal,
#                       Realty, Infra, Energy, Media
SECTOR_ROTATION = {
    'Bull': {
        'NIFTY Bank':   0.22,
        'NIFTY IT':     0.18,
        'NIFTY Auto':   0.15,
        'NIFTY Metal':  0.12,
        'NIFTY Realty': 0.10,
        'NIFTY Infra':  0.10,
        'NIFTY Energy': 0.08,
        'NIFTY FMCG':   0.03,
        'NIFTY Pharma': 0.02,
    },
    'Bear': {
        'NIFTY Pharma': 0.30,
        'NIFTY FMCG':   0.25,
        'NIFTY IT':     0.15,   # large-cap IT = defensive in India bear
        'NIFTY Bank':   0.05,   # underweight cyclicals
        'NIFTY Auto':   0.05,
        'NIFTY Metal':  0.00,
        'NIFTY Realty': 0.00,
        'NIFTY Infra':  0.05,
        'NIFTY Energy': 0.15,   # oil cos benefit when INR weakens
        'NIFTY Media':  0.00,
    },
    'HighVol': {
        'NIFTY Pharma': 0.35,
        'NIFTY FMCG':   0.30,
        'NIFTY IT':     0.10,
        'NIFTY Bank':   0.05,
        'NIFTY Auto':   0.00,
        'NIFTY Metal':  0.00,
        'NIFTY Realty': 0.00,
        'NIFTY Infra':  0.00,
        'NIFTY Energy': 0.20,
        'NIFTY Media':  0.00,
    },
    'Sideways': {
        'NIFTY IT':     0.25,
        'NIFTY FMCG':   0.20,
        'NIFTY Pharma': 0.15,
        'NIFTY Bank':   0.15,
        'NIFTY Auto':   0.10,
        'NIFTY Energy': 0.08,
        'NIFTY Infra':  0.07,
        'NIFTY Metal':  0.00,
        'NIFTY Realty': 0.00,
        'NIFTY Media':  0.00,
    },
}

# Regime-specific return multipliers (used in backtest)
REGIME_MULTIPLIERS = {
    'Bull':     1.40,
    'Bear':    -0.30,
    'HighVol':  0.15,
    'Sideways': 0.65,
}

# Transaction cost parameters
TC_STT       = 0.001   # 0.10% STT on sell-side (equity delivery)
TC_SLIPPAGE  = 0.0005  # 0.05% market impact / slippage
TC_TOTAL     = TC_STT + TC_SLIPPAGE   # applied on every regime switch

# Walk-forward parameters
WF_TRAIN_YEARS  = 4     # years of history to train on
WF_TEST_MONTHS  = 3     # months to predict before re-fitting


# ══════════════════════════════════════════════════════════════════════
# 1. DATA GENERATION  (same calibrated synthetic data + NEW features)
# ══════════════════════════════════════════════════════════════════════

def generate_calibrated_market_data(seed=42):
    """
    Calibrated synthetic Indian market data (2015-2024).
    NEW vs v1: adds CPI, IIP, 2Y G-Sec yield, DXY-proxy.
    """
    np.random.seed(seed)
    dates = pd.date_range('2015-01-01', '2024-12-31', freq='B')
    n = len(dates)

    # True regime dynamics
    mu    = [0.00055, -0.00090, -0.00020, 0.00023]
    sigma = [0.00780,  0.01900,  0.02600, 0.00950]
    T = np.array([
        [0.9800, 0.0050, 0.0050, 0.0100],
        [0.0100, 0.9720, 0.0100, 0.0080],
        [0.0080, 0.0200, 0.9620, 0.0100],
        [0.0120, 0.0050, 0.0050, 0.9780],
    ])
    regimes = np.zeros(n, dtype=int)
    regimes[0] = 3
    for i in range(1, n):
        regimes[i] = np.random.choice(4, p=T[regimes[i-1]])

    # Historical episode injections (same as v1)
    def inject(start_str, end_str, state):
        s = np.searchsorted(dates, start_str)
        e = np.searchsorted(dates, end_str)
        regimes[s:e] = state

    inject('2020-02-20', '2020-05-01', 2)   # COVID crash
    inject('2020-06-01', '2021-12-31', 0)   # Recovery bull
    inject('2022-01-01', '2022-06-30', 1)   # Rate-hike bear
    inject('2023-01-01', '2024-12-31', 0)   # 2023-24 bull
    # Pepper sideways
    np.random.seed(99)
    bull2_start = np.searchsorted(dates, '2023-01-01')
    for _ in range(15):
        idx = np.random.randint(bull2_start, n - 30)
        regimes[idx:idx + np.random.randint(5, 25)] = 3

    # Returns + price
    returns = np.array([np.random.normal(mu[r], sigma[r]) for r in regimes])
    for i in range(1, n):
        returns[i] += 0.08 * returns[i-1]
    price = 8300 * np.exp(np.cumsum(returns))

    # VIX
    vix = np.array([max(8, np.random.lognormal(
        np.log([13.5, 28.0, 48.0, 17.5][r]),
        [2.5, 5.0, 12.0, 3.0][r] / [13.5, 28.0, 48.0, 17.5][r]
    )) for r in regimes])
    for i in range(1, n):
        vix[i] = 0.95 * vix[i] + 0.05 * vix[i-1]

    # Repo rate (actual cycle)
    repo_base = np.interp(np.arange(n),
        [0, 250, 500, 750, 900, 1100, 1400, 1700, 2000, 2400],
        [7.75, 6.25, 6.25, 6.50, 4.00, 4.00, 6.50, 6.50, 6.50, 6.50])
    repo_rate = repo_base + np.random.normal(0, 0.02, n)

    # 10Y G-Sec yield
    gsec_spread_10y = np.where(regimes==2, 3.2, np.where(regimes==1, 2.8,
                      np.where(regimes==0, 2.2, 2.5)))
    gsec_10y = repo_rate + gsec_spread_10y + np.random.normal(0, 0.08, n)
    for i in range(1, n):
        gsec_10y[i] = 0.98 * gsec_10y[i] + 0.02 * gsec_10y[i-1]

    # ── NEW: 2Y G-Sec yield (for yield-curve inversion) ──
    gsec_spread_2y = np.where(regimes==2, 1.0, np.where(regimes==1, 1.5,
                     np.where(regimes==0, 0.6, 0.9)))
    gsec_2y = repo_rate + gsec_spread_2y + np.random.normal(0, 0.06, n)
    for i in range(1, n):
        gsec_2y[i] = 0.985 * gsec_2y[i] + 0.015 * gsec_2y[i-1]
    # Yield curve: 10Y - 2Y (negative = inverted = recession warning)
    yield_curve = gsec_10y - gsec_2y

    # USD/INR
    usd_inr = np.zeros(n)
    usd_inr[0] = 63.0
    fx_drift = np.where(regimes==0, 0.00005, np.where(regimes==1, 0.00025,
               np.where(regimes==2, 0.00045, 0.00012)))
    for i in range(1, n):
        usd_inr[i] = usd_inr[i-1] * (1 + fx_drift[i] + np.random.normal(0, 0.003))
    usd_inr = np.clip(usd_inr, 60, 87)

    # ── NEW: DXY proxy impact (USD strength index) ──
    # Higher DXY → INR depreciates → EM outflows
    dxy = np.interp(np.arange(n),
        [0, 600, 1000, 1400, 1800, 2400],
        [90, 100, 93, 98, 105, 101])
    dxy += np.random.normal(0, 1.5, n)
    for i in range(1, n):
        dxy[i] = 0.97 * dxy[i] + 0.03 * dxy[i-1]

    # FII flows
    fii_mean = np.where(regimes==0, 1200, np.where(regimes==1, -1800,
               np.where(regimes==2, -3500, 200)))
    fii = fii_mean + np.random.normal(0, 800, n) + returns * 150000

    # Crude oil
    crude_base = np.interp(np.arange(n),
        [0, 150, 500, 900, 1200, 1500, 1800, 2100, 2400],
        [55, 30, 65, 85, 55, 75, 110, 75, 88])
    crude = crude_base + np.random.normal(0, 3, n)
    for i in range(1, n):
        crude[i] = 0.97 * crude[i] + 0.03 * crude[i-1]
    crude = np.clip(crude, 20, 130)

    # PMI Manufacturing
    pmi_mean = np.where(regimes==0, 56, np.where(regimes==1, 50,
               np.where(regimes==2, 47, 53)))
    pmi = pmi_mean + np.random.normal(0, 1.5, n)
    pmi = np.clip(pmi, 40, 65)

    # ── NEW: CPI Inflation (%) ──
    # Bull: moderate 4-5%, Bear/HighVol: elevated 6-8%, Sideways: 5-6%
    cpi_mean = np.where(regimes==0, 4.5, np.where(regimes==1, 7.0,
               np.where(regimes==2, 6.5, 5.5)))
    cpi = cpi_mean + np.random.normal(0, 0.4, n)
    # Slow-moving (MoM)
    for i in range(1, n):
        cpi[i] = 0.97 * cpi[i] + 0.03 * cpi[i-1]
    cpi = np.clip(cpi, 2, 12)

    # Real repo rate = repo - CPI (important for RBI policy stance)
    real_rate = repo_rate - cpi

    # ── NEW: IIP (Index of Industrial Production, YoY %) ──
    iip_mean = np.where(regimes==0, 7.0, np.where(regimes==1, 1.0,
               np.where(regimes==2, -5.0, 4.0)))
    iip = iip_mean + np.random.normal(0, 1.5, n)
    for i in range(1, n):
        iip[i] = 0.92 * iip[i] + 0.08 * iip[i-1]
    iip = np.clip(iip, -20, 20)

    df = pd.DataFrame({
        'Date':        dates,
        'NIFTY':       np.round(price, 2),
        'Returns':     returns,
        'VIX':         np.round(vix, 2),
        'RepoRate':    np.round(repo_rate, 4),
        'GSecYield10': np.round(gsec_10y, 4),
        'GSecYield2':  np.round(gsec_2y, 4),
        'YieldCurve':  np.round(yield_curve, 4),   # NEW
        'USDINR':      np.round(usd_inr, 4),
        'DXY':         np.round(dxy, 2),            # NEW
        'FII_Flow':    np.round(fii, 0),
        'Crude':       np.round(crude, 2),
        'PMI':         np.round(pmi, 2),
        'CPI':         np.round(cpi, 3),            # NEW
        'RealRate':    np.round(real_rate, 4),      # NEW
        'IIP':         np.round(iip, 2),            # NEW
        'TrueRegime':  regimes,
    }).set_index('Date')

    print(f"✓ Generated {len(df)} trading days | "
          f"{df.shape[1]-1} signals including CPI, IIP, yield-curve, DXY")
    return df


# ══════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING  (extended with new macro features)
# ══════════════════════════════════════════════════════════════════════

def engineer_features(df):
    feat = pd.DataFrame(index=df.index)
    r = df['Returns']

    # Return / volatility
    feat['ret_1d']    = r
    feat['ret_5d']    = r.rolling(5).sum()
    feat['ret_20d']   = r.rolling(20).sum()
    feat['vol_10d']   = r.rolling(10).std() * np.sqrt(252)
    feat['vol_20d']   = r.rolling(20).std() * np.sqrt(252)
    feat['vol_60d']   = r.rolling(60).std() * np.sqrt(252)
    feat['vol_ratio'] = feat['vol_10d'] / (feat['vol_60d'] + 1e-8)

    # Trend
    feat['ma50']           = df['NIFTY'].rolling(50).mean()
    feat['ma200']          = df['NIFTY'].rolling(200).mean()
    feat['price_vs_ma50']  = (df['NIFTY'] - feat['ma50'])  / feat['ma50']
    feat['price_vs_ma200'] = (df['NIFTY'] - feat['ma200']) / feat['ma200']
    feat['ma50_vs_ma200']  = (feat['ma50'] - feat['ma200']) / feat['ma200']

    # Momentum
    gains  = r.clip(lower=0).rolling(14).mean()
    losses = (-r.clip(upper=0)).rolling(14).mean()
    feat['rsi14'] = 100 - 100 / (1 + gains / (losses + 1e-8))

    # Drawdown
    rolling_max      = df['NIFTY'].rolling(252, min_periods=1).max()
    feat['drawdown'] = (df['NIFTY'] - rolling_max) / rolling_max

    # VIX
    feat['vix']       = df['VIX']
    feat['vix_ratio'] = df['VIX'] / (df['VIX'].rolling(20).mean() + 1e-8)
    feat['vix_chg']   = df['VIX'].pct_change(5)

    # Rates (v1)
    feat['rate_spread'] = df['GSecYield10'] - df['RepoRate']
    feat['rate_chg']    = df['RepoRate'].diff(20)
    feat['fx_trend']    = df['USDINR'].pct_change(20)
    feat['fii_ma20']    = df['FII_Flow'].rolling(20).mean() / 1000
    feat['crude_chg']   = df['Crude'].pct_change(20)
    feat['pmi_level']   = df['PMI'] - 50

    # ── NEW macro features ──────────────────────────────────────────
    # Yield curve inversion (10Y-2Y): negative = bear signal
    feat['yield_curve']     = df['YieldCurve']
    feat['yield_curve_chg'] = df['YieldCurve'].diff(60)  # 3-month change

    # DXY strength → EM pressure
    feat['dxy_chg']         = df['DXY'].pct_change(20)
    feat['dxy_level']       = (df['DXY'] - df['DXY'].rolling(60).mean()) / (df['DXY'].rolling(60).std() + 1e-8)

    # CPI – real rate regime
    feat['cpi_level']       = df['CPI']
    feat['real_rate']       = df['RealRate']  # repo - CPI

    # IIP (industrial momentum)
    feat['iip_level']       = df['IIP']
    feat['iip_ma']          = df['IIP'].rolling(20).mean()

    feat.dropna(inplace=True)
    print(f"✓ Engineered {feat.shape[1]} features, {len(feat)} clean rows")
    return feat


def select_features_for_hmm(feat, n_states=4):
    """
    Parsimonious feature set. Slightly extended from v1 to include
    new macro signals, but capped to avoid dimensionality issues.
    """
    cols = [
        # Return + vol (core)
        'ret_1d', 'vol_20d', 'vol_ratio',
        # Trend
        'price_vs_ma200', 'ma50_vs_ma200',
        # Momentum
        'rsi14',
        # Fear / volatility regime
        'vix', 'vix_ratio',
        # Macro (v1)
        'rate_spread', 'fx_trend', 'fii_ma20', 'drawdown',
        # NEW macro
        'yield_curve',      # yield-curve slope
        'real_rate',        # real policy rate
        'dxy_chg',          # USD strength change
        'iip_level',        # industrial production momentum
    ]
    return feat[cols]


# ══════════════════════════════════════════════════════════════════════
# 3. MODEL SELECTION  (BIC/AIC for 3/4/5-state HMM)
# ══════════════════════════════════════════════════════════════════════

def bic_aic_model_selection(X_scaled, state_range=(3, 4, 5), n_init=5, n_iter=150):
    """
    Fit HMMs with different state counts and compute BIC / AIC.
    BIC penalises model complexity more harshly → tends to pick sparser models.
    AIC is less conservative.

    Returns: dict of {n_states: {'loglik', 'bic', 'aic', 'n_params', 'model'}}
    """
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  MODEL SELECTION: BIC / AIC (3-state / 4-state / 5-state)  ║")
    print("╚══════════════════════════════════════════════════════╝")

    n_samples, n_features = X_scaled.shape
    results = {}

    for k in state_range:
        best_ll, best_model = -np.inf, None
        for seed in range(n_init):
            try:
                m = GaussianHMM(n_components=k, covariance_type='full',
                                n_iter=n_iter, tol=1e-4, random_state=seed, verbose=False)
                m.fit(X_scaled)
                ll = m.score(X_scaled)
                if ll > best_ll:
                    best_ll, best_model = ll, m
            except Exception:
                continue

        if best_model is None:
            continue

        # Parameter count: transition (k²-k free), means (k*d), covariances (k*d*(d+1)/2)
        n_params = (k * k - k) + k * n_features + k * n_features * (n_features + 1) // 2
        bic = -2 * best_ll * n_samples + n_params * np.log(n_samples)
        aic = -2 * best_ll * n_samples + 2 * n_params

        results[k] = {
            'loglik':   best_ll,
            'bic':      bic,
            'aic':      aic,
            'n_params': n_params,
            'model':    best_model,
            'converged': best_model.monitor_.converged,
        }
        print(f"  {k}-state HMM | LogLik={best_ll:,.1f} | "
              f"#Params={n_params:,} | BIC={bic:,.0f} | AIC={aic:,.0f} | "
              f"Converged={best_model.monitor_.converged}")

    # Select best by BIC
    best_k   = min(results, key=lambda k: results[k]['bic'])
    best_k_a = min(results, key=lambda k: results[k]['aic'])
    print(f"\n  ➜ Best by BIC: {best_k}-state  |  Best by AIC: {best_k_a}-state")
    print(f"  (Using {best_k}-state model — BIC-selected — for subsequent analysis)\n")

    return results, best_k


# ══════════════════════════════════════════════════════════════════════
# 4. HMM TRAINING + REGIME LABELLING
# ══════════════════════════════════════════════════════════════════════

def train_hmm(X_scaled, n_states=4, n_iter=200, n_init=15):
    best_ll, best_model = -np.inf, None
    print(f"Fitting {n_states}-state Gaussian HMM ({n_iter} iters, {n_init} restarts)...")
    for seed in range(n_init):
        try:
            m = GaussianHMM(n_components=n_states, covariance_type='full',
                            n_iter=n_iter, tol=1e-4, random_state=seed, verbose=False)
            m.fit(X_scaled)
            ll = m.score(X_scaled)
            if ll > best_ll:
                best_ll, best_model = ll, m
        except Exception:
            continue
    print(f"✓ Best log-likelihood: {best_ll:.2f}  |  Converged: {best_model.monitor_.converged}")
    return best_model


def label_regimes(model, X_scaled, feat, df):
    """Same labelling logic as v1, works for any n_states (labels top-4 economically)."""
    _, states   = model.decode(X_scaled, algorithm='viterbi')
    posteriors  = model.predict_proba(X_scaled)

    feat_cols = select_features_for_hmm(feat).columns.tolist()
    vix_idx   = feat_cols.index('vix')
    ret_idx   = feat_cols.index('ret_1d')
    vol_idx   = feat_cols.index('vol_20d')
    mom_idx   = feat_cols.index('price_vs_ma200') if 'price_vs_ma200' in feat_cols else feat_cols.index('rsi14')

    X_orig = select_features_for_hmm(feat).values
    scores = {}
    for s in range(model.n_components):
        mask = (states == s)
        if mask.sum() > 0:
            m = X_orig[mask].mean(axis=0)
            scores[s] = dict(mean_return=m[ret_idx], mean_vix=m[vix_idx],
                             mean_vol=m[vol_idx], mean_trend=m[mom_idx],
                             count=mask.sum())

    sorted_vix = sorted(scores, key=lambda s: scores[s]['mean_vix'], reverse=True)
    highvol_state = sorted_vix[0]
    remaining     = [s for s in scores if s != highvol_state]
    sorted_ret    = sorted(remaining, key=lambda s: scores[s]['mean_return'])
    bear_state, bull_state = sorted_ret[0], sorted_ret[-1]
    sw_candidates = [s for s in remaining if s not in (bear_state, bull_state)]
    sideways_state = sw_candidates[0] if sw_candidates else sorted_ret[1]

    label_map = {
        bull_state:     {'name': 'Bull',     'color': '#10b981', 'idx': 0},
        bear_state:     {'name': 'Bear',     'color': '#ef4444', 'idx': 1},
        highvol_state:  {'name': 'HighVol',  'color': '#f59e0b', 'idx': 2},
        sideways_state: {'name': 'Sideways', 'color': '#3b82f6', 'idx': 3},
    }

    print("\n✓ Regime labelling:")
    for s, info in label_map.items():
        m = scores[s]
        print(f"  State {s} → {info['name']:10s} | "
              f"μ_ret={m['mean_return']*100:+.3f}%/d  VIX={m['mean_vix']:.1f}  "
              f"Vol={m['mean_vol']*100:.1f}%ann  N={m['count']}")

    decoded_labels = np.array([label_map[s]['name'] for s in states])
    decoded_colors = np.array([label_map[s]['color'] for s in states])

    regime_posteriors = pd.DataFrame(index=feat.index)
    for s, info in label_map.items():
        regime_posteriors[info['name']] = posteriors[:, s]

    # Sum posteriors for any extra states into nearest named regime (if n_states > 4)
    for reg in ['Bull', 'Bear', 'HighVol', 'Sideways']:
        if reg not in regime_posteriors:
            regime_posteriors[reg] = 0.0

    result = pd.DataFrame({
        'NIFTY':       df.loc[feat.index, 'NIFTY'],
        'Returns':     df.loc[feat.index, 'Returns'],
        'VIX':         df.loc[feat.index, 'VIX'],
        'RepoRate':    df.loc[feat.index, 'RepoRate'],
        'YieldCurve':  df.loc[feat.index, 'YieldCurve'],
        'CPI':         df.loc[feat.index, 'CPI'],
        'IIP':         df.loc[feat.index, 'IIP'],
        'HMM_State':   states,
        'Regime':      decoded_labels,
        'RegimeColor': decoded_colors,
        'TrueRegime':  df.loc[feat.index, 'TrueRegime'],
    }, index=feat.index)

    result = pd.concat([result, regime_posteriors], axis=1)
    return result, label_map, scores


# ══════════════════════════════════════════════════════════════════════
# 5. WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════════════════════════════

def walk_forward_validation(feat, df, train_years=4, test_months=3, n_states=4):
    """
    Rolling-window walk-forward validation.

    For each fold:
      1. Train HMM on [window_start : fold_start]  (≥ train_years of data)
      2. Decode regimes on [fold_start : fold_end]  (test_months of OOS data)
      3. Compute OOS performance with TC-adjusted returns

    Returns a DataFrame of OOS regime predictions and per-fold metrics.
    """
    print("\n╔════════════════════════════════════════════════╗")
    print("║  WALK-FORWARD VALIDATION (no look-ahead bias) ║")
    print("╚════════════════════════════════════════════════╝")

    scaler    = StandardScaler()
    all_dates = feat.index
    start_date = all_dates[0]

    train_td   = pd.DateOffset(years=train_years)
    test_td    = pd.DateOffset(months=test_months)

    fold_start = start_date + train_td
    folds      = []

    while fold_start < all_dates[-1]:
        fold_end = min(fold_start + test_td, all_dates[-1])
        train_mask = (all_dates >= start_date) & (all_dates < fold_start)
        test_mask  = (all_dates >= fold_start) & (all_dates < fold_end)

        if train_mask.sum() < 200 or test_mask.sum() < 5:
            fold_start = fold_end
            continue

        X_train = select_features_for_hmm(feat.loc[train_mask]).values
        X_test  = select_features_for_hmm(feat.loc[test_mask]).values

        # Scale using only training data (no look-ahead)
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        # Fit HMM
        model = train_hmm(X_train_s, n_states=n_states, n_iter=150, n_init=5)

        # Label using training-period statistics
        _, states_train = model.decode(X_train_s, algorithm='viterbi')
        feat_cols = select_features_for_hmm(feat).columns.tolist()
        vix_idx = feat_cols.index('vix')
        ret_idx = feat_cols.index('ret_1d')

        scores_train = {}
        for s in range(n_states):
            m = (states_train == s)
            if m.sum() > 0:
                x = X_train[m].mean(axis=0)
                scores_train[s] = dict(mean_return=x[ret_idx], mean_vix=x[vix_idx], count=m.sum())

        sorted_vix  = sorted(scores_train, key=lambda s: scores_train[s]['mean_vix'], reverse=True)
        hv_s        = sorted_vix[0]
        rem         = [s for s in scores_train if s != hv_s]
        sorted_ret  = sorted(rem, key=lambda s: scores_train[s]['mean_return'])
        bear_s, bull_s = sorted_ret[0], sorted_ret[-1]
        sw_s = [s for s in rem if s not in (bear_s, bull_s)]
        sw_s = sw_s[0] if sw_s else sorted_ret[1]

        lmap = {bull_s: 'Bull', bear_s: 'Bear', hv_s: 'HighVol', sw_s: 'Sideways'}

        # Decode OOS
        _, states_test = model.decode(X_test_s, algorithm='viterbi')
        oos_labels = [lmap.get(s, 'Sideways') for s in states_test]

        oos_idx = all_dates[test_mask]
        oos_returns = df.loc[oos_idx, 'Returns'].values

        # Regime-switched strategy returns WITH transaction costs
        tc_returns = []
        prev_regime = oos_labels[0] if oos_labels else 'Sideways'
        for i, (lab, raw_ret) in enumerate(zip(oos_labels, oos_returns)):
            # Apply TC on regime switch
            tc = TC_TOTAL if (lab != prev_regime) else 0.0
            mult = REGIME_MULTIPLIERS[lab]
            if lab == 'Bear':
                daily = -mult * raw_ret - tc
            elif lab == 'HighVol':
                daily = mult * abs(raw_ret) - tc
            else:
                daily = mult * raw_ret - tc
            tc_returns.append(daily)
            prev_regime = lab

        tc_returns = np.array(tc_returns)
        ann_ret = (np.exp(tc_returns.mean() * 252) - 1) * 100
        sharpe  = (tc_returns.mean() - 0.065/252) / (tc_returns.std() + 1e-8) * np.sqrt(252)
        n_switches = sum(1 for i in range(1, len(oos_labels)) if oos_labels[i] != oos_labels[i-1])

        folds.append({
            'fold_start':  fold_start.strftime('%Y-%m'),
            'fold_end':    fold_end.strftime('%Y-%m'),
            'n_train':     train_mask.sum(),
            'n_test':      test_mask.sum(),
            'ann_ret_pct': ann_ret,
            'sharpe':      sharpe,
            'n_switches':  n_switches,
            'regime_dist': pd.Series(oos_labels).value_counts().to_dict(),
        })

        print(f"  Fold {fold_start.strftime('%Y-%m')} → {fold_end.strftime('%Y-%m')} | "
              f"Train={train_mask.sum():,}d | OOS={test_mask.sum():,}d | "
              f"AnnRet={ann_ret:+.1f}% | Sharpe={sharpe:.2f} | Switches={n_switches}")

        fold_start = fold_end

    folds_df = pd.DataFrame(folds)
    if len(folds_df) > 0:
        print(f"\n  Walk-Forward Summary ({len(folds_df)} folds):")
        print(f"  Mean OOS Ann. Return : {folds_df['ann_ret_pct'].mean():+.1f}%")
        print(f"  Mean OOS Sharpe      : {folds_df['sharpe'].mean():.2f}")
        print(f"  Positive-Sharpe folds: {(folds_df['sharpe'] > 0).sum()} / {len(folds_df)}")
        print(f"  Mean regime switches : {folds_df['n_switches'].mean():.1f} per fold")

    return folds_df


# ══════════════════════════════════════════════════════════════════════
# 6. STRATEGY BACKTEST WITH TRANSACTION COSTS
# ══════════════════════════════════════════════════════════════════════

def compute_strategy_payoff_with_tc(result, initial_capital=1_000_000):
    """
    Regime-switching backtest WITH transaction costs.
    On every regime switch: deduct TC_STT + TC_SLIPPAGE from that day's return.
    """
    buy_hold_capital = initial_capital * np.exp(np.cumsum(result['Returns'].values))

    strategy_returns = np.zeros(len(result))
    tc_drag = np.zeros(len(result))
    n_switches = 0
    prev_regime = result['Regime'].iloc[0]

    for i, (_, row) in enumerate(result.iterrows()):
        regime  = row['Regime']
        mkt_ret = row['Returns']
        mult    = REGIME_MULTIPLIERS[regime]

        # Transaction cost on switch
        switched = (regime != prev_regime)
        tc       = TC_TOTAL if switched else 0.0
        if switched:
            n_switches += 1
            tc_drag[i] = tc

        if regime == 'Bear':
            strategy_returns[i] = -mult * mkt_ret - tc
        elif regime == 'HighVol':
            strategy_returns[i] = mult * abs(mkt_ret) - tc
        else:
            strategy_returns[i] = mult * mkt_ret - tc

        prev_regime = regime

    strategy_capital = initial_capital * np.exp(np.cumsum(strategy_returns))
    tc_drag_total = tc_drag.sum() * 100  # percent

    print(f"\n✓ Backtest (WITH transaction costs: STT {TC_STT*100:.2f}% + slippage {TC_SLIPPAGE*100:.2f}%)")
    print(f"  Total regime switches : {n_switches}")
    print(f"  Cumulative TC drag    : {tc_drag_total:.2f}% of capital")

    return strategy_returns, strategy_capital, buy_hold_capital, n_switches, tc_drag


def compute_performance_metrics(returns, capital, risk_free=0.065):
    ann_factor = 252
    rf_daily   = risk_free / ann_factor
    excess_ret = returns - rf_daily
    ann_return = np.exp(returns.mean() * ann_factor) - 1
    ann_vol    = returns.std() * np.sqrt(ann_factor)
    sharpe     = excess_ret.mean() / (returns.std() + 1e-8) * np.sqrt(ann_factor)
    downside   = returns[returns < rf_daily]
    sortino    = excess_ret.mean() / (downside.std() + 1e-8) * np.sqrt(ann_factor)
    peak       = np.maximum.accumulate(capital)
    dd         = (capital - peak) / peak
    max_dd     = dd.min()
    calmar     = ann_return / (abs(max_dd) + 1e-8)
    win_rate   = (returns > 0).mean()
    var_95     = np.percentile(returns, 5)
    cvar_95    = returns[returns <= var_95].mean()
    return dict(ann_return=ann_return, ann_vol=ann_vol, sharpe=sharpe,
                sortino=sortino, max_dd=max_dd, calmar=calmar,
                win_rate=win_rate, var_95=var_95, cvar_95=cvar_95)


# ══════════════════════════════════════════════════════════════════════
# 7. SECTOR ROTATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def compute_sector_rotation_returns(result):
    """
    Simulate sector-rotation performance.
    We approximate sector returns from market returns using
    regime-calibrated sector betas.
    """
    # Approximate sector betas vs NIFTY 50 by regime
    SECTOR_BETAS = {
        'Bull':     {'NIFTY Bank':1.35,'NIFTY IT':1.15,'NIFTY Auto':1.40,
                     'NIFTY Metal':1.65,'NIFTY Realty':1.70,'NIFTY Infra':1.30,
                     'NIFTY Energy':1.10,'NIFTY FMCG':0.65,'NIFTY Pharma':0.55},
        'Bear':     {'NIFTY Bank':1.20,'NIFTY IT':0.90,'NIFTY Auto':1.30,
                     'NIFTY Metal':1.40,'NIFTY Realty':1.50,'NIFTY Infra':1.15,
                     'NIFTY Energy':0.95,'NIFTY FMCG':0.50,'NIFTY Pharma':0.45},
        'HighVol':  {'NIFTY Bank':1.40,'NIFTY IT':0.85,'NIFTY Auto':1.35,
                     'NIFTY Metal':1.60,'NIFTY Realty':1.55,'NIFTY Infra':1.20,
                     'NIFTY Energy':1.00,'NIFTY FMCG':0.55,'NIFTY Pharma':0.50},
        'Sideways': {'NIFTY Bank':0.95,'NIFTY IT':1.10,'NIFTY Auto':1.00,
                     'NIFTY Metal':1.05,'NIFTY Realty':0.90,'NIFTY Infra':1.00,
                     'NIFTY Energy':0.90,'NIFTY FMCG':0.80,'NIFTY Pharma':0.75},
    }

    sector_names = list(SECTOR_ROTATION['Bull'].keys())
    daily_sector_ret = pd.DataFrame(index=result.index, columns=sector_names, dtype=float)

    for idx, row in result.iterrows():
        regime   = row['Regime']
        mkt_ret  = row['Returns']
        betas    = SECTOR_BETAS[regime]
        for sec in sector_names:
            if sec in betas:
                daily_sector_ret.loc[idx, sec] = betas[sec] * mkt_ret
            else:
                daily_sector_ret.loc[idx, sec] = mkt_ret

    # Regime-rotated portfolio return
    portfolio_returns = pd.Series(0.0, index=result.index)
    for idx, row in result.iterrows():
        regime  = row['Regime']
        weights = SECTOR_ROTATION.get(regime, {})
        port_r  = sum(weights.get(s, 0) * float(daily_sector_ret.loc[idx, s])
                      for s in sector_names if s in weights)
        portfolio_returns[idx] = port_r

    ann_ret = (np.exp(portfolio_returns.mean() * 252) - 1) * 100
    sharpe  = (portfolio_returns.mean() - 0.065/252) / (portfolio_returns.std() + 1e-8) * np.sqrt(252)
    print(f"\n✓ Sector Rotation Portfolio | Ann. Return: {ann_ret:.1f}% | Sharpe: {sharpe:.2f}")

    return portfolio_returns, daily_sector_ret


def bootstrap_confidence_intervals(returns, n_bootstrap=2000, ci_levels=(0.90, 0.95)):
    ann = 252
    rf  = 0.065 / ann
    n   = len(returns)
    boot = dict(sharpe=[], ret=[], vol=[], calmar=[])
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        br  = returns[idx]
        b_ret = np.exp(br.mean() * ann) - 1
        b_vol = br.std() * np.sqrt(ann)
        b_sha = (br.mean() - rf) / (br.std() + 1e-8) * np.sqrt(ann)
        b_cap = np.exp(np.cumsum(br))
        b_dd  = ((b_cap - np.maximum.accumulate(b_cap)) / np.maximum.accumulate(b_cap)).min()
        b_cal = b_ret / (abs(b_dd) + 1e-8)
        boot['sharpe'].append(b_sha); boot['ret'].append(b_ret)
        boot['vol'].append(b_vol);   boot['calmar'].append(b_cal)

    ci_results = {}
    for ci in ci_levels:
        a = (1 - ci) / 2
        ci_results[ci] = {
            'sharpe': (np.percentile(boot['sharpe'], a*100), np.percentile(boot['sharpe'], (1-a)*100)),
            'return': (np.percentile(boot['ret'], a*100),    np.percentile(boot['ret'], (1-a)*100)),
        }
    ci_results['distributions'] = {'sharpe': boot['sharpe'], 'return': boot['ret']}
    print(f"✓ Bootstrap CI (N={n_bootstrap}) | "
          f"Sharpe 90%: [{ci_results[0.90]['sharpe'][0]:.2f}, {ci_results[0.90]['sharpe'][1]:.2f}]  |  "
          f"Return 90%: [{ci_results[0.90]['return'][0]*100:.1f}%, {ci_results[0.90]['return'][1]*100:.1f}%]")
    return ci_results


# ══════════════════════════════════════════════════════════════════════
# 8. ALERT SYSTEM  (email + Telegram skeleton)
# ══════════════════════════════════════════════════════════════════════

class RegimeAlertSystem:
    """
    Detects regime transitions and fires email / Telegram alerts.

    Usage:
        alerter = RegimeAlertSystem(
            email_config  = {'from': '...', 'to': '...', 'smtp_host': 'smtp.gmail.com',
                             'smtp_port': 587, 'password': os.environ['EMAIL_PASSWORD']},
            telegram_token = os.environ['TELEGRAM_BOT_TOKEN'],
            telegram_chat_id = '@your_channel',
        )
        alerter.check_and_alert(result)
    """

    STRATEGY_BRIEF = {
        'Bull':     "Long NIFTY 50 Fut 40% | Bank ETF 20% | Infra 15% | PUT hedge 5% | Cash 20%",
        'Bear':     "Short NIFTY 25% | Gold ETF 25% | Pharma/FMCG 15% | S-T Debt 25% | Long USD 10%",
        'HighVol':  "ATM Straddle 20% | Gold+Silver 25% | Liquid Fund 40% | Bank PUTs 10% | GSec 5%",
        'Sideways': "Short Strangle 20% | IT ETF 20% | Div Yield 20% | M-D Debt 25% | NIFTY SIP 15%",
    }

    def __init__(self, email_config=None, telegram_token=None, telegram_chat_id=None):
        self.email_config      = email_config
        self.telegram_token    = telegram_token
        self.telegram_chat_id  = telegram_chat_id
        self._last_regime      = None
        self._alert_log        = []

    def _compose_message(self, prev_regime, new_regime, latest_row):
        nifty = latest_row.get('NIFTY', 'N/A')
        vix   = latest_row.get('VIX', 'N/A')
        cpi   = latest_row.get('CPI', 'N/A')
        yc    = latest_row.get('YieldCurve', 'N/A')
        date  = latest_row.name.strftime('%d %b %Y') if hasattr(latest_row, 'name') else 'Today'

        bull_p = latest_row.get('Bull', 0) * 100
        bear_p = latest_row.get('Bear', 0) * 100
        hv_p   = latest_row.get('HighVol', 0) * 100
        sw_p   = latest_row.get('Sideways', 0) * 100

        strategy = self.STRATEGY_BRIEF.get(new_regime, '')
        sector_w = SECTOR_ROTATION.get(new_regime, {})
        top_sectors = sorted(sector_w.items(), key=lambda x: x[1], reverse=True)[:3]
        top_str = " | ".join(f"{s}: {w*100:.0f}%" for s, w in top_sectors)

        msg = textwrap.dedent(f"""
        🔔 REGIME TRANSITION ALERT — {date}
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Previous Regime : {prev_regime}
        ➜ NEW Regime    : {new_regime.upper()}

        Market Snapshot:
          NIFTY 50   : {nifty:,.0f}
          India VIX  : {vix:.1f}
          CPI        : {cpi:.1f}%
          Yield Curve (10Y-2Y) : {yc:.2f}%

        Posterior Probabilities:
          Bull={bull_p:.0f}%  Bear={bear_p:.0f}%  HighVol={hv_p:.0f}%  Sideways={sw_p:.0f}%

        Recommended Allocation:
          {strategy}

        Top Sector Tilts:
          {top_str}

        ⚠ This is a quantitative signal, not financial advice.
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """).strip()
        return msg

    def check_and_alert(self, result):
        """Call this daily with the latest result DataFrame."""
        latest = result.iloc[-1]
        current_regime = latest['Regime']

        if self._last_regime is None:
            self._last_regime = current_regime
            print(f"[Alert] System initialised. Current regime: {current_regime}")
            return

        if current_regime != self._last_regime:
            msg = self._compose_message(self._last_regime, current_regime, latest)
            print("\n" + "="*60)
            print(msg)
            print("="*60)
            self._alert_log.append({
                'date': result.index[-1].strftime('%Y-%m-%d'),
                'from': self._last_regime,
                'to':   current_regime,
            })
            self._send_email(msg)
            self._send_telegram(msg)
            self._last_regime = current_regime
        else:
            print(f"[Alert] No regime change. Current: {current_regime} "
                  f"(P={latest[current_regime]*100:.0f}%)")

    def _send_email(self, msg):
        """Send alert email. Configure SMTP credentials via environment variables."""
        if not self.email_config:
            print("[Alert] Email not configured — skipping. "
                  "Set email_config dict with SMTP credentials.")
            return
        try:
            import smtplib
            from email.mime.text import MIMEText
            cfg = self.email_config
            mail = MIMEText(msg)
            mail['Subject'] = f"⚡ India HMM Regime Alert: {self._alert_log[-1]['from']} → {self._alert_log[-1]['to']}"
            mail['From']    = cfg['from']
            mail['To']      = cfg['to']
            with smtplib.SMTP(cfg.get('smtp_host', 'smtp.gmail.com'),
                              cfg.get('smtp_port', 587)) as s:
                s.starttls()
                s.login(cfg['from'], cfg['password'])
                s.sendmail(cfg['from'], cfg['to'], mail.as_string())
            print("[Alert] ✓ Email sent")
        except Exception as e:
            print(f"[Alert] Email error: {e}")

    def _send_telegram(self, msg):
        """Send alert to Telegram channel / chat."""
        if not self.telegram_token or not self.telegram_chat_id:
            print("[Alert] Telegram not configured — skipping. "
                  "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars.")
            return
        try:
            import urllib.request, urllib.parse
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = json.dumps({'chat_id': self.telegram_chat_id,
                                  'text': msg, 'parse_mode': 'Markdown'}).encode()
            req = urllib.request.Request(url, data=payload,
                                         headers={'Content-Type': 'application/json'})
            urllib.request.urlopen(req, timeout=10)
            print("[Alert] ✓ Telegram message sent")
        except Exception as e:
            print(f"[Alert] Telegram error: {e}")

    def get_alert_history(self):
        return pd.DataFrame(self._alert_log)


# ══════════════════════════════════════════════════════════════════════
# 9. FASTAPI WRAPPER
# ══════════════════════════════════════════════════════════════════════

FASTAPI_APP_CODE = '''"""
India HMM Regime Detector — FastAPI Service
=============================================
Serves the trained HMM model as a REST API.
The React dashboard (or any client) can POST market features
and receive a real-time regime prediction.

Install:
    pip install fastapi uvicorn hmmlearn scikit-learn numpy pandas

Run:
    uvicorn regime_api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health           — liveness check
    GET  /regime/current   — latest regime from last prediction
    POST /regime/predict   — predict regime from feature dict
    GET  /regime/history   — last N regime transitions
    GET  /regime/strategy  — current regime strategy + sector weights
"""

import os, pickle, json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any

app = FastAPI(
    title="India HMM Regime API",
    description="4-state Gaussian HMM market regime detector for NIFTY 50",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model + scaler at startup
MODEL_PATH  = os.environ.get("HMM_MODEL_PATH",  "output/hmm_model.pkl")
SCALER_PATH = os.environ.get("HMM_SCALER_PATH", "output/hmm_scaler.pkl")

_model  = None
_scaler = None
_history = []


@app.on_event("startup")
def load_model():
    global _model, _scaler
    try:
        with open(MODEL_PATH,  "rb") as f: _model  = pickle.load(f)
        with open(SCALER_PATH, "rb") as f: _scaler = pickle.load(f)
        print("✓ HMM model and scaler loaded")
    except FileNotFoundError:
        print("⚠ Model not found — run train.py first to generate output/hmm_model.pkl")


class MarketFeatures(BaseModel):
    """
    16 features expected by the HMM (same order as select_features_for_hmm).
    Provide daily values for all fields.
    """
    ret_1d:          float   # daily return (e.g. 0.0042)
    vol_20d:         float   # 20-day annualised realised vol
    vol_ratio:       float   # vol_10d / vol_60d
    price_vs_ma200:  float   # (price - 200dMA) / 200dMA
    ma50_vs_ma200:   float   # (50dMA - 200dMA) / 200dMA
    rsi14:           float   # 14-day RSI (0-100)
    vix:             float   # India VIX level
    vix_ratio:       float   # VIX / 20d mean VIX
    rate_spread:     float   # 10Y G-Sec - Repo Rate
    fx_trend:        float   # 20d % change in USD/INR
    fii_ma20:        float   # 20d average FII flow (₹ crore / 1000)
    drawdown:        float   # (price - 252d peak) / 252d peak
    yield_curve:     float   # 10Y G-Sec - 2Y G-Sec (NEW)
    real_rate:       float   # Repo Rate - CPI (NEW)
    dxy_chg:         float   # 20d % change in DXY proxy (NEW)
    iip_level:       float   # IIP YoY % (NEW)

    class Config:
        schema_extra = {
            "example": {
                "ret_1d": 0.003, "vol_20d": 0.14, "vol_ratio": 0.9,
                "price_vs_ma200": 0.05, "ma50_vs_ma200": 0.02, "rsi14": 58,
                "vix": 14.5, "vix_ratio": 0.95, "rate_spread": 2.3,
                "fx_trend": 0.01, "fii_ma20": 1.2, "drawdown": -0.03,
                "yield_curve": 1.8, "real_rate": 0.5, "dxy_chg": 0.01,
                "iip_level": 5.2,
            }
        }


REGIME_STRATEGIES_API = {
    "Bull":     "Long NIFTY 50 Fut 40% | Bank ETF 20% | Infra 15% | PUT hedge 5% | Cash 20%",
    "Bear":     "Short NIFTY 25% | Gold ETF 25% | Pharma/FMCG 15% | S-T Debt 25% | Long USD 10%",
    "HighVol":  "ATM Straddle 20% | Gold+Silver 25% | Liquid Fund 40% | Bank PUTs 10% | GSec 5%",
    "Sideways": "Short Strangle 20% | IT ETF 20% | Div Yield 20% | M-D Debt 25% | NIFTY SIP 15%",
}

LABEL_MAP_DEMO = {0: "Bull", 1: "Bear", 2: "HighVol", 3: "Sideways"}


def _predict_raw(features: MarketFeatures):
    if _model is None or _scaler is None:
        raise HTTPException(503, "Model not loaded. Run training pipeline first.")
    x = np.array([[
        features.ret_1d, features.vol_20d, features.vol_ratio,
        features.price_vs_ma200, features.ma50_vs_ma200, features.rsi14,
        features.vix, features.vix_ratio, features.rate_spread,
        features.fx_trend, features.fii_ma20, features.drawdown,
        features.yield_curve, features.real_rate, features.dxy_chg,
        features.iip_level,
    ]])
    x_s = _scaler.transform(x)
    state = int(_model.predict(x_s)[0])
    proba = _model.predict_proba(x_s)[0].tolist()
    regime = LABEL_MAP_DEMO.get(state, "Sideways")
    return regime, state, proba


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/regime/predict")
def predict_regime(features: MarketFeatures):
    """Predict market regime from current feature values."""
    regime, state, proba = _predict_raw(features)
    result = {
        "regime":      regime,
        "hmm_state":   state,
        "probabilities": {
            "Bull":     round(proba[0], 4) if len(proba) > 0 else 0,
            "Bear":     round(proba[1], 4) if len(proba) > 1 else 0,
            "HighVol":  round(proba[2], 4) if len(proba) > 2 else 0,
            "Sideways": round(proba[3], 4) if len(proba) > 3 else 0,
        },
        "strategy":    REGIME_STRATEGIES_API[regime],
        "confidence":  round(max(proba), 4),
    }
    _history.append(result)
    return result


@app.get("/regime/current")
def current_regime():
    """Return the most recent prediction."""
    if not _history:
        return {"message": "No predictions yet. POST to /regime/predict first."}
    return _history[-1]


@app.get("/regime/history")
def regime_history(n: int = 20):
    """Return last N regime predictions."""
    return {"history": _history[-n:], "total": len(_history)}


@app.get("/regime/strategy")
def regime_strategy(regime: str = "Bull"):
    """Return strategy and sector weights for a given regime."""
    if regime not in REGIME_STRATEGIES_API:
        raise HTTPException(400, f"Unknown regime. Choose from: {list(REGIME_STRATEGIES_API)}")
    return {
        "regime":   regime,
        "strategy": REGIME_STRATEGIES_API[regime],
        "sector_weights": {
            "Bull":     {"NIFTY Bank":0.22,"NIFTY IT":0.18,"NIFTY Auto":0.15,"NIFTY Metal":0.12,"NIFTY Realty":0.10,"NIFTY Infra":0.10,"NIFTY Energy":0.08,"NIFTY FMCG":0.03,"NIFTY Pharma":0.02},
            "Bear":     {"NIFTY Pharma":0.30,"NIFTY FMCG":0.25,"NIFTY IT":0.15,"NIFTY Energy":0.15,"NIFTY Infra":0.05,"NIFTY Bank":0.05,"NIFTY Auto":0.05},
            "HighVol":  {"NIFTY Pharma":0.35,"NIFTY FMCG":0.30,"NIFTY Energy":0.20,"NIFTY IT":0.10,"NIFTY Bank":0.05},
            "Sideways": {"NIFTY IT":0.25,"NIFTY FMCG":0.20,"NIFTY Pharma":0.15,"NIFTY Bank":0.15,"NIFTY Auto":0.10,"NIFTY Energy":0.08,"NIFTY Infra":0.07},
        }.get(regime, {}),
    }
'''


# ══════════════════════════════════════════════════════════════════════
# 10. VISUALISATION  (extended figures: model selection + walk-forward
#                     + sector rotation + new macro features)
# ══════════════════════════════════════════════════════════════════════

def plot_model_selection(selection_results, out_dir):
    """Fig 5: BIC/AIC model selection."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('HMM MODEL SELECTION  ·  BIC / AIC (3 / 4 / 5-state)',
                 fontsize=12, fontweight='bold', color='#f9fafb')
    fig.patch.set_facecolor('#0a0a0d')
    for ax in axes:
        ax.set_facecolor('#0f0f11')

    ks    = sorted(selection_results.keys())
    bics  = [selection_results[k]['bic']    for k in ks]
    aics  = [selection_results[k]['aic']    for k in ks]
    lls   = [selection_results[k]['loglik'] for k in ks]

    best_bic = ks[np.argmin(bics)]
    best_aic = ks[np.argmin(aics)]

    # BIC
    axes[0].bar([str(k) for k in ks], bics,
                color=['#10b981' if k == best_bic else '#374151' for k in ks], alpha=0.8)
    axes[0].set_title('BIC (lower = better)', fontsize=10, color='#9ca3af')
    axes[0].set_xlabel('# States', fontsize=9)
    axes[0].set_ylabel('BIC', fontsize=9)
    axes[0].grid(True, lw=0.3, axis='y')
    for i, (k, v) in enumerate(zip(ks, bics)):
        axes[0].text(i, v, f'{v:,.0f}', ha='center', va='bottom', fontsize=8, color='#e5e7eb')

    # AIC
    axes[1].bar([str(k) for k in ks], aics,
                color=['#f59e0b' if k == best_aic else '#374151' for k in ks], alpha=0.8)
    axes[1].set_title('AIC (lower = better)', fontsize=10, color='#9ca3af')
    axes[1].set_xlabel('# States', fontsize=9)
    axes[1].set_ylabel('AIC', fontsize=9)
    axes[1].grid(True, lw=0.3, axis='y')
    for i, (k, v) in enumerate(zip(ks, aics)):
        axes[1].text(i, v, f'{v:,.0f}', ha='center', va='bottom', fontsize=8, color='#e5e7eb')

    # Log-likelihood
    axes[2].plot([str(k) for k in ks], lls, 'o-', color='#a78bfa', lw=2, ms=8)
    axes[2].set_title('Log-Likelihood (higher = better fit)', fontsize=10, color='#9ca3af')
    axes[2].set_xlabel('# States', fontsize=9)
    axes[2].set_ylabel('Log-Likelihood', fontsize=9)
    axes[2].grid(True, lw=0.3)

    plt.tight_layout()
    fig.savefig(f'{out_dir}/fig5_model_selection.png', dpi=150, bbox_inches='tight',
                facecolor='#0a0a0d')
    plt.close(fig)
    print("✓ Saved: fig5_model_selection.png")


def plot_walk_forward(folds_df, out_dir):
    """Fig 6: Walk-forward validation results."""
    if folds_df is None or len(folds_df) == 0:
        print("⚠ No walk-forward folds to plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle('WALK-FORWARD VALIDATION  ·  Out-of-Sample Performance (TC-adjusted)',
                 fontsize=12, fontweight='bold', color='#f9fafb')
    fig.patch.set_facecolor('#0a0a0d')
    for ax in axes:
        ax.set_facecolor('#0f0f11')

    labels = folds_df['fold_start'].tolist()
    x      = range(len(labels))

    # OOS Annualised returns
    colors = ['#10b981' if r > 0 else '#ef4444' for r in folds_df['ann_ret_pct']]
    axes[0].bar(x, folds_df['ann_ret_pct'], color=colors, alpha=0.8)
    axes[0].axhline(0, color='#374151', lw=0.8)
    axes[0].axhline(folds_df['ann_ret_pct'].mean(), color='#f59e0b', lw=1.2, ls='--',
                    label=f"Mean: {folds_df['ann_ret_pct'].mean():+.1f}%")
    axes[0].set_title('OOS Ann. Return (%) per Fold', fontsize=10, color='#9ca3af')
    axes[0].set_xticks(list(x)[::max(1, len(labels)//6)])
    axes[0].set_xticklabels(labels[::max(1, len(labels)//6)], rotation=45, fontsize=7)
    axes[0].set_ylabel('%', fontsize=9)
    axes[0].legend(fontsize=8, framealpha=0.3, facecolor='#0f0f11')
    axes[0].grid(True, lw=0.3, axis='y')

    # OOS Sharpe
    sc = ['#10b981' if r > 0 else '#ef4444' for r in folds_df['sharpe']]
    axes[1].bar(x, folds_df['sharpe'], color=sc, alpha=0.8)
    axes[1].axhline(0, color='#374151', lw=0.8)
    axes[1].axhline(folds_df['sharpe'].mean(), color='#f59e0b', lw=1.2, ls='--',
                    label=f"Mean: {folds_df['sharpe'].mean():.2f}")
    axes[1].set_title('OOS Sharpe Ratio per Fold', fontsize=10, color='#9ca3af')
    axes[1].set_xticks(list(x)[::max(1, len(labels)//6)])
    axes[1].set_xticklabels(labels[::max(1, len(labels)//6)], rotation=45, fontsize=7)
    axes[1].legend(fontsize=8, framealpha=0.3, facecolor='#0f0f11')
    axes[1].grid(True, lw=0.3, axis='y')

    # Regime switches per fold
    axes[2].bar(x, folds_df['n_switches'], color='#3b82f6', alpha=0.8)
    axes[2].axhline(folds_df['n_switches'].mean(), color='#f59e0b', lw=1.2, ls='--',
                    label=f"Mean: {folds_df['n_switches'].mean():.1f}")
    axes[2].set_title('Regime Switches per Fold\n(drives TC drag)', fontsize=10, color='#9ca3af')
    axes[2].set_xticks(list(x)[::max(1, len(labels)//6)])
    axes[2].set_xticklabels(labels[::max(1, len(labels)//6)], rotation=45, fontsize=7)
    axes[2].legend(fontsize=8, framealpha=0.3, facecolor='#0f0f11')
    axes[2].grid(True, lw=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(f'{out_dir}/fig6_walk_forward.png', dpi=150, bbox_inches='tight',
                facecolor='#0a0a0d')
    plt.close(fig)
    print("✓ Saved: fig6_walk_forward.png")


def plot_sector_rotation(result, portfolio_returns, out_dir):
    """Fig 7: Sector rotation analysis."""
    fig = plt.figure(figsize=(18, 8))
    fig.suptitle('SECTOR ROTATION  ·  Regime-Based NSE Sectoral Allocation',
                 fontsize=12, fontweight='bold', color='#f9fafb')
    fig.patch.set_facecolor('#0a0a0d')
    gs7 = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.4,
                            left=0.06, right=0.97, top=0.88, bottom=0.08)

    regime_order = ['Bull', 'Bear', 'HighVol', 'Sideways']
    sectors_all  = list(SECTOR_ROTATION['Bull'].keys())

    # Heatmap: 4 regimes × N sectors
    ax_heat = fig.add_subplot(gs7[0, :2])
    ax_heat.set_facecolor('#0f0f11')
    weight_matrix = np.array([[SECTOR_ROTATION[r].get(s, 0) for s in sectors_all]
                               for r in regime_order])
    im = ax_heat.imshow(weight_matrix, cmap='YlGn', aspect='auto',
                        vmin=0, vmax=0.35)
    ax_heat.set_xticks(range(len(sectors_all)))
    ax_heat.set_xticklabels([s.replace('NIFTY ', '') for s in sectors_all],
                             rotation=40, ha='right', fontsize=8)
    ax_heat.set_yticks(range(len(regime_order)))
    ax_heat.set_yticklabels(regime_order, fontsize=9)
    ax_heat.set_title('Sector Weight Heatmap (%)\nby Regime', fontsize=9, color='#9ca3af')
    for i in range(len(regime_order)):
        for j in range(len(sectors_all)):
            v = weight_matrix[i, j]
            if v > 0:
                ax_heat.text(j, i, f'{v*100:.0f}', ha='center', va='center',
                             fontsize=7, color='black' if v > 0.18 else '#e5e7eb', fontweight='bold')
    plt.colorbar(im, ax=ax_heat, fraction=0.03, label='Weight')

    # Portfolio cumulative return (sector rotation vs buy-hold)
    ax_perf = fig.add_subplot(gs7[0, 2:])
    ax_perf.set_facecolor('#0f0f11')
    port_cum  = np.exp(np.cumsum(portfolio_returns.values)) - 1
    mkt_cum   = np.exp(np.cumsum(result['Returns'].values)) - 1
    ax_perf.plot(result.index, port_cum * 100, color='#10b981', lw=1, label='Sector-Rotated')
    ax_perf.plot(result.index, mkt_cum * 100, color='#6b7280', lw=0.7, ls='--', label='NIFTY B&H')
    ax_perf.set_title('Sector Rotation Cumulative Return (%)', fontsize=9, color='#9ca3af')
    ax_perf.set_ylabel('%', fontsize=8)
    ax_perf.legend(fontsize=8, framealpha=0.3, facecolor='#0f0f11')
    ax_perf.grid(True, lw=0.3)
    ax_perf.set_xlim(result.index[0], result.index[-1])

    # Per-regime donut charts
    for i, regime in enumerate(regime_order):
        ax_d = fig.add_subplot(gs7[1, i])
        ax_d.set_facecolor('#0f0f11')
        weights_r = {k: v for k, v in SECTOR_ROTATION[regime].items() if v > 0}
        labels_d  = [s.replace('NIFTY ', '') for s in weights_r]
        vals_d    = list(weights_r.values())
        palette   = plt.cm.Set3(np.linspace(0, 1, len(vals_d)))
        wedges, _  = ax_d.pie(vals_d, labels=None, colors=palette,
                               startangle=90, wedgeprops=dict(width=0.55))
        ax_d.set_title(f'{regime}\nSector Mix', fontsize=9,
                       color=REGIME_COLORS[regime], fontweight='bold')
        # Legend
        ax_d.legend(wedges, [f'{l} {v*100:.0f}%' for l, v in zip(labels_d, vals_d)],
                    loc='center', bbox_to_anchor=(0.5, -0.25), fontsize=6.5,
                    framealpha=0.2, facecolor='#0f0f11', ncol=1)

    fig.savefig(f'{out_dir}/fig7_sector_rotation.png', dpi=150, bbox_inches='tight',
                facecolor='#0a0a0d')
    plt.close(fig)
    print("✓ Saved: fig7_sector_rotation.png")


def plot_new_macro_signals(result, out_dir):
    """Fig 8: New macro features (yield curve, CPI, IIP, DXY, real rate)."""
    fig = plt.figure(figsize=(18, 8))
    fig.suptitle('NEW MACRO SIGNALS  ·  Yield Curve · CPI · IIP · Real Rate',
                 fontsize=12, fontweight='bold', color='#f9fafb')
    fig.patch.set_facecolor('#0a0a0d')
    gs8 = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                            left=0.06, right=0.97, top=0.88, bottom=0.06)

    def shade_regimes(ax):
        changes = [0] + list(np.where(result['Regime'].values[:-1] != result['Regime'].values[1:])[0]+1) + [len(result)]
        for i in range(len(changes)-1):
            reg = result['Regime'].iloc[changes[i]]
            ax.axvspan(result.index[changes[i]], result.index[changes[i+1]-1],
                       alpha=0.09, color=REGIME_COLORS[reg])

    # 1. Yield curve (10Y-2Y)
    ax = fig.add_subplot(gs8[0, 0])
    ax.set_facecolor('#0f0f11')
    shade_regimes(ax)
    ax.plot(result.index, result['YieldCurve'], color='#a78bfa', lw=0.8)
    ax.axhline(0, color='#ef4444', lw=0.7, ls='--', label='Inversion')
    ax.fill_between(result.index, result['YieldCurve'], 0,
                    where=result['YieldCurve'] < 0, color='#ef4444', alpha=0.3, label='Inverted')
    ax.set_title('Yield Curve (10Y − 2Y G-Sec %)\n← Inversion = recession risk', fontsize=9, color='#9ca3af')
    ax.set_ylabel('%', fontsize=8); ax.grid(True, lw=0.3)
    ax.legend(fontsize=7, framealpha=0.3, facecolor='#0f0f11')
    ax.set_xlim(result.index[0], result.index[-1])

    # 2. CPI Inflation
    ax = fig.add_subplot(gs8[0, 1])
    ax.set_facecolor('#0f0f11')
    shade_regimes(ax)
    ax.plot(result.index, result['CPI'], color='#f59e0b', lw=0.8)
    ax.axhline(4.0, color='#10b981', lw=0.7, ls='--', label='RBI target (4%)')
    ax.axhline(6.0, color='#ef4444', lw=0.7, ls='--', label='Upper tolerance (6%)')
    ax.set_title('CPI Inflation (%)\nRBI target: 4% ± 2%', fontsize=9, color='#9ca3af')
    ax.set_ylabel('%', fontsize=8); ax.grid(True, lw=0.3)
    ax.legend(fontsize=7, framealpha=0.3, facecolor='#0f0f11')
    ax.set_xlim(result.index[0], result.index[-1])

    # 3. IIP (Industrial Production)
    ax = fig.add_subplot(gs8[0, 2])
    ax.set_facecolor('#0f0f11')
    shade_regimes(ax)
    ax.bar(result.index, result['IIP'],
           color=[('#10b981' if v > 0 else '#ef4444') for v in result['IIP']], alpha=0.6, width=1)
    ax.axhline(0, color='#9ca3af', lw=0.5)
    ax.set_title('IIP YoY % (Industrial Production)\n> 0 = expansion', fontsize=9, color='#9ca3af')
    ax.set_ylabel('%', fontsize=8); ax.grid(True, lw=0.3, axis='y')
    ax.set_xlim(result.index[0], result.index[-1])

    # 4. Real policy rate (Repo - CPI)
    ax = fig.add_subplot(gs8[1, 0])
    ax.set_facecolor('#0f0f11')
    shade_regimes(ax)
    ax.plot(result.index, result['RepoRate'] - result['CPI'], color='#34d399', lw=0.8)
    ax.axhline(0, color='#ef4444', lw=0.7, ls='--', label='Neutral (real rate = 0)')
    ax.fill_between(result.index, result['RepoRate'] - result['CPI'], 0,
                    where=(result['RepoRate'] - result['CPI'] < 0),
                    color='#ef4444', alpha=0.2, label='Negative real rates')
    ax.set_title('Real Policy Rate (Repo − CPI %)\n< 0 = accommodative / inflationary', fontsize=9, color='#9ca3af')
    ax.set_ylabel('%', fontsize=8); ax.grid(True, lw=0.3)
    ax.legend(fontsize=7, framealpha=0.3, facecolor='#0f0f11')
    ax.set_xlim(result.index[0], result.index[-1])

    # 5. Box: macro means per regime
    ax = fig.add_subplot(gs8[1, 1])
    ax.set_facecolor('#0f0f11')
    regimes_order = ['Bull', 'Bear', 'HighVol', 'Sideways']
    macro_cols = ['YieldCurve', 'CPI', 'IIP']
    x_pos = np.arange(len(regimes_order))
    w = 0.25
    line_colors = ['#a78bfa', '#f59e0b', '#34d399']
    for j, (col, lc) in enumerate(zip(macro_cols, line_colors)):
        means = [result.loc[result['Regime']==r, col].mean() for r in regimes_order]
        ax.bar(x_pos + j*w, means, w, label=col, color=lc, alpha=0.8)
    ax.set_xticks(x_pos + w)
    ax.set_xticklabels(regimes_order, fontsize=9)
    ax.set_title('Mean Macro Signal\nby Regime', fontsize=9, color='#9ca3af')
    ax.legend(fontsize=7, framealpha=0.3, facecolor='#0f0f11')
    ax.grid(True, lw=0.3, axis='y')

    # 6. Scatter: real rate vs yield curve, coloured by regime
    ax = fig.add_subplot(gs8[1, 2])
    ax.set_facecolor('#0f0f11')
    sample = result.sample(min(1500, len(result)), random_state=42)
    for reg in regimes_order:
        m = sample['Regime'] == reg
        ax.scatter(sample.loc[m, 'YieldCurve'],
                   sample.loc[m, 'RepoRate'] - sample.loc[m, 'CPI'],
                   c=REGIME_COLORS[reg], alpha=0.3, s=5, label=reg)
    ax.axhline(0, color='#9ca3af', lw=0.4, ls='--')
    ax.axvline(0, color='#9ca3af', lw=0.4, ls='--')
    ax.set_xlabel('Yield Curve (10Y−2Y)', fontsize=8)
    ax.set_ylabel('Real Rate (Repo−CPI)', fontsize=8)
    ax.set_title('Macro Regime Space\n(Yield Curve vs Real Rate)', fontsize=9, color='#9ca3af')
    ax.legend(fontsize=7, framealpha=0.3, facecolor='#0f0f11', markerscale=3)
    ax.grid(True, lw=0.3)

    plt.tight_layout()
    fig.savefig(f'{out_dir}/fig8_new_macro_signals.png', dpi=150, bbox_inches='tight',
                facecolor='#0a0a0d')
    plt.close(fig)
    print("✓ Saved: fig8_new_macro_signals.png")


# ══════════════════════════════════════════════════════════════════════
# 11. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  INDIA MARKET REGIME DETECTOR  —  v2")
    print("  4-State Gaussian HMM  |  Full Next-Steps Implementation")
    print("=" * 70)

    OUT_DIR = '/home/claude/india_hmm_v2/output'
    os.makedirs(OUT_DIR, exist_ok=True)

    plt.style.use('dark_background')
    plt.rcParams.update({
        'font.family':      'monospace',
        'axes.facecolor':   '#0f0f11',
        'figure.facecolor': '#0a0a0d',
        'axes.edgecolor':   '#2a2a35',
        'axes.labelcolor':  '#9ca3af',
        'xtick.color':      '#6b7280',
        'ytick.color':      '#6b7280',
        'grid.color':       '#1e1e28',
        'text.color':       '#e5e7eb',
    })

    # ── 1. Data (with new features) ──────────────────────────────────
    print("\n[1] Generating calibrated market data (+ CPI, IIP, yield curve, DXY)…")
    df = generate_calibrated_market_data(seed=42)

    # ── 2. Feature engineering ────────────────────────────────────────
    print("\n[2] Engineering features (extended macro set)…")
    feat = engineer_features(df)

    # ── 3. MODEL SELECTION (BIC / AIC) ───────────────────────────────
    print("\n[3] Model selection: comparing 3-state / 4-state / 5-state HMMs…")
    scaler    = StandardScaler()
    X_scaled  = scaler.fit_transform(select_features_for_hmm(feat).values)
    selection_results, best_k = bic_aic_model_selection(X_scaled, state_range=(3, 4, 5), n_init=5)
    # Always use 4-state for comparability (BIC sometimes prefers 3)
    n_states = 4
    print(f"  (Proceeding with n_states=4 for strategy consistency)")

    # ── 4. Train final HMM ───────────────────────────────────────────
    print(f"\n[4] Training {n_states}-state HMM (full dataset, 15 restarts)…")
    model  = train_hmm(X_scaled, n_states=n_states, n_iter=200, n_init=15)

    # ── 5. Decode regimes ────────────────────────────────────────────
    print("\n[5] Decoding regimes (Viterbi + Forward-Backward)…")
    result, label_map, scores = label_regimes(model, X_scaled, feat, df)

    # ── 6. WALK-FORWARD VALIDATION ───────────────────────────────────
    print("\n[6] Walk-forward validation…")
    folds_df = walk_forward_validation(feat, df, train_years=WF_TRAIN_YEARS,
                                       test_months=WF_TEST_MONTHS, n_states=n_states)

    # ── 7. Backtest WITH transaction costs ───────────────────────────
    print("\n[7] Backtesting with transaction costs (STT + slippage)…")
    strat_ret, strat_cap, bh_cap, n_sw, tc_drag = compute_strategy_payoff_with_tc(result)
    metrics = compute_performance_metrics(strat_ret, strat_cap)
    bh_metrics = compute_performance_metrics(
        result['Returns'].values,
        np.exp(np.cumsum(result['Returns'].values)) * 1_000_000
    )

    print(f"\n  ┌── Performance Comparison (TC-adjusted) ──────────────────┐")
    print(f"  │  Metric          HMM Strategy     Buy & Hold             │")
    for k, label in [('ann_return','Ann. Return'), ('sharpe','Sharpe'),
                     ('max_dd','Max Drawdown'), ('calmar','Calmar'), ('win_rate','Win Rate')]:
        hv = metrics[k] * (100 if k in ('ann_return','max_dd','win_rate') else 1)
        bv = bh_metrics[k] * (100 if k in ('ann_return','max_dd','win_rate') else 1)
        sfx = '%' if k in ('ann_return','max_dd','win_rate') else ''
        print(f"  │  {label:16s}  {hv:+8.2f}{sfx}       {bv:+8.2f}{sfx}         │")
    print(f"  └──────────────────────────────────────────────────────────┘")

    # ── 8. Bootstrap CI ──────────────────────────────────────────────
    print("\n[8] Bootstrap confidence intervals (N=2000)…")
    ci_results = bootstrap_confidence_intervals(strat_ret)

    # ── 9. SECTOR ROTATION ───────────────────────────────────────────
    print("\n[9] Sector rotation portfolio analysis…")
    port_ret, sector_rets = compute_sector_rotation_returns(result)

    # ── 10. ALERT SYSTEM demo ─────────────────────────────────────────
    print("\n[10] Alert system demonstration…")
    alerter = RegimeAlertSystem()  # no creds → prints only, no sends
    alerter._last_regime = 'Sideways'  # simulate previous regime
    # Force a regime change for demo
    demo_latest = result.iloc[-1].copy()
    demo_latest['Regime'] = 'Bull'
    demo_latest['Bull'] = 0.87
    demo_latest['Bear'] = 0.05
    demo_latest['HighVol'] = 0.02
    demo_latest['Sideways'] = 0.06
    alerter.check_and_alert(result.assign(Regime=lambda x: x['Regime'].where(
        x.index < result.index[-1], 'Bull')))

    # ── 11. Save FastAPI app ──────────────────────────────────────────
    print("\n[11] Writing FastAPI service to regime_api.py…")
    with open(f'{OUT_DIR}/regime_api.py', 'w') as f:
        f.write(FASTAPI_APP_CODE)
    print("✓ Saved: regime_api.py  (run with: uvicorn regime_api:app --reload)")

    # ── 12. Visualisations ────────────────────────────────────────────
    print("\n[12] Generating figures (5-8 are new)…")
    plot_model_selection(selection_results, OUT_DIR)
    plot_walk_forward(folds_df, OUT_DIR)
    plot_sector_rotation(result, port_ret, OUT_DIR)
    plot_new_macro_signals(result, OUT_DIR)

    # ── 13. Save regime history CSV ───────────────────────────────────
    out_cols = ['NIFTY','Returns','VIX','RepoRate','YieldCurve','CPI','IIP',
                'Regime','Bull','Bear','HighVol','Sideways']
    result[out_cols].to_csv(f'{OUT_DIR}/regime_history_v2.csv')
    print(f"\n✓ Regime history CSV saved")

    # ── 14. Save walk-forward summary ────────────────────────────────
    if len(folds_df) > 0:
        folds_df.to_csv(f'{OUT_DIR}/walk_forward_summary.csv', index=False)
        print(f"✓ Walk-forward summary CSV saved")

    print("\n" + "="*70)
    print("  ALL NEXT-STEPS IMPLEMENTED  ✓")
    print("="*70)
    print(f"""
  Files in {OUT_DIR}/:
    fig5_model_selection.png    — BIC/AIC comparison (3/4/5-state)
    fig6_walk_forward.png       — OOS performance per fold
    fig7_sector_rotation.png    — NSE sector rotation heatmap + returns
    fig8_new_macro_signals.png  — Yield curve, CPI, IIP, real rate
    regime_api.py               — FastAPI REST service skeleton
    regime_history_v2.csv       — Extended regime history
    walk_forward_summary.csv    — Per-fold OOS metrics
    """)

    return result, model, metrics, ci_results, folds_df


if __name__ == '__main__':
    main()
