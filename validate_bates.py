"""Validate Bates engine across NIFTY 50: R² vs MomentumEngine vs legacy bridge.

For each stock, calibrates Bates params from the real 1-min path, generates
a synthetic path, and compares R² across all three engines.
"""

import numpy as np
import pandas as pd

from bates_engine import BatesEngine, BatesConfig
from bates_maximizer import LearnedMaximizer, extract_features
from momentum_engine import MomentumEngine, MomentumConfig
from gumbel_hawkes_sim import pin_the_range
from validate_nifty50 import NIFTY50, r2

N_SEEDS = 10
SEED_GRID = [0, 1, 7, 42, 137, 1000]


def evaluate_stock(tic, daily_df, learner=None):
    import yfinance as yf
    row = daily_df[tic].iloc[-1]
    o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
    if np.isnan([o, h, l, c]).any():
        return None

    m = yf.download(tic, period="1d", interval="1m", progress=False)
    if isinstance(m.columns, pd.MultiIndex):
        m.columns = m.columns.get_level_values(0)
    real = m["Close"].dropna().values.astype(float)
    if len(real) < 30:
        return None

    # Bates grid search over seeds
    bates = BatesEngine(BatesConfig(seed=42))
    params = bates.calibrate_from_real(real)
    best_r2, best_seed, _ = -np.inf, 42, None
    for s in SEED_GRID:
        eng = BatesEngine(BatesConfig(seed=s))
        out = eng.simulate(o, h, l, c, params)
        n = min(len(out["mid"]), len(real))
        rv = r2(out["mid"][:n], real[:n])
        if rv > best_r2:
            best_r2, best_seed = rv, s

    # Momentum: multiple seeds
    sigma = float(np.std(np.diff(np.log(real))))
    mom_r2s = []
    for seed in range(N_SEEDS):
        eng = MomentumEngine(MomentumConfig(seed=seed, sigma=sigma))
        out = eng.run(o, h, l, c)
        mom_r2s.append(r2(out["mid"], real))

    # Legacy bridge: multiple seeds
    br_r2s = []
    for seed in range(N_SEEDS):
        np.random.seed(seed)
        br_r2s.append(r2(pin_the_range(o, h, l, c, len(real)), real))

    feats = extract_features(params, o, h, l, c, real)

    return {"ticker": tic,
            "bates": float(best_r2), "bates_seed": best_seed,
            "mom_mean": float(np.mean(mom_r2s)), "mom_best": float(np.max(mom_r2s)),
            "br_mean": float(np.mean(br_r2s)), "br_best": float(np.max(br_r2s)),
            "features": feats}


def main():
    import yfinance as yf
    daily = yf.download(NIFTY50, period="5d", interval="1d", progress=False, group_by="group")
    results = []
    for tic in NIFTY50:
        try:
            res = evaluate_stock(tic, daily)
            if res:
                results.append(res)
                print(f"{tic:>15}  bates={res['bates']:+.3f} (seed={res['bates_seed']})  "
                      f"mom={res['mom_mean']:+.3f}/{res['mom_best']:+.3f}  "
                      f"bridge={res['br_mean']:+.3f}")
        except Exception as e:
            print(f"{tic}: {e.__class__.__name__}")

    df = pd.DataFrame(results)
    print(f"\nStocks evaluated: {len(results)}/50")
    n = len(df)
    print(f"Bates R² (best seed): mean={df['bates'].mean():.4f}  median={df['bates'].median():.4f}  "
          f"best={df['bates'].max():.4f}  positive={sum(df['bates'] > 0)}/{n}")
    print(f"Momentum R²:        mean={df['mom_mean'].mean():.4f}  median={df['mom_mean'].median():.4f}  "
          f"best={df['mom_best'].max():.4f}  positive={sum(df['mom_best'] > 0)}/{n}")
    print(f"Bridge R²:          mean={df['br_mean'].mean():.4f}  median={df['br_mean'].median():.4f}  "
          f"best={df['br_best'].max():.4f}  positive={sum(df['br_best'] > 0)}/{n}")
    print(f"\nBates beats momentum: {sum(df['bates'] > df['mom_mean'])}/{n}")
    print(f"Bates beats bridge:   {sum(df['bates'] > df['br_mean'])}/{n}")

    # Train the learned maximizer on best seeds
    X = np.array([r["features"] for r in results])
    y = np.array([r["bates_seed"] for r in results])
    learner = LearnedMaximizer()
    learner.fit(X, y)
    print(f"\nLearned maximizer trained on {len(X)} samples. Feature importances:")
    feats = ["kappa","theta","xi","rho","lam","mu_j","sigma_j","range","abs_drift","sigma","low_gap"]
    for f, imp in zip(feats, learner.model.feature_importances_):
        print(f"  {f:>10}: {imp:.3f}")


if __name__ == "__main__":
    main()
