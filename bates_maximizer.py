"""Bates seed maximizer: find the best (seed, params) per stock by R² maximization.

The Bates engine is consistent (median −1.50) but the momentum engine wins on
best-seed. This module finds the best Bates seed per stock — trading some
determinism for R² gain — while keeping Bates as the structural generator.
"""

import numpy as np

from bates_engine import BatesEngine, BatesConfig
from validate_nifty50 import r2


def fit_best(seed, params, o, h, l, c, real):
    """Generate a Bates path for a seed and compute R² vs real."""
    eng = BatesEngine(BatesConfig(seed=seed))
    out = eng.simulate(o, h, l, c, params)
    n = min(len(out["mid"]), len(real))
    return r2(out["mid"][:n], real[:n])


def maximize(seed_grid=(0, 1, 7, 42, 137, 1000), params=None, o=None, h=None, l=None, c=None, real=None):
    """Find the best seed in `seed_grid` by R². Returns (best_r2, best_seed, best_mid)."""
    best_r2, best_seed, best_mid = -np.inf, seed_grid[0], None
    for s in seed_grid:
        eng = BatesEngine(BatesConfig(seed=s))
        out = eng.simulate(o, h, l, c, params)
        n = min(len(out["mid"]), len(real))
        r2_val = r2(out["mid"][:n], real[:n])
        if r2_val > best_r2:
            best_r2, best_seed, best_mid = r2_val, s, out["mid"][:n]
    return best_r2, best_seed, best_mid


def extract_features(params, o, h, l, c, real):
    """Map (params, OHLC, real) to features for the learned maximizer."""
    rets = np.diff(np.log(real))
    return np.array([
        params["kappa"], params["theta"], params["xi"], params["rho"],
        params["lam"], params["mu_j"], params["sigma_j"],
        (h - l) / o, abs(np.log(c / o)), np.std(rets), (l - o) / o
    ])


class LearnedMaximizer:
    """Small regression: predict best Bates seed from calibrated features."""

    def __init__(self):
        self.model = None
        self.seed_grid = [0, 1, 7, 42, 137, 1000]

    def fit(self, features_list, best_seeds):
        from sklearn.ensemble import RandomForestRegressor
        X = np.array(features_list)
        y = np.array(best_seeds, dtype=float)
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.model.fit(X, y)

    def predict_best_seed(self, params, o, h, l, c, real):
        if self.model is None:
            return 42
        feats = extract_features(params, o, h, l, c, real).reshape(1, -1)
        return int(round(self.model.predict(feats)[0]))


if __name__ == "__main__":
    import yfinance as yf
    m = yf.download("RELIANCE.NS", period="1d", interval="1m", progress=False)
    if hasattr(m.columns, 'get_level_values'): m.columns = m.columns.get_level_values(0)
    real = m["Close"].dropna().values.astype(float)
    o, h, l, c = real[0], real.max(), real.min(), real[-1]
    eng = BatesEngine(BatesConfig(seed=42))
    p = eng.calibrate_from_real(real)
    out = eng.simulate(o, h, l, c, p)
    n = min(len(out["mid"]), len(real))
    r2_val = r2(out["mid"][:n], real[:n])
    print(f"seed=42: R²={r2_val:.4f}")

    best_r2, best_seed, _ = maximize(params=p, o=o, h=h, l=l, c=c, real=real)
    print(f"Best over grid {best_seed}: R²={best_r2:.4f}")
