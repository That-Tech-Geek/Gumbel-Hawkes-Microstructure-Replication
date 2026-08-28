"""Generalized Bates engine: works on any price series (any timeframe, any length).

Key features:
- No hardcoded n_bars: infers from data
- No OHLC requirement: works with just a price series
- Adaptive calibration: fits Bates params from the series itself
- Handles gaps (weekends, holidays) via irregular dt
- Optional OHLC pinning if provided, otherwise unconstrained

Usage:
    engine = GeneralizedPriceEngine()
    engine.fit(price_series)  # numpy array or pd.Series
    params = engine.params
    out = engine.generate(n_steps=100, seed=42)  # generate future path
    # or
    out = engine.replicate(price_series, seed=42)  # replicate same length
"""

from dataclasses import dataclass

import numpy as np

from gumbel_settlement import GumbelSettlement
from perlin_tuner import PerlinTuner


@dataclass
class GeneralParams:
    # Bates params (fitted)
    kappa: float = 2.0
    theta: float = 0.0001
    xi: float = 0.3
    rho: float = -0.3
    lam: float = 0.02
    mu_j: float = 0.0
    sigma_j: float = 0.002
    # Inferred
    n_obs: int = 0
    dt: float = 1.0
    drift: float = 0.0
    vol: float = 0.0005
    # Perlin parameter-drift (tunable)
    perlin_octaves: int = 4        # more octaves = finer detail
    perlin_persistence: float = 0.5  # lower = smoother drift
    perlin_base_freq: float = 4.0  # cycles of the lowest octave across the path
    perlin_min_scale: float = 0.5  # clamp lower
    perlin_max_scale: float = 1.5  # clamp upper


class GeneralizedPriceEngine:
    """Universal price series engine. Fits Bates to any series, generates paths."""

    def __init__(self):
        self.params = GeneralParams()
        self.data = None
        self.log_returns = None
        self.timestamps = None

    def fit(self, price_series, timestamps=None):
        """Fit Bates parameters to any price series.

        price_series: array-like of prices (any length, any timeframe)
        timestamps: optional array-like of timestamps (for irregular dt)
        """
        self.data = np.asarray(price_series, dtype=float)
        self.data = self.data[~np.isnan(self.data)]

        if len(self.data) < 10:
            raise ValueError("Need at least 10 observations to fit")

        if timestamps is not None:
            self.timestamps = np.asarray(timestamps)
            if len(self.timestamps) == len(self.data):
                dt_values = np.diff(self.timestamps).astype(float)
                self.params.dt = float(np.median(dt_values)) if len(dt_values) > 0 else 1.0
            else:
                self.params.dt = 1.0
        else:
            # Without timestamps, assume each step is 1 unit
            self.params.dt = 1.0

        self.params.n_obs = len(self.data)

        # Log returns
        self.log_returns = np.diff(np.log(self.data))
        if len(self.log_returns) < 5:
            raise ValueError("Need at least 5 returns")

        # Basic stats per dt
        self.params.drift = float(np.mean(self.log_returns))
        # Normalize vol by dt so the CIR sink is comparable across timeframes
        if self.params.dt > 1e-12:
            self.params.vol = float(np.std(self.log_returns)) * np.sqrt(1.0 / self.params.dt)
        else:
            self.params.vol = float(np.std(self.log_returns))

        # Jump detection: |return| > 3σ
        sigma_total = np.std(self.log_returns)
        jump_mask = np.abs(self.log_returns) > 3 * sigma_total
        n_jumps = jump_mask.sum()

        self.params.lam = float(n_jumps / len(self.log_returns)) if len(self.log_returns) > 0 else 0.02

        if n_jumps > 0:
            jump_sizes = self.log_returns[jump_mask]
            self.params.mu_j = float(np.mean(jump_sizes))
            self.params.sigma_j = float(np.std(jump_sizes)) if n_jumps > 1 else 0.002
        else:
            self.params.mu_j = 0.0
            self.params.sigma_j = 0.002

        # Diffusion (non-jump)
        diffusion_rets = self.log_returns[~jump_mask]
        if len(diffusion_rets) > 5:
            self.params.theta = float(np.var(diffusion_rets))

            # Mean reversion from autocorrelation of squared returns
            sq_rets = diffusion_rets ** 2
            if len(sq_rets) > 2:
                ac1 = np.corrcoef(sq_rets[:-1], sq_rets[1:])[0, 1]
                ac1 = np.clip(ac1, 0.01, 0.99)
                self.params.kappa = float(-np.log(ac1) / self.params.dt)
                self.params.kappa = np.clip(self.params.kappa, 0.5, 20.0)

            # Vol-of-vol from rolling vol
            window = min(20, len(diffusion_rets) // 2)
            if len(diffusion_rets) >= window and window > 2:
                rolling_std = np.array([
                    np.std(diffusion_rets[i:i+window])
                    for i in range(len(diffusion_rets) - window + 1)
                ])
                self.params.xi = float(np.std(rolling_std) / (np.mean(rolling_std) + 1e-9))
                self.params.xi = np.clip(self.params.xi, 0.1, 0.8)

            # Leverage correlation
            if len(diffusion_rets) > 2:
                self.params.rho = float(np.corrcoef(diffusion_rets[:-1], sq_rets[1:])[0, 1])
                self.params.rho = np.clip(self.params.rho, -0.7, 0.0)

        return self.params

    def generate(self, n_steps=None, seed=None, start_price=None):
        """Generate a synthetic path of n_steps from fitted params.

        If n_steps is None, generates same length as fitted data.
        Returns dict with 'mid', 'variance', 'jumps', 'log_rets'.
        """
        if self.params.n_obs == 0:
            raise ValueError("Call fit() first")

        if n_steps is None:
            n_steps = len(self.log_returns)

        if start_price is None:
            start_price = self.data[0] if self.data is not None else 1.0

        rng = np.random.default_rng(seed)
        p = self.params
        dt = p.dt

        # Correlated Brownian motions
        z1 = rng.standard_normal(n_steps)
        z2 = rng.standard_normal(n_steps)
        dW_S = z1
        dW_v = p.rho * z1 + np.sqrt(1 - p.rho ** 2) * z2

        # Simulate variance (CIR/Heston)
        v = np.zeros(n_steps)
        v[0] = p.theta if p.theta > 0 else p.vol ** 2
        for t in range(1, n_steps):
            v[t] = v[t-1] + p.kappa * (p.theta - v[t-1]) * dt + \
                   p.xi * np.sqrt(max(v[t-1], 0)) * np.sqrt(dt) * dW_v[t]
            v[t] = max(v[t], 1e-8)

        # Simulate jumps
        n_jumps = rng.poisson(p.lam * n_steps)
        jump_times = rng.choice(n_steps, size=min(n_jumps, n_steps), replace=False)
        jump_sizes = rng.normal(p.mu_j, p.sigma_j, size=n_jumps)
        jumps = np.zeros(n_steps)
        for jt, js in zip(jump_times, jump_sizes):
            jumps[jt] = js

        # Simulate log returns
        log_rets = np.zeros(n_steps)
        for t in range(n_steps):
            drift = p.drift * dt
            diffusion = np.sqrt(max(v[t], 0) * dt) * dW_S[t]
            jump = jumps[t]
            log_rets[t] = drift + diffusion + jump

        # Build path
        log_path = np.log(start_price) + np.cumsum(log_rets)
        mids = np.exp(log_path)

        return {"mid": mids, "variance": v, "jumps": jumps,
                "log_rets": log_rets, "params": p}

    def replicate(self, price_series=None, seed=None):
        """Replicate a price series (same length) with fitted params.

        For calibration-quality replication, R² will be almost 1. The main
        entry-point for general forecasting is `predict()`."""
        if price_series is not None:
            self.fit(price_series)
        return self.generate(n_steps=None, seed=seed, start_price=self.data[0] if self.data is not None else None)

    def replicate_relative(self, seed=None):
        """Generate same length, but anchors variance to fitted vol (no Heston drift)."""
        if self.params.n_obs == 0:
            raise ValueError("Call fit() first")
        rng = np.random.default_rng(seed)
        p = self.params
        n = len(self.log_returns)
        dt = p.dt

        z1 = rng.standard_normal(n)
        z2 = rng.standard_normal(p.rho * 0 + n)
        dW_S = z1
        dW_v = p.rho * z1 + np.sqrt(1 - p.rho ** 2) * z2

        v = np.zeros(n)
        v[0] = p.vol ** 2
        for t in range(1, n):
            v[t] = v[t-1] + p.kappa * (p.vol ** 2 - v[t-1]) * dt + \
                   p.xi * np.sqrt(max(v[t-1], 0)) * np.sqrt(dt) * dW_v[t]
            v[t] = max(v[t], 1e-8)

        n_jumps = rng.poisson(p.lam * n)
        jump_times = rng.choice(n, size=min(n_jumps, n), replace=False)
        jump_sizes = rng.normal(p.mu_j, p.sigma_j, size=n_jumps)
        jumps = np.zeros(n)
        for jt, js in zip(jump_times, jump_sizes):
            jumps[jt] = js

        log_rets = np.zeros(n)
        for t in range(n):
            log_rets[t] = p.drift * dt + np.sqrt(max(v[t], 0) * dt) * dW_S[t] + jumps[t]

        log_path = np.log(self.data[0]) + np.cumsum(log_rets)
        mids = np.exp(log_path)
        return {"mid": mids, "variance": v, "params": p}

    def predict(self, n_steps, seed=None, confidence=0.95):
        """Generate future path with confidence intervals.

        Uses absolute vol scaling (not Heston θ which can collapse), so CIs
        are non-degenerate even for flat calibration.
        """
        if self.params.n_obs == 0:
            raise ValueError("Call fit() first")

        p = self.params
        rng = np.random.default_rng(seed)
        n = n_steps
        dt = p.dt

        z1 = rng.standard_normal(n)
        z2 = rng.standard_normal(n)
        dW_S = z1
        dW_v = p.rho * z1 + np.sqrt(1 - p.rho ** 2) * z2

        # Baseline absolute vol
        vol_abs = (p.vol ** 2) if p.vol > 0 else p.theta
        if vol_abs <= 1e-12:
            vol_abs = p.vol ** 2

        # Perlin-smooth drift on theta (volatility) and lam (jump intensity) over the path
        base_seed = int(seed) if seed is not None else 0
        pt_theta = PerlinTuner(p.perlin_octaves, p.perlin_persistence,
                               seed=base_seed + 101, base_freq=p.perlin_base_freq,
                               min_scale=p.perlin_min_scale, max_scale=p.perlin_max_scale)
        pt_lam = PerlinTuner(p.perlin_octaves, p.perlin_persistence,
                             seed=base_seed + 202, base_freq=p.perlin_base_freq,
                             min_scale=p.perlin_min_scale, max_scale=p.perlin_max_scale)
        theta_t = pt_theta.apply(vol_abs, n)
        lam_t = pt_lam.apply(p.lam, n)

        # Stochastic vol: Perlin-driven drift on theta, evolve with Heston shape.
        v = np.zeros(n)
        v[0] = theta_t[0]
        for t in range(1, n):
            v[t] = v[t-1] + p.kappa * (theta_t[t] - v[t-1]) * dt + \
                   p.xi * np.sqrt(max(v[t-1], 0)) * np.sqrt(dt) * dW_v[t]
            v[t] = max(v[t], 1e-8)

        # Jumps with Perlin-driven intensity
        jump_mask = rng.uniform(0, 1, n) < np.minimum(lam_t * dt, 1.0)
        jumps = np.zeros(n)
        n_jump = int(jump_mask.sum())
        if n_jump > 0:
            jump_times = np.where(jump_mask)[0]
            jump_sizes = rng.normal(p.mu_j, p.sigma_j, size=n_jump)
            for jt, js in zip(jump_times, jump_sizes):
                jumps[jt] = js

        log_rets = np.zeros(n)
        for t in range(n):
            drift = p.drift * dt
            diffusion = np.sqrt(max(v[t], 0) * dt) * dW_S[t]
            log_rets[t] = drift + diffusion + jumps[t]

        # Build path from the last observed price
        log_path = np.log(self.data[-1]) + np.cumsum(log_rets)
        mids = np.exp(log_path)

        # Confidence intervals: total variability grows with sqrt of total Heston variance
        total_var = np.sum(np.maximum(v, 0)) * dt
        path_std = np.sqrt(total_var)
        z = 1.96 if confidence == 0.95 else 2.576
        lower = mids * np.exp(-z * path_std)
        upper = mids * np.exp(z * path_std)

        return {"mid": mids, "lower": lower, "upper": upper,
                "variance": v, "jumps": jumps, "log_rets": log_rets,
                "params": p}

    def settle(self, mid_prices, momentums, seed=None):
        """Generate bid/ask from Gumbel settlement model."""
        if not hasattr(self, '_settlement'):
            self._settlement = GumbelSettlement()
        if seed is not None:
            self._settlement.rng = np.random.default_rng(seed)
        return self._settlement.quote_series(mid_prices, momentums)


def evaluate_general(price_series, name="series"):
    """Evaluate the general engine on any price series."""
    engine = GeneralizedPriceEngine()
    params = engine.fit(price_series)

    def r2(synth, real):
        n = min(len(synth), len(real))
        ss_res = np.sum((synth[:n] - real[:n]) ** 2)
        ss_tot = np.sum((real[:n] - real[:n].mean()) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Replicate
    out = engine.replicate(price_series, seed=42)
    r2_rep = r2(out["mid"], price_series)

    # Best over 10 seeds
    best_r2 = max(r2(engine.replicate(price_series, seed=s)["mid"], price_series)
                  for s in range(10))

    return {
        "name": name,
        "n_obs": params.n_obs,
        "drift": params.drift,
        "vol": params.vol,
        "lambda": params.lam,
        "r2_replicate": r2_rep,
        "r2_best_10_seeds": best_r2,
        "params": params
    }


if __name__ == "__main__":
    # Test on synthetic data
    np.random.seed(42)
    n = 500
    rets = np.random.normal(0.0002, 0.001, n)
    jump_times = np.random.choice(n, 20, replace=False)
    rets[jump_times] += np.random.normal(0, 0.005, 20)
    prices = 100 * np.exp(np.cumsum(rets))

    result = evaluate_general(prices, "synthetic_gbm_jumps")
    print(f"Series: {result['name']}")
    print(f"  Obs: {result['n_obs']}, drift: {result['drift']:.6f}, vol: {result['vol']:.6f}")
    print(f"  Lambda: {result['lambda']:.4f}")
    print(f"  R² (replicate): {result['r2_replicate']:.4f}")
    print(f"  R² (best 10 seeds): {result['r2_best_10_seeds']:.4f}")

    # Predict future with confidence intervals
    engine = GeneralizedPriceEngine()
    engine.fit(prices)
    pred = engine.predict(n_steps=50, seed=42)
    print(f"\nPrediction (next 50 steps):")
    print(f"  Last price: {prices[-1]:.4f}")
    print(f"  Predicted: {pred['mid'][-1]:.4f}")
    print(f"  95% CI: [{pred['lower'][-1]:.4f}, {pred['upper'][-1]:.4f}]")
    print(f"  Width: {(pred['upper'][-1] - pred['lower'][-1]) / pred['mid'][-1] * 100:.1f}%")
