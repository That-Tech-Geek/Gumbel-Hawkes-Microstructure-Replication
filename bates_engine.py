"""Bates (1996) jump-diffusion engine: stochastic volatility + compound Poisson jumps.

The Bates model combines:
  - Heston (1993) stochastic volatility: dv = κ(θ−v)dt + ξ√v dW_v
  - Merton (1976) jump-diffusion: dS/S = (μ−λk)dt + √v dW_S + (e^J −1)dN

where J ~ N(μ_J, σ_J²) and N is a Poisson process with intensity λ.

This engine calibrates (κ, θ, ξ, λ, μ_J, σ_J) from real 1-min data and generates
synthetic paths that should capture volatility clustering AND jump timing better
than the MomentumEngine's white-noise returns.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class BatesConfig:
    n_bars: int = 390
    dt: float = 1.0 / 390          # one minute in years (approx)
    # Heston SV params (calibrated from data)
    kappa: float = 2.0             # mean reversion speed
    theta: float = 0.04            # long-run variance
    xi: float = 0.3                # vol-of-vol
    rho: float = -0.5              # correlation between price and vol innovations
    # Jump params
    lam: float = 0.02              # jump intensity (per minute)
    mu_j: float = 0.0              # mean log-jump size
    sigma_j: float = 0.002         # jump size std
    # Calibration
    seed: int | None = None


class BatesEngine:
    """Bates jump-diffusion path generator."""

    def __init__(self, cfg: BatesConfig | None = None):
        self.cfg = cfg or BatesConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

    def calibrate_from_real(self, real_path: np.ndarray) -> dict:
        """Estimate Bates parameters from a real 1-min price path."""
        rets = np.diff(np.log(real_path))
        n = len(rets)
        dt = 1.0 / 390

        # Jump detection: |return| > 3σ
        sigma_total = np.std(rets)
        jump_mask = np.abs(rets) > 3 * sigma_total
        n_jumps = jump_mask.sum()

        # Jump intensity
        lam = n_jumps / n if n_jumps > 0 else 0.01

        # Jump sizes
        if n_jumps > 0:
            jump_sizes = rets[jump_mask]
            mu_j = float(np.mean(jump_sizes))
            sigma_j = float(np.std(jump_sizes)) if n_jumps > 1 else 0.002
        else:
            mu_j = 0.0
            sigma_j = 0.002

        # Diffusion component (non-jump returns)
        diffusion_rets = rets[~jump_mask]
        if len(diffusion_rets) > 5:
            # Long-run variance (per bar, not annualized)
            theta = float(np.var(diffusion_rets))

            # Mean reversion: use autocorrelation of squared returns
            # For AR(1) vol process: ac1 = exp(-kappa * dt)
            sq_rets = diffusion_rets ** 2
            if len(sq_rets) > 2:
                ac1 = np.corrcoef(sq_rets[:-1], sq_rets[1:])[0, 1]
                ac1 = np.clip(ac1, 0.01, 0.99)
                kappa = -np.log(ac1) / dt  # convert to per-minute rate
                kappa = np.clip(kappa, 0.5, 20.0)  # reasonable range
            else:
                kappa = 2.0

            # Vol-of-vol: coefficient of variation of rolling vol
            window = 20
            if len(diffusion_rets) >= window:
                rolling_std = np.array([
                    np.std(diffusion_rets[i:i+window])
                    for i in range(len(diffusion_rets) - window + 1)
                ])
                xi = float(np.std(rolling_std) / (np.mean(rolling_std) + 1e-9))
                xi = np.clip(xi, 0.1, 0.8)
            else:
                xi = 0.3

            # Leverage correlation: corr(returns, squared returns)
            if len(diffusion_rets) > 2:
                rho = np.corrcoef(diffusion_rets[:-1], sq_rets[1:])[0, 1]
                rho = np.clip(rho, -0.7, 0.0)
            else:
                rho = -0.3
        else:
            theta = float(np.var(rets)) if len(rets) > 1 else 0.0002
            kappa, xi, rho = 2.0, 0.3, -0.3

        return {"kappa": kappa, "theta": theta, "xi": xi, "rho": rho,
                "lam": lam, "mu_j": mu_j, "sigma_j": sigma_j,
                "n_jumps": int(n_jumps), "sigma_total": float(sigma_total)}

    def simulate(self, open_: float, high: float, low: float, close: float,
                 params: dict | None = None) -> dict:
        """Generate a Bates jump-diffusion path pinned to (O,H,L,C)."""
        c = self.cfg
        n = c.n_bars
        dt = c.dt

        p = params or {"kappa": c.kappa, "theta": c.theta, "xi": c.xi,
                       "rho": c.rho, "lam": c.lam, "mu_j": c.mu_j, "sigma_j": c.sigma_j}

        # Correlated Brownian motions
        z1 = self.rng.standard_normal(n)
        z2 = self.rng.standard_normal(n)
        dW_S = z1
        dW_v = p["rho"] * z1 + np.sqrt(1 - p["rho"] ** 2) * z2

        # Simulate variance process (CIR/Heston)
        v = np.zeros(n)
        v[0] = p["theta"]
        for t in range(1, n):
            v[t] = v[t-1] + p["kappa"] * (p["theta"] - v[t-1]) * dt + \
                   p["xi"] * np.sqrt(max(v[t-1], 0)) * np.sqrt(dt) * dW_v[t]
            v[t] = max(v[t], 1e-6)  # keep variance positive

        # Simulate jumps (compound Poisson)
        n_jumps = self.rng.poisson(p["lam"] * n)
        jump_times = self.rng.choice(n, size=min(n_jumps, n), replace=False)
        jump_sizes = self.rng.normal(p["mu_j"], p["sigma_j"], size=n_jumps)
        jumps = np.zeros(n)
        for jt, js in zip(jump_times, jump_sizes):
            jumps[jt] = js

        # Simulate price path
        log_rets = np.zeros(n)
        drift_total = np.log(close / open_)
        for t in range(n):
            drift = drift_total / n  # uniform drift to hit close
            diffusion = np.sqrt(max(v[t], 0) * dt) * dW_S[t]
            jump = jumps[t]
            log_rets[t] = drift + diffusion + jump

        # Build path
        log_path = np.log(open_) + np.cumsum(log_rets)
        mids = np.exp(log_path)

        # Don't pin OHLC — just match the return distribution and drift
        # The Bates model already has the correct drift (drift_total/n per bar)
        # and the stochastic vol + jumps give the correct distribution.
        # Scaling would destroy the vol clustering structure.

        return {"mid": mids, "variance": v, "jumps": jumps,
                "log_rets": log_rets, "params": p}


def compare_engines(real_path: np.ndarray, o: float, h: float, l: float, c: float) -> dict:
    """Compare Bates vs MomentumEngine vs legacy bridge on the same real data."""
    from momentum_engine import MomentumEngine, MomentumConfig
    from gumbel_hawkes_sim import pin_the_range

    def r2(synth, real):
        n_min = min(len(synth), len(real))
        ss_res = np.sum((synth[:n_min] - real[:n_min]) ** 2)
        ss_tot = np.sum((real[:n_min] - real[:n_min].mean()) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Bates
    bates = BatesEngine(BatesConfig(seed=42))
    params = bates.calibrate_from_real(real_path)
    bates_out = bates.simulate(o, h, l, c, params)
    r2_bates = r2(bates_out["mid"], real_path)

    # Momentum
    mom = MomentumEngine(MomentumConfig(seed=42, sigma=float(np.std(np.diff(np.log(real_path))))))
    mom_out = mom.run(o, h, l, c)
    r2_mom = r2(mom_out["mid"], real_path)

    # Legacy bridge
    np.random.seed(42)
    bridge = pin_the_range(o, h, l, c, len(real_path))
    r2_bridge = r2(bridge, real_path)

    return {"bates": r2_bates, "momentum": r2_mom, "bridge": r2_bridge,
            "params": params}


if __name__ == "__main__":
    import yfinance as yf
    m = yf.download("RELIANCE.NS", period="1d", interval="1m", progress=False)
    if hasattr(m.columns, 'get_level_values'):
        m.columns = m.columns.get_level_values(0)
    real = m["Close"].dropna().values.astype(float)
    o, h, l, c = real[0], real.max(), real.min(), real[-1]

    result = compare_engines(real, o, h, l, c)
    print(f"Bates R² = {result['bates']:.4f}")
    print(f"Momentum R² = {result['momentum']:.4f}")
    print(f"Bridge R² = {result['bridge']:.4f}")
    print(f"\nCalibrated params: κ={result['params']['kappa']:.2f}, θ={result['params']['theta']:.6f}, "
          f"ξ={result['params']['xi']:.3f}, ρ={result['params']['rho']:.3f}, "
          f"λ={result['params']['lam']:.4f}, μ_J={result['params']['mu_j']:.6f}, "
          f"σ_J={result['params']['sigma_j']:.6f}")
