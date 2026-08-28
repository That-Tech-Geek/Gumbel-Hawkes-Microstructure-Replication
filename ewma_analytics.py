"""EWMA analytics: exponentially-weighted estimators for engine parameters.

Instead of plain mean/std over a lookback window, weight observations by
exp(-t/tau) so recent data dominates. Tunable via half-life or alpha.

Used for:
- Volatility (RiskMetrics-style): sigma_t^2 = lambda * sigma_{t-1}^2 + (1-lambda) * r_t^2
- Drift: EWMA mean of returns
- Jump intensity: EWMA rate of |r| > k*sigma events
- Lookback optimization: pick tau (half-life) that maximizes out-of-sample fit
"""

from dataclasses import dataclass

import numpy as np


def ewma_weights(n, tau):
    """Exponential weights over n points with half-life tau (in bars)."""
    t = np.arange(n)
    w = np.exp(-np.log(2) * t / tau)  # 0.5 at t=tau
    return w / w.sum()


def ewma_mean(x, tau):
    """EWMA mean of x with half-life tau."""
    x = np.asarray(x, dtype=float)
    w = ewma_weights(len(x), tau)
    return float(np.sum(w * x))


def ewma_vol(rets, tau):
    """RiskMetrics-style EWMA volatility with half-life tau."""
    rets = np.asarray(rets, dtype=float)
    w = ewma_weights(len(rets), tau)
    return float(np.sqrt(np.sum(w * rets**2)))


def ewma_series(x, tau):
    """Running EWMA of x at each point (causal). Returns array like x."""
    x = np.asarray(x, dtype=float)
    alpha = 1 - np.exp(-np.log(2) / tau)
    out = np.zeros(len(x))
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = alpha * x[t] + (1 - alpha) * out[t-1]
    return out


def ewma_vol_series(rets, tau):
    """Running EWMA vol (RiskMetrics recursion)."""
    rets = np.asarray(rets, dtype=float)
    alpha = 1 - np.exp(-np.log(2) / tau)
    v = np.zeros(len(rets))
    v[0] = rets[0] ** 2
    for t in range(1, len(rets)):
        v[t] = alpha * rets[t-1] ** 2 + (1 - alpha) * v[t-1]
    return np.sqrt(v)


@dataclass
class EWMAFit:
    tau: float
    vol: float
    drift: float
    lam: float
    score: float


def fit_ewma_params(rets, tau, jump_k=3.0):
    """Fit engine parameters using EWMA weighting with half-life tau."""
    rets = np.asarray(rets, dtype=float)
    vol = ewma_vol(rets, tau)
    drift = ewma_mean(rets, tau)
    # Jump intensity: EWMA rate of exceedances
    exceed = (np.abs(rets) > jump_k * vol).astype(float)
    lam = ewma_mean(exceed, tau)
    return {"tau": tau, "vol": vol, "drift": drift, "lam": lam}


def optimize_lookback(rets, taus=None, jump_k=3.0, scorer=None):
    """Grid-search half-life tau to maximize a scoring function.

    Default scorer: 1-step-ahead EWMA vol predicts |r_{t+1}| well
    (minimize MSE of |r| - ewma_vol). Lower MSE = better vol fit.
    Returns best tau and full table.
    """
    rets = np.asarray(rets, dtype=float)
    if taus is None:
        taus = np.logspace(np.log10(2), np.log10(len(rets) / 2), 12)
    if scorer is None:
        def scorer(tau):
            vs = ewma_vol_series(rets, tau)
            # 1-step-ahead: predict |r_t| with vol_{t-1}
            pred = vs[:-1]
            actual = np.abs(rets[1:])
            return -float(np.mean((pred - actual) ** 2))  # negative MSE

    table = []
    for tau in taus:
        score = scorer(tau)
        table.append(EWMAFit(tau=float(tau), vol=ewma_vol(rets, tau),
                             drift=ewma_mean(rets, tau),
                             lam=fit_ewma_params(rets, tau, jump_k)["lam"],
                             score=score))
    best = max(table, key=lambda f: f.score)
    return best, table


if __name__ == "__main__":
    np.random.seed(42)
    # Synthetic: vol clustering (low then high)
    rets = np.concatenate([np.random.normal(0, 0.001, 200),
                           np.random.normal(0, 0.004, 200)])
    best, table = optimize_lookback(rets)
    print(f"Best tau: {best.tau:.1f} bars (score={best.score:.2e})")
    print(f"  EWMA vol={best.vol:.6f}, drift={best.drift:.6f}, lam={best.lam:.4f}")
    for f in table:
        print(f"  tau={f.tau:6.1f}  score={f.score:.3e}")
