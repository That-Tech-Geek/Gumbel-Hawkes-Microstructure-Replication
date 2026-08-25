"""General price replication algorithm: phase alignment via event anchoring.

Universal interface — provider hands us (OHLC, observed minute path) and gets back
a structural synthetic path (mid, bid, ask). No assumptions about the provider:
works for NIFTY 1-min bars, crypto ticks, or any intraday series.

Core idea (fixes the phase problem): the Brownian bridge pins the *value* of the
High/Low but not their *time*. Meanwhile, observed event times (|Δlog| > threshold)
fix the phase. So: anchor the first k minutes to the real path, run the day via a
re-pinned bridge from the anchor, and inject deterministic jumps exactly at the
observed event minutes with magnitude calibrated by the fitted Hawkes branching
ratio. Spread sizing uses rolling realized volatility (not momentum).
"""

from dataclasses import dataclass

import numpy as np

from gumbel_hawkes_sim import pin_the_range


@dataclass
class ReplicationConfig:
    anchor_minutes: int = 10          # minutes of real path used as conditioning anchor
    event_threshold: float = 0.001    # |log return| above this flags an event
    base_intensity: float = 0.15
    alpha: float = 0.036              # fitted branching ratio (NSE 1-min MLE)
    beta: float = 0.286               # fitted decay (NSE 1-min MLE)
    spread_base: float = 0.0005       # baseline half-spread
    spread_vol_scale: float = 0.5     # how strongly realized vol widens the spread
    vol_window: int = 5               # rolling vol estimate window (minutes)
    gumbel_quantile: float = 0.6      # fixed Gumbel quantile -> deterministic spread
    skew_sensitivity: float = 30.0    # momentum -> asymmetric bid/ask tilt
    n_bars: int = 390                 # fallback length if no real path given
    seed: int | None = None


def extract_events(real_path: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (event_times, event_magnitudes) from an observed price path."""
    logr = np.diff(np.log(np.asarray(real_path, dtype=float)))
    idx = np.where(np.abs(logr) > threshold)[0]
    return idx + 1, np.abs(logr[idx])


def rolling_log_vol(path: np.ndarray, window: int) -> np.ndarray:
    """Rolling per-minute realized log-return volatility, causal."""
    logr = np.diff(np.log(np.asarray(path, dtype=float)))
    n = len(path)
    out = np.full(n, np.std(logr[:window]) if len(logr) >= window else 0.0)
    for t in range(window, n):
        out[t] = np.std(logr[max(0, t - window):t])
    return out


def deterministic_quantile_uniform(q: float, n: int) -> np.ndarray:
    """Fixed-sequence deterministic spread samples near quantile q (no RNG)."""
    return np.clip(np.full(n, q) + np.linspace(-0.02, 0.02, n), 0, 1)


class ReplicationEngine:
    """Event-anchored synthetic path generator. Fully deterministic given inputs."""

    def __init__(self, cfg: ReplicationConfig | None = None):
        self.cfg = cfg or ReplicationConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

    def replicate(self, open_: float, high: float, low: float, close: float,
                  real_path: np.ndarray) -> dict:
        c = self.cfg
        real = np.asarray(real_path, dtype=float)
        n = len(real)
        k = min(c.anchor_minutes, n - 1)

        # Step 1 (conditioning): copy the first k minutes verbatim
        anchor = real[:k]

        # Step 2: re-pin the bridge from the anchor endpoint through the day's close
        n_rest = n - k
        if n_rest > 0:
            base_rest = pin_the_range(anchor[-1], high, low, close, n_rest)
        else:
            base_rest = np.array([])

        # Step 3: observed event schedule -> deterministic jump injection
        ev_times, ev_mags = extract_events(real, c.event_threshold)
        ev_set = set(int(t) for t in ev_times)
        median_jump = float(np.median(ev_mags)) if len(ev_mags) else 0.002

        # drift sign: events push toward close (bridge trend direction)
        day_sign = np.sign(close - anchor[-1]) if n_rest > 0 else 1.0

        lam = c.base_intensity
        mids_rest, jumps = [], []
        for i in range(n_rest):
            base = base_rest[i]
            lam = c.base_intensity + (lam - c.base_intensity) * np.exp(-c.beta)
            t_abs = k + i
            if t_abs in ev_set:
                lam = c.base_intensity + c.alpha * median_jump * 100
                jump = day_sign * median_jump * (lam / (c.base_intensity + 1e-9))
            else:
                jump = 0.0
            mids_rest.append(float(np.clip(base * np.exp(jump), low, high)))
            jumps.append(jump)

        mid = np.concatenate([anchor, np.array(mids_rest)])

        # Step 4: volatility-sized deterministic spread + momentum tilt
        vol = rolling_log_vol(real, c.vol_window)
        momentum = np.gradient(np.log(mid)) if n > 1 else np.zeros(n)
        qseq = deterministic_quantile_uniform(c.gumbel_quantile, n)
        spread_pct = np.maximum(
            c.spread_base * (0.5 + 0.2 * qseq) * (1 + c.spread_vol_scale * vol * 100),
            0.0001)

        tilt = np.minimum(c.skew_sensitivity * momentum * 10, 0.3)  # signed
        ask_pct = spread_pct * (1 + tilt)
        bid_pct = spread_pct * (1 - tilt)
        bid = mid * (1 - bid_pct)
        ask = mid * (1 + ask_pct)

        return {"mid": mid, "bid": bid, "ask": ask,
                "events": ev_times, "lambda_decay": lam}


if __name__ == "__main__":
    real = np.linspace(100, 102, 390) + np.random.default_rng(0).normal(0, 0.3, 390)
    eng = ReplicationEngine(ReplicationConfig(seed=0))
    out = eng.replicate(100.0, 105.0, 95.0, 102.0, real)
    print(f"n={len(out['mid'])} events={len(out['events'])} "
          f"max={out['mid'].max():.3f} min={out['mid'].min():.3f} "
          f"first-anchored={out['mid'][0]:.4f}")
