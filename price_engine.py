"""Pure price-generation engine: lognormal base + momentum-Hawkes + Gumbel spread.

No trading logic. Produces realistic bid/ask/mid tick streams from one daily OHLC bar.
"""

from dataclasses import dataclass

import numpy as np

from gumbel_hawkes_sim import pin_the_range


@dataclass
class EngineConfig:
    n_bars: int = 390
    mu_daily: float = 0.0            # drift (in log-returns per bar)
    sigma_daily: float = 0.008       # per-bar volatility of log returns
    base_intensity: float = 0.15     # Hawkes baseline events per bar
    fit_alpha: float = 0.036         # fitted on Nifty 1-min data
    fit_beta: float = 0.286          # fitted decay
    momentum_coupling: float = 5.0   # how strongly |momentum| raises jump intensity
    spread_base: float = 0.0005      # baseline half-spread
    skew_sensitivity: float = 30.0   # how strongly momentum skews bid vs ask width
    momentum_ema: float = 10.0       # EMA horizon in bars for momentum calc
    seed: int | None = None


class MomentumHawkes:
    """Self-exciting jump process where intensity scales with momentum magnitude."""

    def __init__(self, cfg: EngineConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.lambda_t = cfg.base_intensity
        self.momentum_ref = 0.0

    def jump_now(self, momentum: float = 0.0) -> tuple[float, float]:
        """Advance one bar. Returns (jump_size_log, new_lambda_t)."""
        c, r = self.cfg, self.rng
        lam = c.base_intensity + (self.lambda_t - c.base_intensity) * np.exp(-c.fit_beta)
        # momentum drives excitement up multiplicatively
        lam *= 1.0 + c.momentum_coupling * abs(momentum)
        if r.random() < lam:
            # small jumps: sigma_daily / sqrt(n_bars) per bar, not 3x unleashed
            magnitude = r.standard_t(df=5) * (c.sigma_daily / np.sqrt(c.n_bars)) * 0.5
            self.lambda_t += c.fit_alpha * abs(magnitude) * 100  # excite by meaningful amount
            return magnitude, self.lambda_t
        self.lambda_t = lam
        return 0.0, self.lambda_t


def asymmetric_spread(mid: float, momentum: float, spread_pct: float,
                      cfg: EngineConfig) -> tuple[float, float]:
    """Gumbel-style asymmetric quote: momentum widens the far side modestly."""
    tilt = min(cfg.skew_sensitivity * abs(momentum), 0.3)
    if momentum > 0:
        bid = mid * (1 - spread_pct * (1 - tilt))
        ask = mid * (1 + spread_pct * (1 + tilt))
    else:
        bid = mid * (1 - spread_pct * (1 + tilt))
        ask = mid * (1 + spread_pct * (1 - tilt))
    return bid, ask


def gumbel_u_sample(rng: np.random.Generator) -> float:
    """Generalized Gumbel-style spread sample via inverse-CDF of max of exponentials."""
    # sample from a Gumbel-like shape: max of two exponentials -> single-sided fat tail
    x = rng.exponential(1.0)
    y = rng.exponential(1.0)
    return max(x, y) / 5.0


class PriceEngine:
    """Full tick-generation engine. Yields dict per bar with mid, bid, ask, aux signals."""

    def __init__(self, cfg: EngineConfig | None = None):
        self.cfg = cfg or EngineConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

    def run(self, open_price: float, high: float, low: float, close: float):
        """Bridge-anchored base + Hawkes log overlay + asymmetric Gumbel quote."""
        base_bridge = pin_the_range(open_price, high, low, close, self.cfg.n_bars)
        log_base = np.log(base_bridge / open_price)

        hawkes = MomentumHawkes(self.cfg, self.rng)
        ema_ref = open_price
        mids, bids, asks, momentums, lambdas = [], [], [], [], []

        for t in range(self.cfg.n_bars):
            base_mid = open_price * np.exp(log_base[t])
            # momentum = log distance from EMA anchor
            momentum = (np.log(base_mid) - np.log(ema_ref)) if ema_ref > 0 else 0.0
            ema_ref = ema_ref * np.exp(momentum / self.cfg.momentum_ema)

            jump, lam = hawkes.jump_now(momentum * 100)  # intensity uses standardized momentum (x100)
            mid_ = base_mid * np.exp(jump)  # Hawkes overlay in log space
            # Keep within daily bounds (pin base, clip jumps)
            mid_ = float(np.clip(mid_, low, high))
            ema_ref = ema_ref  # keep EMA anchored to base, not to clipped mid

            u = gumbel_u_sample(self.rng)
            # Cap momentum contribution: spread grows by up to 2x base in trending regimes
            spboost = min(abs(momentum) * 2, 2.0)
            spread_pct = max(self.cfg.spread_base * (1 + spboost) * (0.5 + u * 0.2), 0.0001)
            bid, ask = asymmetric_spread(mid_, momentum * 100, spread_pct, self.cfg)

            mids.append(mid_)
            bids.append(bid)
            asks.append(ask)
            momentums.append(momentum)
            lambdas.append(lam)

        return {
            "mid": np.array(mids),
            "bid": np.array(bids),
            "ask": np.array(asks),
            "momentum": np.array(momentums),
            "lambda": np.array(lambdas),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
        }


if __name__ == "__main__":
    e = PriceEngine(EngineConfig(seed=42))
    out = e.run(100.0, 105.0, 95.0, 102.0)
    m = out["mid"]
    spread_pct = (out["ask"] - out["bid"]) / out["mid"]
    print(f"generated {len(m)} bars | start={m[0]:.4f} end={m[-1]:.4f} "
          f"max={m.max():.4f} min={m.min():.4f}")
    print(f"spread bps: mean={10000*spread_pct.mean():.2f}, max={10000*spread_pct.max():.2f}, "
          f"bad ticks={(out['bid'] >= out['ask']).sum()}")
    print(f"avg hawkes lambda: {out['lambda'].mean():.3f} (base {e.cfg.base_intensity})")
