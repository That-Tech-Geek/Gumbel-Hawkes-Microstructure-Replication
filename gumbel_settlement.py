"""Gumbel settlement model: bid/ask distribution fitted to price momentum.

The idea: settlement happens at some (bid, ask) around the mid. The spread is not
symmetric — it skews with momentum. We fit a Gumbel distribution to the spread
residual (how far bid/ask deviate from mid) and use momentum to tilt the asymmetry.

Model:
    spread = base_spread + gumbel_sample * scale
    tilt = tanh(momentum * skew_sensitivity)  # -1 to +1
    bid = mid * (1 - spread * (1 - tilt))   # wider bid when momentum negative
    ask = mid * (1 + spread * (1 + tilt))   # wider ask when momentum positive

The Gumbel distribution captures the right-skewed spread distribution (most trades
have tight spreads, but occasional wide spreads during volatility).
"""

import numpy as np


class GumbelSettlement:
    """Gumbel bid/ask settlement model."""

    def __init__(self, base_spread=0.0005, scale=0.0003, skew_sensitivity=50.0):
        self.base_spread = base_spread
        self.scale = scale
        self.skew_sensitivity = skew_sensitivity
        self.rng = np.random.default_rng()

    def fit(self, mid_prices, bid_prices, ask_prices):
        """Fit Gumbel parameters from observed bid/ask data.

        mid_prices: array of mid prices
        bid_prices: array of bid prices
        ask_prices: array of ask prices
        """
        mid = np.asarray(mid_prices, dtype=float)
        bid = np.asarray(bid_prices, dtype=float)
        ask = np.asarray(ask_prices, dtype=float)

        # Compute spread residuals
        spread = (ask - bid) / mid
        spread = spread[~np.isnan(spread)]
        spread = spread[spread > 0]

        # Fit lognormal to spread distribution (positive, right-skewed)
        log_spread = np.log(spread)
        mu = np.mean(log_spread)
        sigma = np.std(log_spread)
        self.base_spread = float(np.exp(mu))  # geometric mean
        self.scale = float(sigma)

        # Fit momentum tilt
        momentum = np.gradient(np.log(mid))
        # Regress: spread ~ base + |momentum| * skew
        abs_mom = np.abs(momentum)
        if len(abs_mom) > 5:
            # Simple linear fit
            x = np.column_stack([np.ones(len(abs_mom)), abs_mom])
            y = spread
            coef = np.linalg.lstsq(x, y, rcond=None)[0]
            self.base_spread = float(coef[0])
            self.skew_sensitivity = float(coef[1])

        return {"base_spread": self.base_spread, "scale": self.scale,
                "skew_sensitivity": self.skew_sensitivity}

    def sample_spread(self, n=1):
        """Sample spread from lognormal distribution."""
        log_spread = self.rng.normal(np.log(self.base_spread), self.scale, n)
        return np.maximum(np.exp(log_spread), 0.0001)

    def quote(self, mid, momentum):
        """Generate bid/ask from mid price and momentum.

        momentum: scalar or array (e.g., from np.gradient(np.log(mid)))
        """
        spread = self.sample_spread(1)[0]
        tilt = np.tanh(momentum * self.skew_sensitivity)
        # tilt > 0: widen ask (bullish), tilt < 0: widen bid (bearish)
        bid = mid * (1 - spread * (1 - tilt))
        ask = mid * (1 + spread * (1 + tilt))
        return bid, ask

    def quote_series(self, mid_prices, momentums):
        """Generate bid/ask series from mid prices and momentums."""
        mid = np.asarray(mid_prices, dtype=float)
        mom = np.asarray(momentums, dtype=float)
        n = len(mid)
        spreads = self.sample_spread(n)
        tilts = np.tanh(mom * self.skew_sensitivity)
        bids = mid * (1 - spreads * (1 - tilts))
        asks = mid * (1 + spreads * (1 + tilts))
        return bids, asks


def validate_gumbel_settlement():
    """Validate on real NIFTY 1-min data."""
    import yfinance as yf
    m = yf.download("RELIANCE.NS", period="1d", interval="1m", progress=False)
    if hasattr(m.columns, 'get_level_values'):
        m.columns = m.columns.get_level_values(0)
    mid = m["Close"].dropna().values.astype(float)
    high = m["High"].dropna().values.astype(float)
    low = m["Low"].dropna().values.astype(float)

    # Proxy bid/ask from high/low
    # Assume bid ~ low, ask ~ high for 1-min bars
    bid_proxy = low
    ask_proxy = high

    engine = GumbelSettlement()
    engine.fit(mid, bid_proxy, ask_proxy)

    print(f"Fitted: base_spread={engine.base_spread*10000:.2f}bps, "
          f"scale={engine.scale*10000:.2f}bps, skew={engine.skew_sensitivity:.2f}")

    # Generate synthetic bid/ask
    momentums = np.gradient(np.log(mid))
    bids, asks = engine.quote_series(mid, momentums)

    # Compare to proxy
    real_spread = (ask_proxy - bid_proxy) / mid
    synth_spread = (asks - bids) / mid
    print(f"Real spread: mean={np.mean(real_spread)*10000:.2f}bps, "
          f"std={np.std(real_spread)*10000:.2f}bps")
    print(f"Synth spread: mean={np.mean(synth_spread)*10000:.2f}bps, "
          f"std={np.std(synth_spread)*10000:.2f}bps")

    # Check if synthetic spread distribution matches real
    from scipy import stats
    ks_stat, ks_p = stats.ks_2samp(real_spread, synth_spread)
    print(f"KS test: stat={ks_stat:.4f}, p={ks_p:.4f}")

    return engine, real_spread, synth_spread


if __name__ == "__main__":
    engine, real, synth = validate_gumbel_settlement()
