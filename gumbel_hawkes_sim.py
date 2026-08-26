"""
Gumbel-Hawkes-Range-Pinning Simulator (MVP)

Phase 1: pin_the_range   - turns ONE daily OHLC row into 390 minute mid-prices
Phase 2: SimpleHawkes    - self-exciting jump process (volatility clusters)
Phase 3: gumbel_spread   - momentum-skewed Bid/Ask quotes
Phase 4: Full loop       - simulated tick-by-tick trading day
"""

import numpy as np
import pandas as pd


# ---------------- Phase 1: The Range-Pinning Bridge ----------------

def pin_the_range(open_, high, low, close, n_bars=390):
    """
    The Magic Bridge: forces the path to start near Open, end at Close,
    and hit High/Low exactly.
    """
    t = np.linspace(0, 1, n_bars)
    trend = open_ + (close - open_) * t

    raw_noise = np.random.normal(0, 1, n_bars).cumsum()
    # Hard-pin both ends to 0 so the path starts exactly at Open
    bridge = raw_noise - raw_noise[0] - t * (raw_noise[-1] - raw_noise[0])

    # Zero both boundaries, then rescale interior into the feasible envelope
    lo_lim = low - trend
    hi_lim = high - trend
    bridge -= bridge[0] * (1 - t) + bridge[-1] * t
    span = bridge.max() - bridge.min()
    if span > 0:
        bridge = (bridge - bridge.min()) / span * (hi_lim - lo_lim) + lo_lim
    bridge[[0, -1]] = 0.0

    return trend + bridge


# ---------------- Phase 2: The Hawkes "Excitement" ----------------

class SimpleHawkes:
    # Defaults fitted on real Nifty 1-min data via MLE (see fit_hawkes.py)
    def __init__(self, alpha=0.036, beta=0.286, base_intensity=0.2):
        self.alpha = alpha
        self.beta = beta
        self.lambda_t = base_intensity
        self.base = base_intensity

    def step(self, price, dt=1 / 390):
        self.lambda_t = self.base + (self.lambda_t - self.base) * np.exp(-self.beta * dt)
        if np.random.rand() < self.lambda_t * dt * 50:
            jump_size = np.random.normal(0, 0.002) * price
            self.lambda_t += self.alpha
            return price + jump_size
        return price


# ---------------- Phase 3: The Gumbel Spread Engine ----------------

def gumbel_spread(mid_price, momentum, base_spread=0.001):
    mu = 0.0005 + abs(momentum) * 0.01
    sigma = 0.0005 + abs(momentum) * 0.005

    u = np.random.rand()
    spread_pct = mu - sigma * np.log(-np.log(u))
    spread_pct = max(spread_pct, 0.0001)

    if momentum > 0:
        ask = mid_price * (1 + spread_pct * 1.5)
        bid = mid_price * (1 - spread_pct * 0.5)
    else:
        ask = mid_price * (1 + spread_pct * 0.5)
        bid = mid_price * (1 - spread_pct * 1.5)

    return bid, ask


# ---------------- Data Loader (with offline fallback) ----------------

def get_daily_ohlc(ticker="RELIANCE.NS"):
    """Fetch the most recent daily candle; fall back to synthetic data offline."""
    try:
        import yfinance as yf
        data = yf.download(ticker, period="5d", interval="1d", progress=False)
        if data is None or data.empty:
            raise ValueError("empty dataframe")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        row = data.iloc[-1]
        return float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"]), str(data.index[-1].date())
    except Exception as e:
        print(f"[yfinance unavailable ({e.__class__.__name__}) - using synthetic OHLC]")
        return 2500.0, 2560.0, 2475.0, 2530.0, "synthetic"


# ---------------- Phase 4: The "Sanitized Mock Market" Loop ----------------

def run_simulation(seed=None):
    if seed is not None:
        np.random.seed(seed)

    open_p, high_p, low_p, close_p, day = get_daily_ohlc()
    print(f"Modeling {day}: Open={open_p:.2f}, High={high_p:.2f}, Low={low_p:.2f}, Close={close_p:.2f}")

    mid_ticks = pin_the_range(open_p, high_p, low_p, close_p, n_bars=390)
    print(f"Bridge check -> Start: {mid_ticks[0]:.2f} | End: {mid_ticks[-1]:.2f} | "
          f"Max: {mid_ticks.max():.2f} | Min: {mid_ticks.min():.2f}")

    hawkes = SimpleHawkes()
    inventory = 0
    cash = 100_000

    for i, mid in enumerate(mid_ticks):
        mid = np.clip(hawkes.step(mid), low_p, high_p)  # keep jumps inside daily range
        momentum = (mid - open_p) / open_p
        bid, ask = gumbel_spread(mid, momentum)

        if i % 10 == 0:  # placeholder agent: passive buy every 10 minutes
            shares_to_buy = int(cash * 0.1 / ask)
            cash -= shares_to_buy * ask
            inventory += shares_to_buy

        if inventory * mid > 50_000:  # drawdown safety
            print(f"⚠️ Liquidation at minute {i}! Selling at {bid:.2f}")
            cash += inventory * bid
            inventory = 0

    final_mid = mid_ticks[-1]
    print(f"End of Day - Cash: ₹{cash:,.2f}, Inventory: {inventory} shares")
    print(f"Total Equity: ₹{cash + inventory * final_mid:,.2f}")
    return cash, inventory, final_mid


if __name__ == "__main__":
    # --- Phase 1 standalone test ---
    print("=== Phase 1 Test ===")
    ticks = pin_the_range(100, 105, 95, 102, 390)
    print(f"Start: {ticks[0]:.2f}")
    print(f"End: {ticks[-1]:.2f}")
    print(f"Max: {ticks.max():.2f}")
    print(f"Min: {ticks.min():.2f}")
    print(f"First 10 ticks: {np.round(ticks[:10], 2)}")

    print("\n=== Phase 4 Full Simulation ===")
    run_simulation()
