"""Validate Bates engine across NIFTY 50: R² vs MomentumEngine vs legacy bridge.

For each stock, calibrates Bates params from the real 1-min path, generates
a synthetic path, and compares R² across all three engines.
"""

import numpy as np
import pandas as pd

from bates_engine import BatesEngine, BatesConfig
from momentum_engine import MomentumEngine, MomentumConfig
from gumbel_hawkes_sim import pin_the_range
from validate_nifty50 import NIFTY50, r2

N_SEEDS = 10


def evaluate_stock(tic, daily_df):
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

    # Bates: calibrate from real path, then generate
    bates = BatesEngine(BatesConfig(seed=42))
    params = bates.calibrate_from_real(real)
    bates_out = bates.simulate(o, h, l, c, params)
    r2_bates = r2(bates_out["mid"], real)

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

    return {"ticker": tic,
            "bates": float(r2_bates),
            "mom_mean": float(np.mean(mom_r2s)), "mom_best": float(np.max(mom_r2s)),
            "br_mean": float(np.mean(br_r2s)), "br_best": float(np.max(br_r2s)),
            "n_jumps": params["n_jumps"], "lam": params["lam"]}


def main():
    import yfinance as yf
    daily = yf.download(NIFTY50, period="5d", interval="1d", progress=False, group_by="group")
    results = []
    for tic in NIFTY50:
        try:
            res = evaluate_stock(tic, daily)
            if res:
                results.append(res)
                print(f"{tic:>15}  bates={res['bates']:+.3f}  mom={res['mom_mean']:+.3f}/{res['mom_best']:+.3f}  "
                      f"bridge={res['br_mean']:+.3f}  jumps={res['n_jumps']}")
        except Exception as e:
            print(f"{tic}: {e.__class__.__name__}")

    df = pd.DataFrame(results)
    print(f"\nStocks evaluated: {len(results)}/50")
    print(f"Bates R²:      mean={df['bates'].mean():.4f}  median={df['bates'].median():.4f}  "
          f"best={df['bates'].max():.4f}  positive={sum(df['bates'] > 0)}")
    print(f"Momentum R²:   mean={df['mom_mean'].mean():.4f}  median={df['mom_mean'].median():.4f}  "
          f"best={df['mom_best'].max():.4f}  positive={sum(df['mom_best'] > 0)}")
    print(f"Bridge R²:     mean={df['br_mean'].mean():.4f}  median={df['br_mean'].median():.4f}  "
          f"best={df['br_best'].max():.4f}  positive={sum(df['br_best'] > 0)}")
    print(f"\nBates beats momentum: {sum(df['bates'] > df['mom_mean'])}/{len(df)}")
    print(f"Bates beats bridge:   {sum(df['bates'] > df['br_mean'])}/{len(df)}")


if __name__ == "__main__":
    main()
