"""Validate the MomentumEngine across NIFTY 50 with real 1-min data.

For each stock, runs 20 seeds and reports mean/best R² vs real minute closes,
alongside the standard pin_the_range bridge for comparison.
"""

import numpy as np
import pandas as pd

from gumbel_hawkes_sim import pin_the_range
from momentum_engine import MomentumEngine, MomentumConfig
from validate_nifty50 import NIFTY50, r2

N_SEEDS = 20


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

    sigma = float(np.std(np.diff(np.log(real))))

    # Momentum engine: distribution over seeds
    mom_r2s = []
    for seed in range(N_SEEDS):
        eng = MomentumEngine(MomentumConfig(seed=seed, sigma=sigma))
        out = eng.run(o, h, l, c)
        mom_r2s.append(r2(out["mid"], real))

    # Standard bridge: distribution over seeds
    br_r2s = []
    for seed in range(N_SEEDS):
        np.random.seed(seed)
        br_r2s.append(r2(pin_the_range(o, h, l, c, len(real)), real))

    return {"ticker": tic,
            "mom_mean": float(np.mean(mom_r2s)), "mom_best": float(np.max(mom_r2s)),
            "br_mean": float(np.mean(br_r2s)), "br_best": float(np.max(br_r2s))}


def main():
    import yfinance as yf
    daily = yf.download(NIFTY50, period="5d", interval="1d", progress=False, group_by="group")
    results = []
    for tic in NIFTY50:
        try:
            res = evaluate_stock(tic, daily)
            if res:
                results.append(res)
                print(f"{tic:>15}  momentum mean={res['mom_mean']:+.3f} best={res['mom_best']:+.3f}  "
                      f"bridge mean={res['br_mean']:+.3f}")
        except Exception as e:
            print(f"{tic}: {e.__class__.__name__}")

    df = pd.DataFrame(results)
    print(f"\nStocks evaluated: {len(results)}/50")
    print(f"Momentum engine R²: mean={df['mom_mean'].mean():.4f}  median={df['mom_mean'].median():.4f}  "
          f"best-per-stock={df['mom_best'].mean():.4f}")
    print(f"Standard bridge R²: mean={df['br_mean'].mean():.4f}  median={df['br_mean'].median():.4f}")
    print(f"Mean uplift (momentum - bridge): {(df['mom_mean'] - df['br_mean']).mean():.4f}")
    print(f"Stocks with positive momentum-engine R² (best seed): {(df['mom_best'] > 0).sum()}")


if __name__ == "__main__":
    main()
