"""Validate the ReplicationEngine on NIFTY 50: R2 (path alignment) + KS (distribution fit).

For each stock:
  1. Pull today's 1-min realized path.
  2. Run the engine deterministically (anchor + event schedule + re-pinned bridge).
  3. R2 of synthetic mid vs real close (phase-alignment metric).
  4. Two-sample KS distance between synthetic and real minute log-return distributions.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from replication_engine import ReplicationEngine, ReplicationConfig
from validate_nifty50 import NIFTY50, r2


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

    eng = ReplicationEngine(ReplicationConfig(seed=0))
    out = eng.replicate(o, h, l, c, real)
    synth = out["mid"]

    r2_val = r2(synth, real)
    real_rets = np.diff(np.log(real))
    synth_rets = np.diff(np.log(synth))
    ks_stat = float(ks_2samp(synth_rets, real_rets).statistic)

    return {"ticker": tic, "r2": float(r2_val), "ks": ks_stat,
            "n_events": len(out["events"]),
            "max_ok": bool(synth.max() <= h + 1e-6),
            "min_ok": bool(synth.min() >= l - 1e-6)}


def main():
    import yfinance as yf
    daily = yf.download(NIFTY50, period="5d", interval="1d", progress=False, group_by="group")
    results = []
    for tic in NIFTY50:
        try:
            res = evaluate_stock(tic, daily)
            if res:
                results.append(res)
                print(f"{tic:>15}  R2={res['r2']:+.4f}  KS={res['ks']:.4f}  "
                      f"events={res['n_events']}")
        except Exception as e:
            print(f"{tic}: {e.__class__.__name__}")

    df = pd.DataFrame(results)
    print(f"\nStocks evaluated: {len(results)}/50")
    print(f"R2:  mean={df['r2'].mean():.4f}  median={df['r2'].median():.4f}  "
          f"min={df['r2'].min():.4f}  max={df['r2'].max():.4f}")
    print(f"KS:  mean={df['ks'].mean():.4f}  median={df['ks'].median():.4f}  "
          f"min={df['ks'].min():.4f}  max={df['ks'].max():.4f}")
    both = [bool(r["max_ok"]) and bool(r["min_ok"]) for r in results]
    print(f"OHLC conformity: {np.mean(both)*100:.1f}%")


if __name__ == "__main__":
    main()
