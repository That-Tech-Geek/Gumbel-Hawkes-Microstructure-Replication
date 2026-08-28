"""Explore optimal EWMA lookback across NIFTY 50 stocks.

For each stock: optimize EWMA half-life on returns, then compare engine fit
(EWMA vs flat) on out-of-sample prediction accuracy and CI calibration.
"""

import numpy as np
import pandas as pd

from general_engine import GeneralizedPriceEngine
from ewma_analytics import optimize_lookback
from validate_nifty50 import NIFTY50, r2

TAUS = np.logspace(np.log10(3), np.log10(120), 10)  # 3 to 120 bars


def study_stock(tic, prices):
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    split = int(n * 0.8)
    train, test = prices[:split], prices[split:]
    rets = np.diff(np.log(train))
    if len(rets) < 30 or len(test) < 5:
        return None

    best, table = optimize_lookback(rets, taus=TAUS)

    # Engine with optimized EWMA tau
    eng_e = GeneralizedPriceEngine()
    eng_e.params.ewma_tau = best.tau
    eng_e.fit(train)
    pred_e = eng_e.predict(n_steps=len(test), seed=42)

    # Engine flat (no EWMA)
    eng_f = GeneralizedPriceEngine()
    eng_f.fit(train)
    pred_f = eng_f.predict(n_steps=len(test), seed=42)

    # CI coverage
    cov_e = float(np.mean((test >= pred_e["lower"]) & (test <= pred_e["upper"])))
    cov_f = float(np.mean((test >= pred_f["lower"]) & (test <= pred_f["upper"])))

    # CI width (tighter is better if coverage is similar)
    w_e = float(np.mean((pred_e["upper"] - pred_e["lower"]) / pred_e["mid"]))
    w_f = float(np.mean((pred_f["upper"] - pred_f["lower"]) / pred_f["mid"]))

    return {"ticker": tic, "best_tau": best.tau,
            "ewma_vol": best.vol, "ewma_lam": best.lam,
            "cov_ewma": cov_e, "cov_flat": cov_f,
            "width_ewma": w_e, "width_flat": w_f,
            "r2_ewma": r2(pred_e["mid"], test), "r2_flat": r2(pred_f["mid"], test)}


def main():
    import yfinance as yf
    rows = []
    for tic in NIFTY50:
        try:
            m = yf.download(tic, period="1y", interval="1d", progress=False)
            if isinstance(m.columns, pd.MultiIndex):
                m.columns = m.columns.get_level_values(0)
            prices = m["Close"].dropna().values.astype(float)
            res = study_stock(tic, prices)
            if res:
                rows.append(res)
                print(f"{tic:>15}  tau={res['best_tau']:5.1f}  "
                      f"cov: {res['cov_ewma']*100:.0f}%/{res['cov_flat']*100:.0f}%  "
                      f"width: {res['width_ewma']*100:.1f}%/{res['width_flat']*100:.1f}%  "
                      f"R2: {res['r2_ewma']:+.3f}/{res['r2_flat']:+.3f}")
        except Exception as e:
            print(f"{tic}: {e.__class__.__name__}")

    df = pd.DataFrame(rows)
    print(f"\nStocks: {len(df)}")
    print(f"Optimal tau: mean={df['best_tau'].mean():.1f}  median={df['best_tau'].median():.1f}")
    print(f"EWMA vs flat: cov {df['cov_ewma'].mean()*100:.0f}%/{df['cov_flat'].mean()*100:.0f}%  "
          f"width {df['width_ewma'].mean()*100:.1f}%/{df['width_flat'].mean()*100:.1f}%  "
          f"R2 {df['r2_ewma'].mean():+.3f}/{df['r2_flat'].mean():+.3f}")


if __name__ == "__main__":
    main()
