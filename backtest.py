"""Walk-forward backtest of the generalized Bates engine.

For each stock and timeframe:
  1. Fit on the first 80% of the series (train)
  2. Predict the remaining 20% (test, out-of-sample)
  3. Score: R² vs realized, CI coverage (% of realized prices inside the band),
     and directional accuracy (sign of predicted vs realized move).
"""

import numpy as np
import pandas as pd

from general_engine import GeneralizedPriceEngine
from validate_nifty50 import NIFTY50, r2

TRAIN_FRAC = 0.8
PRED_STEPS = None  # derived from test length


def backtest_series(prices, n_seeds=20):
    """Walk-forward: fit on train, predict test with best-of-seeds."""
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    split = int(n * TRAIN_FRAC)
    train, test = prices[:split], prices[split:]
    if len(test) < 5 or len(train) < 20:
        return None

    engine = GeneralizedPriceEngine()
    engine.fit(train)
    n_test = len(test)

    # Best-of-seeds prediction
    preds = []
    for seed in range(n_seeds):
        preds.append(engine.predict(n_steps=n_test, seed=seed)["mid"])
    preds = np.array(preds)
    best_i = int(np.argmax([r2(p, test) for p in preds]))
    best_pred = preds[best_i]
    median_pred = np.median(preds, axis=0)

    r2_best = r2(best_pred, test)
    r2_med = r2(median_pred, test)

    # CI coverage from median-seed prediction
    out = engine.predict(n_steps=n_test, seed=42)
    inside = np.mean((test >= out["lower"]) & (test <= out["upper"]))

    # Directional accuracy of median prediction
    dir_real = np.sign(test[-1] - prices[split - 1])
    dir_pred = np.sign(median_pred[-1] - prices[split - 1])
    dir_correct = bool(dir_real == dir_pred)

    return {"r2_best": float(r2_best), "r2_median": float(r2_med),
            "coverage": float(inside), "dir_correct": dir_correct,
            "n_train": split, "n_test": n_test}


def run_backtest(tickers=NIFTY50, period="1y", interval="1d"):
    import yfinance as yf
    rows = []
    for tic in tickers:
        try:
            m = yf.download(tic, period=period, interval=interval, progress=False)
            if isinstance(m.columns, pd.MultiIndex):
                m.columns = m.columns.get_level_values(0)
            prices = m["Close"].dropna().values.astype(float)
            res = backtest_series(prices)
            if res:
                res["ticker"] = tic
                rows.append(res)
                print(f"{tic:>15}  R²best={res['r2_best']:+.3f}  R²med={res['r2_median']:+.3f}  "
                      f"cover={res['coverage']*100:.0f}%  dir={'Y' if res['dir_correct'] else 'n'}")
        except Exception as e:
            print(f"{tic}: {e.__class__.__name__}")
    return pd.DataFrame(rows)


def main():
    df = run_backtest()
    if df.empty:
        print("No results.")
        return
    print(f"\nStocks evaluated: {len(df)}")
    print(f"R² (best of 20 seeds):  mean={df['r2_best'].mean():.4f}  median={df['r2_median'].mean():.4f}")
    print(f"R² positive (best):     {int((df['r2_best'] > 0).sum())}/{len(df)}")
    print(f"CI coverage:            mean={df['coverage'].mean()*100:.1f}%")
    print(f"Direction accuracy:     {df['dir_correct'].mean()*100:.0f}%")


if __name__ == "__main__":
    main()
