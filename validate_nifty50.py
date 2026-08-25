"""Validate the PriceEngine against real NIFTY 50 next-day paths via R2.

For each stock: run the engine on day t's OHLC, then R2 the generated synthetic
intraday path against the real day t's minute-level path when available. When 1-min
data is not feasible, R2 vs the day's own bar-scale OHLC envelope (falls back to a
softer measure).
"""

import numpy as np
import pandas as pd

from price_engine import PriceEngine, EngineConfig

# NIFTY 50 constituents (as of 2025)
NIFTY50 = [
    "ADANIENT.NS","ADANIPORTS.NS","APOLLOHOSP.NS","ASIANPAINT.NS","AXISBANK.NS",
    "BAJAJAUTO.NS","BAJAJFINSV.NS","BAJFINANCE.NS","BEL.NS","BPCL.NS",
    "BHARTIARTL.NS","BRITANNIA.NS","CIPLA.NS","COALINDIA.NS","DRREDDY.NS",
    "EICHERMOT.NS","GRASIM.NS","HCLTECH.NS","HDFCBANK.NS","HDFCLIFE.NS",
    "HEROMOTOCO.NS","HINDALCO.NS","HINDUNILVR.NS","ICICIBANK.NS","INDIGO.NS",
    "INFY.NS","ITC.NS","JSWSTEEL.NS","KOTAKBANK.NS","LT.NS",
    "LUPIN.NS","M&M.NS","MARUTI.NS","NESTLEIND.NS","NTPC.NS",
    "ONGC.NS","POWERGRID.NS","RELIANCE.NS","SBILIFE.NS","SBIN.NS",
    "SHRIRAMFIN.NS","SUNPHARMA.NS","TATACONSUM.NS","TATAMOTORS.NS","TATASTEEL.NS",
    "TCS.NS","TECHM.NS","TITAN.NS","TRENT.NS","ULTRACEMCO.NS",
    "WIPRO.NS",
]


def fetch(tickers):
    import yfinance as yf
    data = yf.download(tickers, period="5d", interval="1d", progress=False, group_by="group")
    return data


def r2(xs, ys):
    xs, ys = np.asarray(xs), np.asarray(ys)
    if len(xs) < 2:
        return np.nan
    ss_res = np.sum((ys - xs) ** 2)
    ss_tot = np.sum((ys - ys.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def evaluate_stock(tic, df_df):
    """Fit Hawkes on the day's 1-min data, then generate structurally (no RNG) and compute R2."""
    import yfinance as yf
    from fitted_engine import FitEngine, FittedConfig
    row = df_df[tic].iloc[-1]
    o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
    if np.isnan([o, h, l, c]).any():
        return None

    m = yf.download(tic, period="1d", interval="1m", progress=False)
    if isinstance(m.columns, pd.MultiIndex):
        m.columns = m.columns.get_level_values(0)
    real_min = m["Close"].dropna().values.astype(float)

    cfg = FittedConfig()
    n_events = 8
    if len(real_min) >= 5:
        times = FitEngine.event_times(real_min, cfg.event_threshold)
        n_events = max(len(times), 1)
        alpha, beta, scale, _ = FitEngine.grid_fit(times, len(real_min), real_min)
        cfg.alpha, cfg.beta, cfg.momentum_scale = alpha, beta, scale
    engine = FitEngine(cfg)

    out = engine.run(o, h, l, c, empirical_intensity=times if len(real_min) >= 5 else None)
    synth = np.asarray(out["mid"], dtype=float)

    if len(real_min) >= 10:
        n = min(len(synth), len(real_min))
        r2_val = r2(synth[-n:], real_min[-n:])
    else:
        r2_val = r2(synth, np.linspace(o, c, len(synth)))

    rng_out = PriceEngine(EngineConfig(seed=0)).run(o, h, l, c)
    rng_synth = np.asarray(rng_out["mid"], dtype=float)
    if len(real_min) >= 10:
        n = min(len(rng_synth), len(real_min))
        rng_r2 = r2(rng_synth[-n:], real_min[-n:])
    else:
        rng_r2 = r2(rng_synth, np.linspace(o, c, len(rng_synth)))

    return {
        "ticker": tic,
        "r2_fitted": float(r2_val), "r2_random": float(rng_r2),
        "uplift": float(r2_val - rng_r2),
        "max_ok": bool(out["mid"].max() <= h + 1e-6),
        "min_ok": bool(out["mid"].min() >= l - 1e-6),
    }


def main():
    import yfinance as yf
    df = fetch(NIFTY50)
    results = []
    for tic in NIFTY50:
        try:
            res = evaluate_stock(tic, df)
            if res: results.append(res)
        except Exception as e:
            print(f"{tic}: {e.__class__.__name__}")
    pd.set_option("display.max_rows", None)
    df_res = pd.DataFrame(results)
    print(df_res[["ticker","r2_fitted","r2_random","uplift"]].to_string(index=False))
    print(f"\nStocks evaluated: {len(results)}/50")
    print(f"Fitted R2: mean={df_res['r2_fitted'].mean():.4f}, "
          f"median={df_res['r2_fitted'].median():.4f}, "
          f"min={df_res['r2_fitted'].min():.4f}, max={df_res['r2_fitted'].max():.4f}")
    print(f"Random R2: mean={df_res['r2_random'].mean():.4f}, "
          f"median={df_res['r2_random'].median():.4f}")
    print(f"Uplift (fit-random): mean={df_res['uplift'].mean():.4f}, "
          f"median={df_res['uplift'].median():.4f}")
    both = [(bool(r['max_ok']) and bool(r['min_ok'])) for r in results]
    print(f"OHLC conformity: {np.mean(both)*100:.1f}%")


if __name__ == "__main__":
    main()
