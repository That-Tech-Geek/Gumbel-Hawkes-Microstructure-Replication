"""Fit Hawkes (alpha, beta) to real 1-minute Nifty data via maximum likelihood."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def hawkes_log_likelihood(params, times, T):
    alpha, beta = params
    if alpha < 0 or beta < 0:
        return 1e9
    intensity = 0.1
    ll = 0.0
    for i, t_i in enumerate(times):
        intensity = 0.1 + sum(alpha * np.exp(-beta * (t_i - t_j)) for t_j in times[:i])
        ll += np.log(intensity + 1e-9)
    integral = 0.1 * T + (alpha / (beta + 1e-12)) * sum(
        1 - np.exp(-beta * (T - t_j)) for t_j in times
    )
    return -ll + integral


def event_times_from_minute_prices(prices, threshold=0.001):
    """Minutes (from session start) where |return| > threshold (default 0.1%)."""
    rets = np.abs(np.diff(np.log(np.asarray(prices, dtype=float))))
    return np.where(rets > threshold)[0] + 1.0


def fit_one_ticker(ticker, period="5d", interval="1m"):
    import yfinance as yf
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    closes = data["Close"].dropna().values
    if len(closes) < 30:
        return None
    times = event_times_from_minute_prices(closes)
    T = len(closes)
    if len(times) < 5:
        return (0.0, 0.0)
    res = minimize(hawkes_log_likelihood, [0.3, 0.1], args=(times, T),
                   method="Nelder-Mead")
    return tuple(res.x)


def main(tickers=("RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "SBIN.NS")):
    alphas, betas = [], []
    for tic in tickers:
        try:
            ab = fit_one_ticker(tic)
            if ab is not None:
                alphas.append(ab[0]); betas.append(ab[1])
                print(f"{tic:15s} alpha={ab[0]:.4f}  beta={ab[1]:.4f}")
        except Exception as e:
            print(f"{tic}: {e.__class__.__name__}")
    if alphas:
        print(f"\nNifty baseline -> alpha_mean={np.mean(alphas):.4f}, beta_mean={np.mean(betas):.4f}")
    else:
        print("No 1m data; using synthetic fallback alpha=0.5 beta=0.1")


if __name__ == "__main__":
    main()
