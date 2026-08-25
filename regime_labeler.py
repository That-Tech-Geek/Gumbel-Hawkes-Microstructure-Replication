"""Label intraday regimes via K-Means over realized 1-min statistics.

Fetches recent 1-min NSE data, computes 4 features per (stock, day):
  - vol:          std(log returns) * sqrt(390)
  - trend:        mean(log returns) * 390
  - vol_momentum: today's total volume / mean of prior days' volumes
  - spread:       (day high - day low) / day mean price   [Gumbel proxy]

Fits StandardScaler + KMeans(4), persists both models, and prints a
per-regime summary so multipliers can be calibrated from real stats.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from validate_nifty50 import NIFTY50

REGIME_NAMES = {0: "quiet_range", 1: "bull_trend", 2: "bear_crash", 3: "high_vol_chop"}


def day_features(day_df: pd.DataFrame, prior_day_volumes: list[float]) -> pd.Series | None:
    """Feature row for one (stock, day) 1-min frame."""
    close = day_df["Close"].dropna()
    if len(close) < 30:
        return None
    logr = np.diff(np.log(close.values))
    vol = float(np.std(logr) * np.sqrt(390))
    trend = float(np.mean(logr) * 390)
    today_vol = float(day_df["Volume"].sum())
    prior = float(np.mean(prior_day_volumes)) if prior_day_volumes else today_vol
    vol_momentum = today_vol / (prior + 1e-9)
    spread_proxy = float((day_df["High"].max() - day_df["Low"].min()) / (close.mean() + 1e-9))
    return pd.Series({"vol": vol, "trend": trend,
                      "vol_momentum": vol_momentum, "spread": spread_proxy})


def build_feature_table(tickers=NIFTY50, period="5d") -> pd.DataFrame:
    import yfinance as yf
    rows = []
    for tic in tickers:
        try:
            m = yf.download(tic, period=period, interval="1m", progress=False)
            if isinstance(m.columns, pd.MultiIndex):
                m.columns = m.columns.get_level_values(0)
            if len(m) < 100:
                continue
            m = m.dropna(subset=["Close"])
            days = [g for _, g in m.groupby(m.index.normalize())]
            prior_vols: list[float] = []
            for day_df in days:
                feat = day_features(day_df, prior_vols)
                if feat is not None:
                    feat["ticker"] = tic
                    rows.append(feat)
                prior_vols.append(float(day_df["Volume"].sum()))
            print(f"{tic}: {len(days)} days")
        except Exception as e:
            print(f"{tic}: {e.__class__.__name__}")
    return pd.DataFrame(rows)


def fit_regimes(features: pd.DataFrame, n_clusters: int = 4):
    X = features[["vol", "trend", "vol_momentum", "spread"]].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(Xs)
    features = features.copy()
    features["regime"] = labels

    # Order clusters by volatility so regime 0 = quietest, regime 3 = wildest.
    # Reorder the centers (not just labels_) so predict() emits consistent indices.
    order = features.groupby("regime")["vol"].mean().sort_values().index.tolist()
    remap = {old: new for new, old in enumerate(order)}
    features["regime"] = features["regime"].map(remap)
    kmeans.cluster_centers_ = kmeans.cluster_centers_[order]
    kmeans.labels_ = features["regime"].values
    return features, scaler, kmeans


def main():
    features = build_feature_table()
    print(f"\nTotal (stock, day) samples: {len(features)}")
    features, scaler, kmeans = fit_regimes(features)

    summary = features.groupby("regime")[["vol", "trend", "vol_momentum", "spread"]].agg(
        ["mean", "count"]).round(4)
    print("\nRegime summary:")
    print(summary.to_string())

    joblib.dump(scaler, "regime_scaler.pkl")
    joblib.dump(kmeans, "regime_kmeans.pkl")
    features.to_csv("regime_labels.csv", index=False)
    print("\nSaved: regime_scaler.pkl, regime_kmeans.pkl, regime_labels.csv")


if __name__ == "__main__":
    main()
