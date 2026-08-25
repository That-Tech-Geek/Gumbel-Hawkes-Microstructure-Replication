"""Runtime regime detection: predict the current regime from a price window."""

import os

import numpy as np

_MODELS = None


def _load_models():
    global _MODELS
    if _MODELS is None and os.path.exists("regime_kmeans.pkl"):
        import joblib
        _MODELS = (joblib.load("regime_scaler.pkl"), joblib.load("regime_kmeans.pkl"))
    return _MODELS


def predict_regime(window_prices: np.ndarray, spread_proxy: float = 0.015,
                   vol_momentum: float = 1.0) -> int | None:
    """Classify a rolling price window into a regime (0 quietest - 3 wildest).

    Uses vol + trend from the window; spread and volume momentum default to
    their cluster medians when unknown. Returns None if models are missing.
    """
    models = _load_models()
    if models is None:
        return None
    scaler, kmeans = models
    prices = np.asarray(window_prices, dtype=float)
    if len(prices) < 5:
        return None
    logr = np.diff(np.log(prices))
    vol = float(np.std(logr) * np.sqrt(390))
    trend = float(np.mean(logr) * 390)
    feats = scaler.transform([[vol, trend, vol_momentum, spread_proxy]])
    return int(kmeans.predict(feats)[0])
