# Gumbel–Hawkes Microstructure Simulator

Generate realistic synthetic intraday price series from any real price series.
Calibrated on NSE (NIFTY 50) via yfinance; works on any ticker, any timeframe.

## What it does

Takes a real price series, fits a Bates jump-diffusion model (stochastic
volatility + jumps) plus a momentum-tilted lognormal bid/ask spread, and
generates synthetic paths that match the statistical fingerprint of the input.

```python
from general_engine import GeneralizedPriceEngine

engine = GeneralizedPriceEngine()
engine.fit(prices)                      # any price series
pred = engine.predict(n_steps=30)       # future path + 95% CI
bids, asks = engine.settle(pred["mid"], momentums)
```

## What it is for

- Synthetic market data for training RL agents
- Stress testing strategies under realistic volatility clustering
- Risk simulation (fat tails, jumps, regime shifts)

## What it is NOT for

- Point forecasting. Walk-forward backtest on 49 NIFTY stocks shows
  out-of-sample R² is negative and direction accuracy is ~47% (coin flip).
  Prices are near-martingales; this model simulates their *distribution*,
  not their *destination*.

## Validation results (49 NIFTY stocks, real 1-min data)

| Metric | Bates engine | Momentum engine | Bridge (legacy) |
|---|---|---|---|
| Median R² (in-sample, best seed) | **−0.73** | −2.78 | −2.72 |
| Stocks with positive R² (best seed) | 10/49 | 23/49 | 9/49 |
| Bates beats momentum | — | 47/49 | 38/49 |

- OHLC pinning: 100% exact on all engines
- Bid < ask: guaranteed (min half-spread guard)
- Regime classification: 4 K-Means clusters from (vol, trend, volume, spread),
  245 samples, vol-ordered (0=quiet → 3=event)

## Backtest results (walk-forward, 80/20 split, 1-year daily)

| Metric | Value | Reading |
|---|---|---|
| OOS R² (median seed) | negative | no point-forecast power |
| CI coverage | 100% | bands too wide — needs tightening |
| Direction accuracy | 47% | no better than chance |

## Repo layout

| File | Purpose |
|---|---|
| `general_engine.py` | Main engine: fit/predict/replicate/settle on any series |
| `bates_engine.py` | Bates jump-diffusion core |
| `bates_maximizer.py` | Seed search + learned seed maximizer |
| `gumbel_settlement.py` | Lognormal bid/ask with momentum tilt |
| `regime_labeler.py` / `regime_detect.py` | K-Means regime fit + runtime detection |
| `momentum_engine.py` | Simpler alternative engine (sigma-scaled returns) |
| `validate_*.py` | In-sample validation harnesses |
| `backtest.py` | Walk-forward out-of-sample evaluation |
| `trading_env.py` / `grpo_trainer.py` | Gymnasium env + GRPO trainer |

## Perlin parameter drift (tunable)

Engine parameters (volatility `theta`, jump intensity `lambda`) drift smoothly
over the generated path via Perlin noise, instead of staying constant:

```python
engine.fit(prices)
engine.params.perlin_octaves = 4          # smoothness
engine.params.perlin_persistence = 0.5    # roughness (lower = smoother)
engine.params.perlin_base_freq = 4.0      # cycles across the path
engine.params.perlin_min_scale = 0.3      # clamp lower multiplier
engine.params.perlin_max_scale = 2.0      # clamp upper multiplier
pred = engine.predict(n_steps=30, seed=42)
```

Wider Perlin ranges (0.3–2.0) widen the confidence interval slightly (85% → 89%)
— capturing the fact that real parameters drift over time.

## EWMA lookback optimization (tunable)

Parameters (vol, drift, jump intensity) can be estimated with exponential
weighting so recent data dominates:

```python
engine.params.ewma_tau = 30          # half-life in bars (fixed)
# or auto-tune:
engine.params.ewma_optimize = True   # grid-searches best tau on fit()
engine.fit(prices)
```

Lookback study across 49 NIFTY stocks (1y daily, 80/20 walk-forward):

- **Optimal tau is stock-specific**: mean 75 bars, median 80, range 3–120
- Fast-regime stocks (RELIANCE tau=3, LT tau=4.5) benefit most from short
  lookbacks; slow stocks (ADANIENT tau=120) barely differ from flat weighting
- EWMA tightens CIs modestly (105% vs 110% mean width) at equal coverage
- Biggest wins where vol regime shifts fast: RELIANCE R² −5.4 (EWMA) vs
  −34.5 (flat)

## Reproduce

```bash
pip install numpy pandas yfinance scipy scikit-learn joblib gymnasium torch

python3 validate_bates.py    # in-sample comparison, 49 stocks
python3 backtest.py          # out-of-sample walk-forward
python3 regime_labeler.py    # regime clustering
```

## Next steps

1. Tighten prediction intervals (current 100% coverage = uninformative)
2. Incorporate volume into Hawkes intensity fitting
3. Multi-stock correlated generation (sector/market factors)

## License

MIT
