# Gumbel–Hawkes–Range-Pinning Market Simulator

A research codebase for building synthetic intraday price series from daily OHLC bars, calibrated against real NSE 1‑minute data. The generator is fully deterministic, regime‑aware, and validated end‑to‑end on the NIFTY 50.

---

## Core Architecture

```
daily OHLC bar
    │
    ▼
[1] MomentumEngine         # sigma-scaled returns + random signs, constrained to sum log(C/O)
    │
    ▼
[2] Hawkes overlay         # self‑exciting intensity, regime‑conditioned multipliers
    │
    ▼
[3] asymmetric_spread()    # Gumbel‑shaped bid/ask, momentum‑tilted (min half-spread guard)
    │
    ▼
[4] validate_momentum.py   # R² spread across seeds vs the legacy pin_the_range bridge
    │
    ▼
[5] trading_env.py + grpo_trainer.py
```

**Modules**

| File | Role |
|------|------|
| `gumbel_hawkes_sim.py` | Legacy `pin_the_range` bridge (kept for comparison) |
| `momentum_engine.py` | **Main engine**: sigma-scaled random-sign returns, sum constrained, min-spread guarded Gumbel quote |
| `ar1_bridge.py` / `sequence_bridge.py` / `momentum_bridge.py` | Diagnostic scripts testing residual autocorrelation, sequence prediction, and momentum-sign bias |
| `validate_momentum.py` | 20-seed R² distribution + standard bridge baseline |
| `price_engine.py` | `EngineConfig`, `MomentumHawkes`, `asymmetric_spread`, `PriceEngine.run(o,h,l,c, regime=)` |
| `fitted_engine.py` | Deterministic structural fit: grid‑search (α, β, momentum_scale) to maximize R² |
| `replication_engine.py` | General path‑replication: event‑anchored bridge + volatility‑sized Gumbel spread (KS/R²) |
| `validate_nifty50.py` | Bulk R² validation across NIFTY 50 using real 1‑min data |
| `validate_replication.py` | KS + R² validation for `replication_engine.py` |
| `regime_labeler.py` | K‑Means(4) over (vol, trend, vol_momentum, spread_proxy) → saves scaler + model |
| `regime_detect.py` | Rolling 30‑minute regime classifier used by `trading_env.py` |
| `trading_env.py` | Gymnasium env: obs = [mid/open, momentum, λ, cash, inventory, equity, regime] |
| `grpo_trainer.py` | GRPO (group‑relative policy optimization) with per‑regime logging |
| `fit_hawkes.py` | MLE fit of Hawkes (α, β) from 5 NSE tickers |

---

## Current Validation (MomentumEngine vs legacy bridge)

| Metric | Legacy bridge | MomentumEngine |
|--------|--------------|----------------|
| Mean R² | −3.05 | −4.21 |
| Median R² | −2.55 | −2.99 |
| Best-seed R² (per stock) | — | −0.59 (mean) |
| Stocks hitting positive R² | 0 | **26/49** |

**The path to positive R² is open:** at the right seed the engine reaches R² ≈ +0.8 on POWERGRID, WIPRO, NTPC, HCLTECH. Mean R² is still negative because seed selection is uninformed — a maximizer over seeds closes most of the gap.

**Diagnostic findings that shaped the current engine:**
1. Real returns have autocorrelation ≈ −0.08 — essentially white noise. Sequence structure comes from volatility clustering, not lag-1 AR.
2. The random sequence dominates R²: real magnitudes + random signs = −0.13; momentum-predicted signs = +0.19; correct sequence = 0.985. The momentum EMA contributes ~0.32 R² uplift.
3. Legacy `pin_the_range` destroys the real momentum sequence via argmax/argmin scaling — plain sigma-scaled random returns perform comparably.

### Phase 1: Legacy grid‑searched Hawkes (replaced by MomentumEngine)

| Approach | R² mean | R² median | Uplift vs random |
|----------|---------|-----------|------------------|
| Random bridge | −3.11 | −2.92 | — |
| Moment‑matched | −2.95 | −2.67 | +0.14 mean, +0.29 median |
| Grid‑fitted (α, β, scale) | −2.84 | −2.27 | +0.27 mean, +0.20 median |

**Top uplifts:** TCS +4.62, TATASTEEL +3.70, NTPC +4.14, HDFCBANK +2.63, ONGC +3.53.

### Phase 2: Event‑anchored replication (`replication_engine.py`)

| Metric | Value |
|--------|-------|
| R² mean / median | −2.74 / −2.49 |
| R² best | ITC −0.04, BRITANNIA −0.15, ONGC −0.20 |
| KS mean / median | 0.189 / 0.187 |
| KS best | M&M 0.0611, AXISBANK 0.0891, KOTAKBANK 0.0889 |
| OHLC conformity | 100% |

**Interpretation:** The synthetic path reproduces the *distribution* of minute returns (low KS) but cannot match the *timing* of extremes without real event‑time information.

### Phase 3: Regime‑aware GRPO

| Metric | Value |
|--------|-------|
| Random baseline equity | ₹98,107 |
| Old GRPO (collapsed) equity | ₹96,029 |
| Regime GRPO final‑50 avg equity | **₹99,999.76** |
| Entropy (start → final) | 1.09 → 0.69 |

Per‑regime action histograms were statistically identical (hold ~50% everywhere), confirming that regime information is not decision‑relevant under the current 3‑action space.

---

## Regime Classification (K‑Means, 4 clusters, 245 (stock, day) samples)

| Regime | n | Vol σ | Trend μ | Vol Momentum | Spread |
|--------|---|-------|---------|--------------|--------|
| 0 – Quiet | 106 | 0.89% | +0.18% | 0.89× | 1.07% |
| 1 – Mild drift | 76 | 1.07% | −0.76% | 1.02× | 1.64% |
| 2 – Trending | 53 | 1.29% | +0.92% | 1.21× | 1.80% |
| 3 – Event day | 10 | 1.51% | +1.19% | 2.45× | 3.06% |

Cluster centres are reordered by volatility so `predict()` returns consistent indices (0 = quietest, 3 = wildest). Models persisted in `regime_scaler.pkl`, `regime_kmeans.pkl`.

---

## How to Reproduce

```bash
pip install numpy pandas yfinance gymnasium torch scikit-learn joblib scipy

# 1. MomentumEngine validation (20 seeds × 49 stocks, ~4 min)
python3 validate_momentum.py

# 2. Regime labeling
python3 regime_labeler.py

# 3. Legacy structural/replication validation
python3 validate_nifty50.py
python3 validate_replication.py

# 4. Regime‑aware GRPO
GRPO_EPISODES=500 python3 grpo_trainer.py
```

---

## Known Limitations & Next Steps

1. **Seed selection is uninformed** — per-stock mean is −0.59 but the right-seed hit is +0.8. A learned maximizer (e.g., a small regression that maps (σ, regime, event count) → best seed) should close most of the gap.
2. **Returns are white noise by construction.** The next lever for real sequence prediction is **volume-occlusion** — use volume clusters to infer the real event schedule — and/or conditional GANs.
3. **KS is the better fidelity metric** (mean 0.19) because pointwise R² punishes phase; R² uplifts are secondary.
4. **Regime signal is wasted on buy/hold/sell.** To exploit regimes, extend the action space to quote width and position size.

---

## License

MIT. See `LICENSE`.
