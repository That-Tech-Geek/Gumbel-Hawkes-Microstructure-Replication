"""Tunable Perlin noise generator for smooth parameter perturbations.

Perlin noise gives smooth, continuous, pseudo-random variation. We use it to
drift engine parameters (volatility, jump intensity, spread) smoothly over a
path — instead of keeping them constant or drawing white noise each step.

Tunable via octaves (smoothness), persistence (roughness), and seed.
"""

import numpy as np


def fade(t):
    """Smooth interpolation curve 6t^5 - 15t^4 + 10t^3."""
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin_1d(n, octaves=4, persistence=0.5, seed=0, base_freq=4.0):
    """1D Perlin noise in [-1, 1]. Tunable smoothness via octaves/persistence.

    octaves: more octaves = finer detail layered on top
    persistence: amplitude falloff per octave (lower = smoother)
    base_freq: cycles of the lowest octave across the path
    """
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    amp, freq = 1.0, base_freq
    total_amp = 0.0
    for _ in range(octaves):
        n_p = int(freq) + 2
        grads = rng.uniform(-1, 1, n_p)
        x = np.linspace(0, freq, n)
        i = np.minimum(np.floor(x).astype(int), n_p - 2)
        t = x - np.floor(x)
        g0, g1 = grads[i], grads[i + 1]
        val = g0 * (1 - fade(t)) + g1 * fade(t)
        out += amp * val
        total_amp += amp
        amp *= persistence
        freq *= 2
    return out / (total_amp + 1e-9)


class PerlinTuner:
    """Applies smooth Perlin perturbation to scalar engine parameters."""

    def __init__(self, octaves=4, persistence=0.5, seed=0, base_freq=4.0,
                 min_scale=0.2, max_scale=2.0):
        self.octaves = octaves
        self.persistence = persistence
        self.seed = seed
        self.base_freq = base_freq
        self.min_scale = min_scale
        self.max_scale = max_scale

    def series(self, n):
        """Return a smooth multiplier series clamped to [min,max]."""
        p = perlin_1d(n, self.octaves, self.persistence, self.seed, self.base_freq)
        # Map [-1,1] -> [min_scale, max_scale]
        span = (self.max_scale - self.min_scale) / 2
        mid = (self.max_scale + self.min_scale) / 2
        return mid + span * np.clip(p, -1, 1)

    def apply(self, base_value, n):
        base = np.asarray(base_value, dtype=float)
        return base * self.series(n)


if __name__ == "__main__":
    pt = PerlinTuner(octaves=4, persistence=0.5, seed=0)
    s = pt.series(12)
    print("Perlin multipliers (smooth):", np.round(s, 3))
    print("mean ~", round(float(np.mean(s)), 3), " | std:", round(float(np.std(s)), 3))
