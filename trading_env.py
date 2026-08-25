"""Gymnasium trading environment over the Gumbel-Hawkes-Range simulator."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gumbel_hawkes_sim import SimpleHawkes, gumbel_spread, pin_the_range


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, ohlc, n_bars=390, initial_cash=100_000, max_position_value=50_000):
        super().__init__()
        self.open_, self.high, self.low, self.close = map(float, ohlc)
        self.n_bars = n_bars
        self.initial_cash = initial_cash
        self.max_position_value = max_position_value

        self.action_space = spaces.Discrete(3)  # 0=hold, 1=buy, 2=sell/liquidate
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)  # explicit seed only, so parallel-env resets stay independent
        self.mid_ticks = pin_the_range(self.open_, self.high, self.low, self.close, self.n_bars)
        self.hawkes = SimpleHawkes()
        self.step_idx = 0
        self.cash = float(self.initial_cash)
        self.inventory = 0
        self.equity = self.initial_cash
        return self._obs(self.mid_ticks[0]), {"equity": self.equity}

    def _obs(self, mid):
        momentum = (mid - self.open_) / self.open_
        return np.array(
            [mid, momentum, self.hawkes.lambda_t, self.cash, self.inventory, self.equity],
            dtype=np.float32,
        )

    def step(self, action):
        if self.step_idx >= self.n_bars:
            raise RuntimeError("step called after episode terminated (reset required)")
        mid = float(self.mid_ticks[self.step_idx])
        mid = float(np.clip(self.hawkes.step(mid), self.low, self.high))
        momentum = (mid - self.open_) / self.open_
        bid, ask = gumbel_spread(mid, momentum)

        # Baseline equity before the action, marked at this minute's mid
        prev_equity = self.cash + self.inventory * mid

        if action == 1:  # buy with 10% of cash
            shares = int(self.cash * 0.1 / ask)
            if shares > 0:
                self.cash -= shares * ask
                self.inventory += shares
        elif action == 2 and self.inventory > 0:  # liquidate
            self.cash += self.inventory * bid
            self.inventory = 0

        self.equity = self.cash + self.inventory * mid

        # forced liquidation guard
        if self.inventory * mid > self.max_position_value:
            self.cash += self.inventory * bid
            self.inventory = 0
            self.equity = self.cash

        # Reward = change in equity this step (so total reward = final equity - initial cash)
        reward = self.equity - prev_equity
        self.step_idx += 1
        terminated = self.step_idx >= self.n_bars - 1

        return self._obs(mid), float(reward), terminated, False, {"equity": self.equity}


if __name__ == "__main__":
    env = TradingEnv([1304.30, 1317.10, 1300.00, 1317.00])
    rewards = []
    for ep in range(10):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            obs, r, done, _, info = env.step(env.action_space.sample())
        rewards.append(info["equity"])
    print("Random agent over 10 episodes:", np.round(rewards, 2))
    print(f"Mean final equity: ₹{np.mean(rewards):,.2f} (started ₹100,000)")
