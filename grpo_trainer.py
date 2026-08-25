"""GRPO trainer for the TradingEnv. Group = parallel envs; advantage = (R - mean(R)) / std(R)."""

import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from trading_env import TradingEnv


# --- 1. The Policy Network (Micro-sized for HFT speed) ---
class PolicyNet(nn.Module):
    def __init__(self, obs_dim=6, action_dim=3, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.out = nn.Linear(hidden_dim // 2, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


# --- 2. The GRPO Trainer Class ---
class GRPOTrainer:
    def __init__(self, env_creator, num_envs=8, lr=1e-4, clip_eps=0.2, gamma=1.0, seed=None):
        self.num_envs = num_envs
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.env_creator = env_creator
        self.envs = [env_creator() for _ in range(num_envs)]

        self.policy = PolicyNet()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.episode_rewards = deque(maxlen=100)
        # Snapshot of behavior policy (frozen) so new/old log-prob ratios aren't trivially 1
        self.behavior_policy = PolicyNet()
        self.behavior_policy.load_state_dict(self.policy.state_dict())

    def collect_trajectories(self, steps_per_episode=390):
        seed_rng = np.random.default_rng()
        # Give each parallel env its own seed so group returns vary
        obs_list = []
        for env in self.envs:
            s = int(seed_rng.integers(0, 2**31 - 1))
            obs, _ = env.reset(seed=s)
            obs_list.append(obs)
        obs_batch = torch.tensor(np.array(obs_list), dtype=torch.float32)

        all_obs, all_actions, all_rewards, all_log_probs = [], [], [], []
        env_returns = np.zeros(self.num_envs)

        for step in range(steps_per_episode):
            with torch.no_grad():
                logits = self.behavior_policy(obs_batch)
            dist = Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

            next_obs_list, reward_list, done_list = [], [], []
            for i, env in enumerate(self.envs):
                next_obs, reward, done, truncated, info = env.step(actions[i].item())
                next_obs_list.append(next_obs)
                reward_list.append(reward)
                done_list.append(done)
                env_returns[i] += reward * self.gamma ** step

            all_obs.append(obs_batch)
            all_actions.append(actions)
            all_rewards.append(torch.tensor(reward_list, dtype=torch.float32))
            all_log_probs.append(log_probs)

            obs_batch = torch.tensor(np.array(next_obs_list), dtype=torch.float32)
            if all(done_list):
                break

        return (torch.stack(all_obs), torch.stack(all_actions),
                torch.stack(all_rewards), torch.stack(all_log_probs), env_returns)

    def train_step(self, obs_tensor, action_tensor, reward_tensor, log_prob_tensor, env_returns):
        mean_return = float(np.mean(env_returns))
        std_return = float(np.std(env_returns))
        # If the group collapses to one return (common in low-vol regimes), std→0 kills the gradient.
        # Floor std at 1% of |mean| so tiny spreads still give a meaningful learning signal.
        std_return = max(std_return, 0.01 * abs(mean_return), 1e-3)
        advantages = torch.tensor((env_returns - mean_return) / std_return, dtype=torch.float32)
        advantages = advantages.unsqueeze(0).expand_as(action_tensor)

        old_log_probs = log_prob_tensor.detach()

        T, B, obs_dim = obs_tensor.shape
        obs_flat = obs_tensor.view(T * B, obs_dim)
        actions_flat = action_tensor.view(T * B)

        logits = self.policy(obs_flat)
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions_flat).view(T, B)

        ratio = torch.exp(new_log_probs - old_log_probs)
        adv = advantages.detach()
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        entropy = dist.entropy().mean()
        loss = policy_loss - 0.01 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        # Refresh the frozen behavior snapshot so next trajectory has meaningful old log-probs
        self.behavior_policy.load_state_dict(self.policy.state_dict())

        return loss.item(), policy_loss.item(), entropy.item()

    def train_episodes(self, num_episodes=500):
        for ep in range(num_episodes):
            obs_t, act_t, rew_t, log_t, returns = self.collect_trajectories()
            loss, p_loss, ent = self.train_step(obs_t, act_t, rew_t, log_t, returns)
            mean_return = float(np.mean(returns))
            self.episode_rewards.append(mean_return)

            if ep % 20 == 0 or ep == num_episodes - 1:
                avg_ret = float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0
                print(f"Episode {ep:4d} | Avg Group Return: {mean_return:8.2f} | "
                      f"Running Avg: {avg_ret:8.2f} | Loss: {loss:.4f} | Ent: {ent:.3f}",
                      flush=True)

        rewards = list(self.episode_rewards)
        tail = rewards[-50:] if len(rewards) >= 50 else rewards
        return float(np.mean(tail))

    def save_model(self, path="policy_grpo.pth"):
        torch.save(self.policy.state_dict(), path)
        print(f"Model saved to {path}")


# --- Regime mix: sample across vol regimes so the policy doesn't overfit one day ---
OHLC_POOL = [
    (1304.30, 1317.10, 1300.00, 1317.00),  # RELIANCE-like low-vol day
    (2500.00, 2560.00, 2475.00, 2530.00),  # mid-vol
    (100.00, 105.00, 95.00, 102.00),       # high-vol synthetic
    (1500.00, 1560.00, 1480.00, 1580.00),  # trending
    (800.00, 803.00, 795.00, 797.00),      # flat / range-bound
    (2200.00, 2310.00, 2190.00, 2260.00),  # wide range
]


def env_factory():
    # 60% real Reliance anchor, 40% sampled from regime pool
    if np.random.rand() < 0.6:
        ohlc = (1304.30, 1317.10, 1300.00, 1317.00)
    else:
        idx = int(np.random.randint(len(OHLC_POOL)))
        ohlc = OHLC_POOL[idx]
    return TradingEnv(ohlc, initial_cash=100_000, n_bars=390)


if __name__ == "__main__":
    n_episodes = int(os.environ.get("GRPO_EPISODES", "500"))
    num_envs = int(os.environ.get("GRPO_ENVS", "8"))

    trainer = GRPOTrainer(env_creator=env_factory, num_envs=num_envs, lr=3e-4, clip_eps=0.2)
    print(f"🔥 GRPO Training over {n_episodes} episodes, group size {num_envs}...")

    final50 = trainer.train_episodes(num_episodes=n_episodes)
    print(f"\nFinal avg over last 50 episodes: ₹{final50:,.2f} (random baseline ≈ ₹98,107)")

    trainer.save_model("policy_grpo.pth")
