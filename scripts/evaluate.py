#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from rm_sentry_rl.config import load_config
from rm_sentry_rl.envs.omni_navigation_env import OmniNavigationEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="确定性评估PPO控制策略")
    parser.add_argument("model", type=Path)
    parser.add_argument("--env-config", type=Path, default=PROJECT_ROOT / "configs/env.yaml")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10000)
    args = parser.parse_args()

    config = load_config(args.env_config)
    env = OmniNavigationEnv(config)
    model = PPO.load(args.model, device="cpu")
    metrics: dict[str, list[float]] = defaultdict(list)

    for episode in range(args.episodes):
        observation, _ = env.reset(seed=args.seed + episode)
        terminated = False
        truncated = False
        episode_reward = 0.0
        min_clearance = np.inf
        path_errors: list[float] = []
        final_info = {}

        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, final_info = env.step(action)
            episode_reward += float(reward)
            min_clearance = min(min_clearance, float(final_info["clearance"]))
            path_errors.append(float(final_info["path_error"]))

        metrics["reward"].append(episode_reward)
        metrics["success"].append(float(final_info.get("goal_reached", False)))
        metrics["collision"].append(float(final_info.get("collision", False)))
        metrics["min_clearance"].append(min_clearance)
        metrics["path_rmse"].append(float(np.sqrt(np.mean(np.square(path_errors)))))

    print(f"episodes:       {args.episodes}")
    print(f"success_rate:   {np.mean(metrics['success']):.3f}")
    print(f"collision_rate: {np.mean(metrics['collision']):.3f}")
    print(f"reward_mean:    {np.mean(metrics['reward']):.3f}")
    print(f"path_rmse:      {np.mean(metrics['path_rmse']):.3f} m")
    print(f"min_clearance:  {np.mean(metrics['min_clearance']):.3f} m")
    env.close()


if __name__ == "__main__":
    main()
