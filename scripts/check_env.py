#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3.common.env_checker import check_env

from rm_sentry_rl.config import load_config
from rm_sentry_rl.envs.omni_navigation_env import OmniNavigationEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="检查Gymnasium环境接口并执行随机动作冒烟测试")
    parser.add_argument("--env-config", type=Path, default=PROJECT_ROOT / "configs/env.yaml")
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    config = load_config(args.env_config)
    env = OmniNavigationEnv(config)
    check_env(env, warn=True)

    observation, info = env.reset(seed=42)
    print(
        "环境检查通过: "
        f"observation={observation.shape}, "
        f"map={info['map_width']}x{info['map_height']}, "
        f"resolution={info['map_resolution']:.3f}m"
    )

    resets = 0
    for _ in range(args.steps):
        action = env.action_space.sample().astype(np.float32)
        observation, _, terminated, truncated, _ = env.step(action)
        if not np.all(np.isfinite(observation)):
            raise RuntimeError("观测中出现NaN或Inf")
        if terminated or truncated:
            observation, _ = env.reset()
            resets += 1

    print(f"随机动作冒烟测试通过: steps={args.steps}, resets={resets}")
    env.close()


if __name__ == "__main__":
    main()
