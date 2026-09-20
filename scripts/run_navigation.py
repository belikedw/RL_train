#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from rm_sentry_rl.config import load_config
from rm_sentry_rl.envs.omni_navigation_env import OmniNavigationEnv
from rm_sentry_rl.visualization import RvizEnvironment, load_scenario

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="RViz规划调试/策略评估")
    parser.add_argument("--mode", choices=("plan", "follow", "evaluate"), default="plan")
    parser.add_argument("--env-config", type=Path, default=PROJECT_ROOT / "configs/env.yaml")
    parser.add_argument("--session", type=Path, default=PROJECT_ROOT / "artifacts/rviz")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--scenario", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--paused", action="store_true")
    parser.add_argument("--auto-reset", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0, help="非零时限制演示总步数，用于冒烟测试")
    args = parser.parse_args()
    if args.mode == "evaluate" and args.model is None:
        parser.error("evaluate模式需要--model")
    config = load_config(args.env_config)
    scenario = load_scenario(args.scenario, config) if args.scenario else None
    model = None
    if args.mode == "evaluate":
        from stable_baselines3 import PPO
        model = PPO.load(args.model, device="cpu")
    env = RvizEnvironment(OmniNavigationEnv(config), args.session, mode=args.mode,
                          speed=args.speed, paused=args.paused, scenario=scenario)
    try:
        observation, _ = env.reset(seed=args.seed)
        done, steps = False, 0
        while args.max_steps == 0 or steps < args.max_steps:
            env.poll()
            if env.reset_requested:
                observation, _ = env.reset()
                done = False
            if args.mode == "plan" or done or (env.paused and env.single_steps == 0):
                time.sleep(0.02)
                continue
            if model is not None:
                action, _ = model.predict(observation, deterministic=True)
            else:
                # 简单前视跟踪，仅用于验证运动显示；不是MPC教师或安全控制器。
                base = env.base
                index = base._nearest_path_index()
                target_s = base.path_arc_length[index] + 0.35
                target = np.array([np.interp(target_s, base.path_arc_length, base.path[:, k]) for k in (0, 1)])
                delta = target - base.pose[:2]
                c, s = np.cos(base.pose[2]), np.sin(base.pose[2])
                local = np.array([c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]])
                action = np.clip(1.5 * local / base.max_speed, -1.0, 1.0).astype(np.float32)
            observation, _, terminated, truncated, info = env.step(action)
            steps += 1
            done = terminated or truncated
            if done:
                print(f"episode={env.episode} goal={info['goal_reached']} collision={info['collision']}", flush=True)
                if args.auto_reset:
                    observation, _ = env.reset()
                    done = False
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
