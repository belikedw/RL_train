#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rm_sentry_rl.config import environment_fingerprint, load_config
from rm_sentry_rl.envs.omni_navigation_env import OmniNavigationEnv
from rm_sentry_rl.mpc import OmniMpcTeacher
from rm_sentry_rl.observation import ObservationBuilder


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="采集静态地图MPC教师示范")
    parser.add_argument("--env-config", type=Path, default=PROJECT_ROOT / "configs/env.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/mpc_demonstrations.npz")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bridge-binary", type=Path, default=None)
    parser.add_argument("--include-failed", action="store_true", help="同时保存碰撞或超时回合")
    args = parser.parse_args()
    if args.episodes < 1 or args.max_samples < 2:
        parser.error("episodes必须为正数，max-samples至少为2")

    config = load_config(args.env_config)
    env = OmniNavigationEnv(config)
    teacher: OmniMpcTeacher | None = None
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    episode_ids: list[int] = []
    success = 0
    collisions = 0
    timeouts = 0
    try:
        teacher = OmniMpcTeacher(env, bridge_binary=args.bridge_binary)
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode)
            teacher.reset(seed=args.seed + episode)
            episode_observations: list[np.ndarray] = []
            episode_actions: list[np.ndarray] = []
            for _ in range(env.max_steps):
                action = teacher.act()
                episode_observations.append(observation.copy())
                episode_actions.append(action.copy())
                observation, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            reached_safely = bool(info["goal_reached"] and not info["collision"])
            success += int(reached_safely)
            collisions += int(info["collision"])
            timeouts += int(not terminated and truncated)
            accepted = bool(reached_safely or args.include_failed)
            if accepted:
                room = args.max_samples - len(observations)
                observations.extend(episode_observations[:room])
                actions.extend(episode_actions[:room])
                episode_ids.extend([episode] * min(room, len(episode_actions)))
            print(
                f"episode={episode + 1}/{args.episodes}, samples={len(observations)}, "
                f"goal={info['goal_reached']}, collision={info['collision']}, saved={accepted}"
            )
            if len(observations) >= args.max_samples:
                break

        if len(observations) < 2:
            raise RuntimeError("示范数据不足，至少需要两个样本")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            observations=np.asarray(observations, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.float32),
            episode_ids=np.asarray(episode_ids, dtype=np.int32),
            observation_version=np.int32(ObservationBuilder.VERSION),
            environment_fingerprint=environment_fingerprint(config),
            teacher_kind="cpp_omni_mpc_v1",
        )
        print(
            f"已保存 {len(observations)} 条示范到 {args.output}; "
            f"success={success}, collisions={collisions}, timeouts={timeouts}"
        )
    finally:
        if teacher is not None:
            teacher.close()
        env.close()


if __name__ == "__main__":
    main()
