#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sqlite3
from functools import partial
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from rm_sentry_rl.config import load_config
from rm_sentry_rl.envs.omni_navigation_env import OmniNavigationEnv
from rm_sentry_rl.imitation import load_demonstrations, pretrain_policy
from rm_sentry_rl.visualization import load_scenario, make_training_environment
from rm_sentry_rl.visualization_transport import VisualizationChannel


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练RM哨兵全向底盘PPO控制策略")
    parser.add_argument("--env-config", type=Path, default=PROJECT_ROOT / "configs/env.yaml")
    parser.add_argument("--train-config", type=Path, default=PROJECT_ROOT / "configs/ppo.yaml")
    parser.add_argument("--run-name", default="ppo_omni_v1")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--mpc-dataset", type=Path, default=None, help="MPC示范NPZ，先做行为克隆再训练PPO")
    parser.add_argument("--bc-epochs", type=int, default=None)
    parser.add_argument("--rviz", action="store_true", help="发布第0号环境并接收RViz控制")
    parser.add_argument("--rviz-session", type=Path, default=PROJECT_ROOT / "artifacts/rviz")
    parser.add_argument("--speed", type=float, default=0.0, help="可视化环境运行倍率，0为不限速")
    parser.add_argument("--scenario", type=Path, help="从RViz保存的起终点场景")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    environment_config = load_config(args.env_config)
    training_config = load_config(args.train_config)
    ppo_config = training_config["ppo"]

    requested_envs = args.num_envs or int(training_config["num_envs"])
    cpu_limit = max(1, (os.cpu_count() or 2) - 1)
    num_envs = max(1, min(requested_envs, cpu_limit))
    seed = int(training_config["seed"])
    total_timesteps = args.timesteps or int(training_config["total_timesteps"])

    run_directory = PROJECT_ROOT / "artifacts" / args.run_name
    checkpoint_directory = run_directory / "checkpoints"
    best_model_directory = run_directory / "best_model"
    log_directory = run_directory / "monitor"
    for directory in (checkpoint_directory, best_model_directory, log_directory):
        directory.mkdir(parents=True, exist_ok=True)

    vector_class = SubprocVecEnv if num_envs > 1 else DummyVecEnv
    scenario = load_scenario(args.scenario, environment_config) if args.scenario else None
    train_env = vector_class([
        partial(make_training_environment, environment_config, rank, str(log_directory),
                str(args.rviz_session) if args.rviz else None, args.speed, scenario)
        for rank in range(num_envs)
    ])
    train_env.seed(seed)
    eval_env = make_vec_env(
        lambda: OmniNavigationEnv(environment_config),
        n_envs=1,
        seed=seed + 10_000,
        vec_env_cls=vector_class,
    )

    checkpoint_steps = max(1, int(training_config["checkpoint_frequency"]) // num_envs)
    evaluation_steps = max(1, int(training_config["evaluation_frequency"]) // num_envs)
    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=checkpoint_steps,
                save_path=str(checkpoint_directory),
                name_prefix=args.run_name,
            ),
            EvalCallback(
                eval_env,
                best_model_save_path=str(best_model_directory),
                log_path=str(run_directory / "evaluation"),
                eval_freq=evaluation_steps,
                n_eval_episodes=int(training_config["evaluation_episodes"]),
                deterministic=True,
            ),
        ]
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=float(ppo_config["learning_rate"]),
        n_steps=int(ppo_config["n_steps"]),
        batch_size=int(ppo_config["batch_size"]),
        n_epochs=int(ppo_config["n_epochs"]),
        gamma=float(ppo_config["gamma"]),
        gae_lambda=float(ppo_config["gae_lambda"]),
        clip_range=float(ppo_config["clip_range"]),
        ent_coef=float(ppo_config["ent_coef"]),
        vf_coef=float(ppo_config["vf_coef"]),
        max_grad_norm=float(ppo_config["max_grad_norm"]),
        policy_kwargs={
            "activation_fn": torch.nn.Tanh,
            "net_arch": list(ppo_config["net_arch"]),
        },
        tensorboard_log=str(run_directory / "tensorboard"),
        seed=seed,
        verbose=1,
        device=args.device,
    )

    print(
        f"开始训练: envs={num_envs}, timesteps={total_timesteps}, "
        f"device={model.device}, output={run_directory}"
    )
    try:
        if args.mpc_dataset is not None:
            observations, actions, episode_ids = load_demonstrations(
                args.mpc_dataset, environment_config
            )
            bc_config = training_config["behavior_cloning"]
            epochs = args.bc_epochs if args.bc_epochs is not None else int(bc_config["epochs"])
            print(f"行为克隆初始化: samples={len(actions)}, epochs={epochs}")
            pretrain_policy(
                model,
                observations,
                actions,
                episode_ids,
                epochs=epochs,
                batch_size=int(bc_config["batch_size"]),
                learning_rate=float(bc_config["learning_rate"]),
                seed=seed,
            )
            model.save(run_directory / "bc_initialized_model")
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
        model.save(run_directory / "final_model")
    except KeyboardInterrupt:
        model.save(run_directory / "interrupted_model")
        print(f"训练已中断，模型已保存：{run_directory / 'interrupted_model.zip'}")
        return
    finally:
        if args.rviz:
            # SubprocVecEnv.close会等待在途step；退出前解除可视化worker的暂停。
            try:
                with VisualizationChannel(args.rviz_session) as channel:
                    channel.send("pause", False)
            except (OSError, sqlite3.Error, RuntimeError):
                pass
        train_env.close()
        eval_env.close()

    print(f"训练完成: {run_directory / 'final_model.zip'}")


if __name__ == "__main__":
    main()
