from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rm_sentry_rl.config import environment_fingerprint
from rm_sentry_rl.observation import ObservationBuilder

if TYPE_CHECKING:
    from stable_baselines3 import PPO


def load_demonstrations(path: Path, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as dataset:
        observations = np.asarray(dataset["observations"], dtype=np.float32)
        actions = np.asarray(dataset["actions"], dtype=np.float32)
        episode_ids = np.asarray(dataset["episode_ids"], dtype=np.int32)
        version = int(dataset["observation_version"])
        fingerprint = str(dataset["environment_fingerprint"])
        teacher_kind = str(dataset["teacher_kind"])

    if version != ObservationBuilder.VERSION:
        raise ValueError(f"示范数据观测协议版本不匹配: {version}")
    if fingerprint != environment_fingerprint(config):
        raise ValueError("示范数据与当前地图、动力学或观测配置不匹配")
    if teacher_kind != "cpp_omni_mpc_v1":
        raise ValueError(f"示范数据不是当前C++ MPC教师生成的: {teacher_kind}")
    expected_dim = (
        int(config["observation"]["num_rays"])
        + 2 * int(config["observation"]["path_points"])
        + 9
    )
    if observations.ndim != 2 or observations.shape[1] != expected_dim:
        raise ValueError(f"示范观测维度错误: {observations.shape}")
    if actions.shape != (observations.shape[0], 2) or observations.shape[0] < 2:
        raise ValueError(f"示范动作维度或样本数错误: {actions.shape}")
    if episode_ids.shape != (observations.shape[0],):
        raise ValueError(f"示范episode_ids维度错误: {episode_ids.shape}")
    if not (np.all(np.isfinite(observations)) and np.all(np.isfinite(actions))):
        raise ValueError("示范数据包含NaN或Inf")
    if np.any(np.abs(observations) > 1.0001) or np.any(np.abs(actions) > 1.0001):
        raise ValueError("示范数据超出归一化观测或动作范围")
    return observations, actions, episode_ids


def pretrain_policy(
    model: PPO,
    observations: np.ndarray,
    actions: np.ndarray,
    episode_ids: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> float:
    """仅用MPC动作监督PPO策略均值；价值网络留给PPO在线学习。"""
    import torch

    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("行为克隆训练参数必须为正数")
    rng = np.random.default_rng(seed)
    unique_episodes = rng.permutation(np.unique(episode_ids))
    if unique_episodes.size > 1:
        validation_episodes = unique_episodes[: max(1, unique_episodes.size // 10)]
        validation_mask = np.isin(episode_ids, validation_episodes)
        validation_indices = np.flatnonzero(validation_mask)
        training_indices = np.flatnonzero(~validation_mask)
    else:
        validation_count = max(1, observations.shape[0] // 10)
        validation_indices = np.arange(observations.shape[0] - validation_count, observations.shape[0])
        training_indices = np.arange(observations.shape[0] - validation_count)
    validation_count = validation_indices.size
    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    policy.set_training_mode(True)

    for epoch in range(epochs):
        for batch in np.array_split(
            rng.permutation(training_indices),
            max(1, int(np.ceil(training_indices.size / batch_size))),
        ):
            x = torch.as_tensor(observations[batch], device=model.device)
            y = torch.as_tensor(actions[batch], device=model.device)
            prediction = policy.get_distribution(x).distribution.mean
            loss = torch.nn.functional.mse_loss(prediction, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            validation_errors = []
            for batch in np.array_split(
                validation_indices,
                max(1, int(np.ceil(validation_count / batch_size))),
            ):
                x = torch.as_tensor(observations[batch], device=model.device)
                y = torch.as_tensor(actions[batch], device=model.device)
                prediction = policy.get_distribution(x).distribution.mean
                validation_errors.append(float(torch.sum((prediction - y) ** 2).item()))
            validation_mse = sum(validation_errors) / (validation_count * actions.shape[1])
        print(f"行为克隆 epoch={epoch + 1}/{epochs}, validation_mse={validation_mse:.6f}")

    policy.set_training_mode(False)
    return validation_mse
