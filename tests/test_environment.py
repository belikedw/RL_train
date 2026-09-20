from __future__ import annotations

from pathlib import Path

import numpy as np

from rm_sentry_rl.config import load_config
from rm_sentry_rl.envs.omni_navigation_env import OmniNavigationEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_environment() -> OmniNavigationEnv:
    return OmniNavigationEnv(load_config(PROJECT_ROOT / "configs/env.yaml"))


def test_reset_and_step_contract() -> None:
    env = make_environment()
    observation, info = env.reset(seed=7)

    assert observation.shape == (97,)
    assert observation.dtype == np.float32
    assert np.all(np.isfinite(observation))
    assert np.all(observation >= -1.0)
    assert np.all(observation <= 1.0)
    assert info["map_resolution"] == 0.05

    action = np.zeros(2, dtype=np.float32)
    observation, reward, terminated, truncated, info = env.step(action)
    assert observation.shape == (97,)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["clearance"] >= 0.0
    env.close()


def test_reset_is_deterministic_for_same_seed() -> None:
    first = make_environment()
    second = make_environment()
    first_observation, _ = first.reset(seed=123)
    second_observation, _ = second.reset(seed=123)

    np.testing.assert_allclose(first.path, second.path)
    np.testing.assert_allclose(first.pose, second.pose)
    np.testing.assert_allclose(first_observation, second_observation)
    first.close()
    second.close()


def test_velocity_is_acceleration_limited() -> None:
    env = make_environment()
    env.reset(seed=20)
    before = env.velocity.copy()
    env.step(np.ones(2, dtype=np.float32))
    maximum_delta = env.max_acceleration * env.dt + 1e-6
    assert np.all(np.abs(env.velocity[:2] - before[:2]) <= maximum_delta)
    assert np.linalg.norm(env.velocity[:2]) <= env.max_speed + 1e-6
    env.close()
