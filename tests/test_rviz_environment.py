from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rm_sentry_rl.config import load_config
from rm_sentry_rl.envs.omni_navigation_env import OmniNavigationEnv
from rm_sentry_rl.visualization import RvizEnvironment, load_scenario
from rm_sentry_rl.visualization_transport import VisualizationChannel

ROOT = Path(__file__).resolve().parents[1]


def environment():
    return OmniNavigationEnv(load_config(ROOT / "configs/env.yaml"))


def test_visible_and_headless_trajectories_match(tmp_path):
    plain = environment()
    visible = RvizEnvironment(environment(), tmp_path, mode="train")
    try:
        obs, _ = plain.reset(seed=123)
        shown, _ = visible.reset(seed=123)
        np.testing.assert_array_equal(obs, shown)
        rng = np.random.default_rng(5)
        for _ in range(40):
            action = rng.uniform(-1, 1, 2).astype(np.float32)
            result, shown = plain.step(action), visible.step(action)
            np.testing.assert_array_equal(result[0], shown[0])
            assert result[1:] == shown[1:]
    finally:
        plain.close()
        visible.close()


def test_goal_switch_truncates_old_episode_before_reset(tmp_path):
    env = RvizEnvironment(environment(), tmp_path, mode="train")
    try:
        env.reset(seed=7)
        original = env.base.path.copy()
        with VisualizationChannel(tmp_path) as ui:
            ui.send("start", [2.0, 2.0, 0.0])
            ui.send("goal", [3.5, 2.0])
        env.poll(force=True)
        np.testing.assert_array_equal(env.base.path, original)
        _, _, _, truncated, info = env.step(np.zeros(2, dtype=np.float32))
        assert truncated and info["interactive_reset"]
        env.reset()
        np.testing.assert_allclose(env.base.pose, [2, 2, 0])
        np.testing.assert_allclose(env.base.path[-1], [3.5, 2])
        assert not env.reset_requested
    finally:
        env.close()


def test_invalid_goal_keeps_current_episode(tmp_path):
    env = RvizEnvironment(environment(), tmp_path, mode="plan")
    try:
        env.reset(seed=5)
        before = env.base.path.copy()
        with VisualizationChannel(tmp_path) as ui:
            ui.send("goal", [-1.0, -1.0])
        env.poll(force=True)
        np.testing.assert_array_equal(env.base.path, before)
        assert not env.reset_requested
        assert "失败" in env.message
    finally:
        env.close()


def test_save_reload_pause_step_and_speed(tmp_path):
    env = RvizEnvironment(environment(), tmp_path, mode="follow")
    try:
        env.reset(seed=12)
        with VisualizationChannel(tmp_path) as ui:
            ui.send("pause", True)
            ui.send("step")
            ui.send("speed", 2.0)
            ui.send("save")
        env.poll(force=True)
        assert env.paused and env.single_steps == 1 and env.speed == 2
        options = load_scenario(tmp_path / "scenario.json", env.base.config)
        np.testing.assert_allclose(options["start"], env.base.pose)
        env.step(np.zeros(2, dtype=np.float32))
        assert env.base.step_count == 1 and env.single_steps == 0
        with VisualizationChannel(tmp_path) as ui:
            ui.send("reset")
        env.poll(force=True)
        assert env.reset_requested
        env.reset()
        np.testing.assert_allclose(env.base.pose, options["start"])
        plain = environment()
        try:
            plain.reset(options=options)
            np.testing.assert_allclose(plain.path, env.base.path, atol=2e-6)
        finally:
            plain.close()
    finally:
        env.close()


def test_channel_bounds_and_producer_ownership(tmp_path):
    with VisualizationChannel(tmp_path, producer=True) as producer:
        with pytest.raises(RuntimeError, match="占用"):
            VisualizationChannel(tmp_path, producer=True)
        with VisualizationChannel(tmp_path) as reader:
            for _ in range(64):
                reader.send("pause", True)
            with pytest.raises(RuntimeError, match="已满"):
                reader.send("reset")
            assert len(producer.drain()) == 64
            assert producer.drain() == []
    with VisualizationChannel(tmp_path, producer=True):
        pass


def test_sb3_auto_reset_retains_terminal_observation(tmp_path):
    from stable_baselines3.common.vec_env import DummyVecEnv

    vector = DummyVecEnv([lambda: RvizEnvironment(environment(), tmp_path, mode="train")])
    try:
        vector.seed(42)
        vector.reset()
        visible = vector.envs[0]
        with VisualizationChannel(tmp_path) as ui:
            ui.send("start", [2.0, 2.0, 0.0])
            ui.send("goal", [3.5, 2.0])
        visible.poll(force=True)
        observations, _, done, info = vector.step(np.zeros((1, 2), dtype=np.float32))
        assert done[0] and info[0]["TimeLimit.truncated"]
        assert "terminal_observation" in info[0]
        np.testing.assert_allclose(visible.base.pose, [2, 2, 0])
        np.testing.assert_array_equal(observations[0], visible.base._observation())
    finally:
        vector.close()
