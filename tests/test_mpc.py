from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rm_sentry_rl.config import environment_fingerprint, load_config
from rm_sentry_rl.envs.omni_navigation_env import OmniNavigationEnv
from rm_sentry_rl.imitation import load_demonstrations
from rm_sentry_rl.mpc import OmniMpcTeacher, find_bridge_binary
from rm_sentry_rl.observation import ObservationBuilder


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_environment() -> OmniNavigationEnv:
    return OmniNavigationEnv(load_config(PROJECT_ROOT / "configs/env.yaml"))


def require_bridge() -> None:
    try:
        find_bridge_binary()
    except FileNotFoundError:
        pytest.skip("尚未构建C++ MPC桥接程序")


def test_cpp_teacher_action_contract_and_progress() -> None:
    require_bridge()
    first = make_environment()
    second = make_environment()
    try:
        first.reset(seed=20)
        second.reset(seed=20)
        with OmniMpcTeacher(first) as first_teacher, OmniMpcTeacher(second) as second_teacher:
            first_teacher.reset()
            second_teacher.reset()
            first_action = first_teacher.act()
            np.testing.assert_allclose(first_action, second_teacher.act(), atol=1e-5)
            assert first_action.shape == (2,)
            assert first_action.dtype == np.float32
            assert np.all(np.isfinite(first_action))
            assert np.all(np.abs(first_action) <= 1.0)

            start_progress = first._path_progress()
            for _ in range(150):
                _, _, terminated, _, info = first.step(first_teacher.act())
                assert not info["collision"]
                if terminated:
                    break
            assert info["progress"] > start_progress + 0.5
    finally:
        first.close()
        second.close()


def test_cpp_teacher_restarts_between_episodes() -> None:
    require_bridge()
    env = make_environment()
    try:
        with OmniMpcTeacher(env) as teacher:
            env.reset(seed=20)
            teacher.reset()
            first_pid = teacher._process.pid
            teacher.act()
            env.reset(seed=7)
            teacher.reset()
            assert teacher._process.pid != first_pid
            assert np.all(np.isfinite(teacher.act()))
    finally:
        env.close()


def test_demonstration_contract_rejects_other_teacher(tmp_path: Path) -> None:
    config = load_config(PROJECT_ROOT / "configs/env.yaml")
    dataset = tmp_path / "demonstrations.npz"
    values = dict(
        observations=np.zeros((2, 97), dtype=np.float32),
        actions=np.zeros((2, 2), dtype=np.float32),
        episode_ids=np.array([0, 1], dtype=np.int32),
        observation_version=np.int32(ObservationBuilder.VERSION),
        environment_fingerprint=environment_fingerprint(config),
    )
    np.savez(dataset, **values, teacher_kind="cpp_omni_mpc_v1")
    observations, actions, episode_ids = load_demonstrations(dataset, config)
    assert observations.shape == (2, 97)
    assert actions.shape == (2, 2)
    assert episode_ids.shape == (2,)

    np.savez(dataset, **values, teacher_kind="python_sampling_mpc")
    with pytest.raises(ValueError, match=r"C\+\+ MPC"):
        load_demonstrations(dataset, config)

    config["robot"]["max_speed"] = 2.0
    with pytest.raises(ValueError, match="不匹配"):
        load_demonstrations(dataset, config)
