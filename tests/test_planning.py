from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rm_sentry_rl.config import load_config
from rm_sentry_rl.envs.omni_navigation_env import OmniNavigationEnv
from rm_sentry_rl.map_loader import load_ros_map
from rm_sentry_rl.path_planner import TrainingPathPlanner

ROOT = Path(__file__).resolve().parents[1]


def thin_wall_map(tmp_path):
    pixels = np.full((80, 80), 254, dtype=np.uint8)
    pixels[:, 39] = 0
    Image.fromarray(pixels).save(tmp_path / "wall.pgm")
    (tmp_path / "wall.yaml").write_text(
        "image: wall.pgm\nresolution: 0.1\norigin: [0.0, 0.0, 0.0]\n", encoding="utf-8"
    )
    return load_ros_map({"yaml_path": tmp_path / "wall.yaml"})


def test_full_resolution_planning_cannot_cross_thin_wall(tmp_path):
    grid = thin_wall_map(tmp_path)
    planner = TrainingPathPlanner(grid, 0.15, {"astar_stride": 4, "safety_margin": 0.0})
    with pytest.raises(ValueError, match="没有可通行路径"):
        planner.plan(np.array([2, 4]), np.array([6, 4]))


def test_swept_collision_and_map_boundary(tmp_path):
    grid = thin_wall_map(tmp_path)
    assert grid.clearance_at(0.01, 2) <= 0.01
    assert grid.clearance_at(2, 4) > 0.15
    assert grid.clearance_at(6, 4) > 0.15
    assert not grid.segment_is_free(np.array([2, 4]), np.array([6, 4]), 0.15)


def test_env_detects_collision_between_free_endpoints(tmp_path):
    grid = thin_wall_map(tmp_path)
    config = load_config(ROOT / "configs/env.yaml")
    config["map"]["yaml_path"] = str(tmp_path / "wall.yaml")
    config["environment"]["dt"] = 1.0
    config["robot"].update(radius=0.15, max_speed=4.0, max_acceleration=10.0)
    env = OmniNavigationEnv(config)
    try:
        env.reset(options={"start": [2, 4, 0], "goal": [2, 6], "spin_velocity": 0.0})
        _, _, terminated, _, info = env.step(np.array([1, 0], dtype=np.float32))
        assert terminated and info["collision"] and not info["goal_reached"]
        assert env.pose[0] < 3.9
    finally:
        env.close()


def test_manual_plan_and_invalid_inputs():
    env = OmniNavigationEnv(load_config(ROOT / "configs/env.yaml"))
    try:
        env.reset(options={"start": [2, 2, 1.2], "goal": [11, 8.5], "spin_velocity": 0})
        np.testing.assert_allclose(env.pose, [2, 2, 1.2])
        np.testing.assert_allclose(env.path[-1], [11, 8.5])
        assert all(env.grid_map.segment_is_free(a, b, env.robot_radius)
                   for a, b in zip(env.path[:-1], env.path[1:]))
        for goal in ([float("nan"), 2], [-1, 3], [4.95, 2], [2, 2]):
            with pytest.raises(ValueError):
                env.path_planner.plan(np.array([2, 2]), np.array(goal))
        with pytest.raises(ValueError):
            env.step(np.array([float("nan"), 0]))
    finally:
        env.close()
