from __future__ import annotations

from pathlib import Path

from rm_sentry_rl.config import load_config
from rm_sentry_rl.map_loader import load_ros_map


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_project_map() -> None:
    config = load_config(PROJECT_ROOT / "configs/env.yaml")
    grid_map = load_ros_map(config["map"])

    assert grid_map.width > 0
    assert grid_map.height > 0
    assert grid_map.resolution == 0.05
    assert grid_map.occupied.shape == grid_map.clearance.shape
    assert grid_map.clearance_at(grid_map.origin_x - 1.0, grid_map.origin_y) == 0.0
