#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rm_sentry_rl.config import load_config
from rm_sentry_rl.map_loader import load_ros_map


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="检查训练地图尺寸和占用统计")
    parser.add_argument("--env-config", type=Path, default=PROJECT_ROOT / "configs/env.yaml")
    args = parser.parse_args()

    config = load_config(args.env_config)
    grid_map = load_ros_map(config["map"])
    print(f"map:        {config['map']['yaml_path']}")
    print(f"cells:      {grid_map.width} x {grid_map.height}")
    print(f"resolution: {grid_map.resolution:.3f} m/cell")
    print(f"size:       {grid_map.width_m:.3f} x {grid_map.height_m:.3f} m")
    print(f"origin:     ({grid_map.origin_x:.3f}, {grid_map.origin_y:.3f})")
    print(f"occupied:   {np.mean(grid_map.occupied):.3%}")
    print(f"unknown:    {np.mean(grid_map.unknown):.3%}")


if __name__ == "__main__":
    main()
