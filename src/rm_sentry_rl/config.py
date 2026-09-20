from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """加载训练配置，并将地图路径解析成绝对路径。"""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError(f"配置文件内容无效: {config_path}")

    result = deepcopy(config)
    if "map" in result and "yaml_path" in result["map"]:
        project_root = config_path.parent.parent
        map_path = Path(result["map"]["yaml_path"]).expanduser()
        if not map_path.is_absolute():
            map_path = (project_root / map_path).resolve()
        result["map"]["yaml_path"] = str(map_path)
    return result


def environment_fingerprint(config: dict[str, Any]) -> str:
    """绑定示范数据与地图文件、机器人动力学和观测配置。"""
    relevant = {
        key: config[key]
        for key in ("environment", "map", "robot", "observation", "path", "mpc")
    }
    relevant["environment_version"] = 2  # 保守净空、沿途碰撞检测和全分辨率规划。
    digest = hashlib.sha256(json.dumps(relevant, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    map_yaml = Path(config["map"]["yaml_path"])
    with map_yaml.open("rb") as stream:
        digest.update(stream.read())
    with map_yaml.open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)
    image_path = Path(metadata["image"]).expanduser()
    if not image_path.is_absolute():
        image_path = map_yaml.parent / image_path
    with image_path.open("rb") as stream:
        digest.update(stream.read())
    return digest.hexdigest()
