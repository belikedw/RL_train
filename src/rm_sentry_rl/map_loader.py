from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class GridMap:
    """采用ROS地图坐标约定的二维栅格地图。"""

    occupied: np.ndarray
    unknown: np.ndarray
    clearance: np.ndarray
    resolution: float
    origin_x: float
    origin_y: float

    @property
    def width(self) -> int:
        return int(self.occupied.shape[1])

    @property
    def height(self) -> int:
        return int(self.occupied.shape[0])

    @property
    def width_m(self) -> float:
        return self.width * self.resolution

    @property
    def height_m(self) -> float:
        return self.height * self.resolution

    def world_to_map(self, x: float, y: float) -> tuple[int, int] | None:
        mx = int(np.floor((x - self.origin_x) / self.resolution))
        my = int(np.floor((y - self.origin_y) / self.resolution))
        if mx < 0 or my < 0 or mx >= self.width or my >= self.height:
            return None
        return mx, my

    def map_to_world(self, mx: int, my: int) -> tuple[float, float]:
        x = self.origin_x + (mx + 0.5) * self.resolution
        y = self.origin_y + (my + 0.5) * self.resolution
        return x, y

    def clearance_at(self, x: float, y: float) -> float:
        cell = self.world_to_map(x, y)
        if cell is None:
            return 0.0
        mx, my = cell
        cx, cy = self.map_to_world(mx, my)
        # 距离场位于栅格中心；扣除查询点偏移，得到保守的机器人净空。
        return max(0.0, min(
            float(self.clearance[my, mx]) - float(np.hypot(x - cx, y - cy)),
            x - self.origin_x, y - self.origin_y,
            self.origin_x + self.width_m - x,
            self.origin_y + self.height_m - y,
        ))

    def segment_is_free(self, start: np.ndarray, end: np.ndarray, radius: float) -> bool:
        """沿运动线段检查圆形足迹，采样余量用于覆盖相邻采样点间隙。"""
        distance = float(np.linalg.norm(end - start))
        count = max(1, int(np.ceil(distance / (self.resolution * 0.5))))
        margin = distance / count * 0.5
        return all(
            self.clearance_at(float(p[0]), float(p[1])) > radius + margin
            for p in np.linspace(start, end, count + 1)
        )


def _load_metadata(yaml_path: Path) -> dict[str, Any]:
    with yaml_path.open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)
    if not isinstance(metadata, dict):
        raise ValueError(f"地图YAML内容无效: {yaml_path}")
    return metadata


def load_ros_map(map_config: dict[str, Any]) -> GridMap:
    """读取PGM/YAML并生成静态欧氏距离场，未知和越界默认视为障碍。"""
    yaml_path = Path(map_config["yaml_path"]).expanduser().resolve()
    metadata = _load_metadata(yaml_path)

    image_path = Path(str(metadata["image"])).expanduser()
    if not image_path.is_absolute():
        image_path = (yaml_path.parent / image_path).resolve()

    image = np.asarray(Image.open(image_path).convert("L"), dtype=np.uint8)
    image = np.flipud(image)

    negate = bool(metadata.get("negate", 0))
    if negate:
        image = np.uint8(255) - image

    if bool(map_config.get("use_project_pixel_thresholds", True)):
        occupied_threshold = int(map_config.get("occupied_pixel_threshold", 10))
        free_threshold = int(map_config.get("free_pixel_threshold", 245))
        occupied = image < occupied_threshold
        free = image > free_threshold
        unknown = ~(occupied | free)
    else:
        probability = (255.0 - image.astype(np.float32)) / 255.0
        occupied = probability > float(metadata.get("occupied_thresh", 0.65))
        free = probability < float(metadata.get("free_thresh", 0.25))
        unknown = ~(occupied | free)

    if bool(map_config.get("unknown_as_occupied", True)):
        occupied = occupied | unknown

    resolution = float(metadata["resolution"])
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("地图分辨率必须为有限正数")
    origin = metadata.get("origin", [0.0, 0.0, 0.0])
    if len(origin) < 2:
        raise ValueError(f"地图origin无效: {origin}")
    if len(origin) >= 3 and abs(float(origin[2])) > 1e-6:
        raise ValueError("首版训练环境暂不支持带旋转的地图原点")

    if not np.all(np.isfinite(origin)):
        raise ValueError("地图原点必须为有限数值")
    # 补障碍边界并扣除障碍格半对角线，避免把障碍物当成零面积点。
    padded = np.pad(~occupied, 1, constant_values=False)
    clearance = distance_transform_edt(padded)[1:-1, 1:-1] * resolution
    clearance = np.maximum(0.0, clearance - resolution / np.sqrt(2.0)).astype(np.float32)
    return GridMap(
        occupied=np.ascontiguousarray(occupied, dtype=np.bool_),
        unknown=np.ascontiguousarray(unknown, dtype=np.bool_),
        clearance=np.ascontiguousarray(clearance, dtype=np.float32),
        resolution=resolution,
        origin_x=float(origin[0]),
        origin_y=float(origin[1]),
    )
