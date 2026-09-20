from __future__ import annotations

import heapq
import math

import numpy as np

from rm_sentry_rl.map_loader import GridMap


_NEIGHBORS = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (1, 1, math.sqrt(2.0)),
)


def _astar(
    traversable: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]] | None:
    """训练端使用的简化八邻域A*，不替代线上全局规划器。"""
    height, width = traversable.shape
    queue: list[tuple[float, float, int, int]] = []
    heapq.heappush(queue, (0.0, 0.0, start[0], start[1]))

    parents: dict[tuple[int, int], tuple[int, int]] = {}
    best_cost = {start: 0.0}

    while queue:
        _, cost, x, y = heapq.heappop(queue)
        current = (x, y)
        if cost > best_cost.get(current, math.inf):
            continue
        if current == goal:
            path = [current]
            while current != start:
                current = parents[current]
                path.append(current)
            path.reverse()
            return path

        for dx, dy, move_cost in _NEIGHBORS:
            nx = x + dx
            ny = y + dy
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            if not traversable[ny, nx]:
                continue
            if dx != 0 and dy != 0:
                if not traversable[y, nx] or not traversable[ny, x]:
                    continue

            next_cost = cost + move_cost
            neighbor = (nx, ny)
            if next_cost >= best_cost.get(neighbor, math.inf):
                continue

            best_cost[neighbor] = next_cost
            parents[neighbor] = current
            heuristic = math.hypot(goal[0] - nx, goal[1] - ny)
            heapq.heappush(queue, (next_cost + heuristic, next_cost, nx, ny))

    return None


def _resample_polyline(path: np.ndarray, spacing: float) -> np.ndarray:
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("路径采样间距必须为有限正数")
    if path.shape[0] < 2:
        return path.astype(np.float32, copy=True)

    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(cumulative[-1])
    if total_length <= 1e-6:
        return path[:1].astype(np.float32, copy=True)

    samples = np.arange(0.0, total_length, spacing, dtype=np.float64)
    if samples.size == 0 or samples[-1] < total_length:
        samples = np.append(samples, total_length)
    # 保留拐点，防止重采样后的连线切入拐角障碍物。
    directions = np.diff(path, axis=0)
    if len(directions) > 1:
        turns = np.flatnonzero(np.any(np.abs(np.diff(directions, axis=0)) > 1e-6, axis=1)) + 1
        samples = np.unique(np.concatenate((samples, cumulative[turns])))

    x = np.interp(samples, cumulative, path[:, 0])
    y = np.interp(samples, cumulative, path[:, 1])
    return np.column_stack((x, y)).astype(np.float32)


class TrainingPathPlanner:
    def __init__(self, grid_map: GridMap, robot_radius: float, config: dict):
        self.grid_map = grid_map
        self.stride = max(1, int(config.get("astar_stride", 2)))
        self.resample_spacing = float(config.get("resample_spacing", 0.20))
        self.min_distance = float(config.get("min_start_goal_distance", 4.0))
        self.max_distance = float(config.get("max_start_goal_distance", 20.0))
        self.max_attempts = int(config.get("max_sampling_attempts", 100))

        required_clearance = robot_radius + float(config.get("safety_margin", 0.10))
        # 搜索使用完整栅格，stride只用于起终点采样，避免跨过细墙。
        self.required_clearance = required_clearance
        self.traversable = grid_map.clearance > required_clearance + grid_map.resolution / np.sqrt(2.0)
        self.free_cells = np.argwhere(self.traversable[:: self.stride, :: self.stride]) * self.stride
        if self.free_cells.shape[0] < 2:
            raise RuntimeError("地图中没有足够的可通行区域")

    def plan(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """在map坐标系中规划，显式拒绝无效、碰撞和不可达的起终点。"""
        start, goal = np.asarray(start, dtype=np.float64), np.asarray(goal, dtype=np.float64)
        cells = []
        for name, point in (("起点", start), ("终点", goal)):
            if point.shape != (2,) or not np.all(np.isfinite(point)):
                raise ValueError(f"{name}必须为有限的[x, y]")
            cell = self.grid_map.world_to_map(*point)
            if cell is None or not self.traversable[cell[1], cell[0]]:
                raise ValueError(f"{name}越界或安全距离不足")
            if self.grid_map.clearance_at(*point) <= self.required_clearance:
                raise ValueError(f"{name}安全距离不足")
            cells.append(cell)
        if np.linalg.norm(goal - start) < 1e-4:
            raise ValueError("起终点不能重合")
        cells_path = _astar(self.traversable, cells[0], cells[1])
        if cells_path is None:
            raise ValueError("起点与终点之间没有可通行路径")
        centers = np.asarray([self.grid_map.map_to_world(*cell) for cell in cells_path])
        path = np.vstack((start, centers, goal))
        path = path[np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-6]]
        if not all(self.grid_map.segment_is_free(a, b, self.required_clearance)
                   for a, b in zip(path[:-1], path[1:])):
            raise ValueError("路径沿途安全距离不足，请调整起终点")
        return _resample_polyline(path, self.resample_spacing)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        for _ in range(self.max_attempts):
            indices = rng.integers(0, self.free_cells.shape[0], size=2)
            start_y, start_x = self.free_cells[indices[0]]
            goal_y, goal_x = self.free_cells[indices[1]]

            distance_cells = math.hypot(goal_x - start_x, goal_y - start_y)
            distance_m = distance_cells * self.grid_map.resolution
            if distance_m < self.min_distance or distance_m > self.max_distance:
                continue

            start = np.asarray(self.grid_map.map_to_world(int(start_x), int(start_y)))
            goal = np.asarray(self.grid_map.map_to_world(int(goal_x), int(goal_y)))
            try:
                return self.plan(start, goal)
            except ValueError:
                continue

        raise RuntimeError("无法在指定距离约束内采样出可达路径")
