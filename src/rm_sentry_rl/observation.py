from __future__ import annotations

import numpy as np

from rm_sentry_rl.map_loader import GridMap


class ObservationBuilder:
    """构造与部署端保持固定协议的97维观测。"""

    VERSION = 2

    def __init__(self, grid_map: GridMap, config: dict, robot_config: dict):
        self.grid_map = grid_map
        self.num_rays = int(config.get("num_rays", 64))
        self.ray_max_range = float(config.get("ray_max_range", 4.0))
        self.path_points = int(config.get("path_points", 12))
        self.path_point_interval = float(config.get("path_point_interval", 0.25))
        self.path_normalization_range = float(config.get("path_normalization_range", 4.0))
        self.max_speed = float(robot_config["max_speed"])
        self.max_angular_speed = float(robot_config["max_angular_speed"])

        self.ray_angles = np.linspace(
            -np.pi,
            np.pi,
            self.num_rays,
            endpoint=False,
            dtype=np.float32,
        )
        self.ray_samples = np.arange(
            self.grid_map.resolution,
            self.ray_max_range + 0.5 * self.grid_map.resolution,
            self.grid_map.resolution,
            dtype=np.float32,
        )
        self.observation_dim = self.num_rays + self.path_points * 2 + 3 + 3 + 2 + 1

    def ray_distances(self, pose: np.ndarray) -> np.ndarray:
        angles = self.ray_angles + np.float32(pose[2])
        cosines = np.cos(angles)[:, None]
        sines = np.sin(angles)[:, None]

        world_x = pose[0] + cosines * self.ray_samples[None, :]
        world_y = pose[1] + sines * self.ray_samples[None, :]
        map_x = np.floor((world_x - self.grid_map.origin_x) / self.grid_map.resolution).astype(np.int32)
        map_y = np.floor((world_y - self.grid_map.origin_y) / self.grid_map.resolution).astype(np.int32)

        outside = (
            (map_x < 0)
            | (map_y < 0)
            | (map_x >= self.grid_map.width)
            | (map_y >= self.grid_map.height)
        )
        clipped_x = np.clip(map_x, 0, self.grid_map.width - 1)
        clipped_y = np.clip(map_y, 0, self.grid_map.height - 1)
        hits = outside | self.grid_map.occupied[clipped_y, clipped_x]

        has_hit = np.any(hits, axis=1)
        first_hit = np.argmax(hits, axis=1)
        distances = np.full(self.num_rays, self.ray_max_range, dtype=np.float32)
        distances[has_hit] = self.ray_samples[first_hit[has_hit]]
        return distances

    @staticmethod
    def nearest_path_index(position: np.ndarray, path: np.ndarray) -> int:
        distances_sq = np.sum((path - position[None, :]) ** 2, axis=1)
        return int(np.argmin(distances_sq))

    def _local_path_points(self, pose: np.ndarray, path: np.ndarray) -> np.ndarray:
        nearest = self.nearest_path_index(pose[:2], path)
        path_spacing = np.linalg.norm(np.diff(path, axis=0), axis=1)
        arc = np.r_[0.0, np.cumsum(path_spacing)]
        samples = arc[nearest] + np.arange(self.path_points) * self.path_point_interval
        points = np.column_stack((np.interp(samples, arc, path[:, 0]),
                                  np.interp(samples, arc, path[:, 1])))
        relative = points - pose[None, :2]

        cosine = np.cos(pose[2])
        sine = np.sin(pose[2])
        local = np.empty_like(relative, dtype=np.float32)
        local[:, 0] = cosine * relative[:, 0] + sine * relative[:, 1]
        local[:, 1] = -sine * relative[:, 0] + cosine * relative[:, 1]
        return np.clip(
            local / self.path_normalization_range,
            -1.0,
            1.0,
        ).reshape(-1)

    def build(
        self,
        pose: np.ndarray,
        velocity: np.ndarray,
        previous_command: np.ndarray,
        path: np.ndarray,
        spin_velocity: float,
    ) -> np.ndarray:
        rays = 2.0 * self.ray_distances(pose) / self.ray_max_range - 1.0
        local_path = self._local_path_points(pose, path)

        velocity_normalized = np.array(
            [
                velocity[0] / self.max_speed,
                velocity[1] / self.max_speed,
                velocity[2] / self.max_angular_speed,
            ],
            dtype=np.float32,
        )
        command_normalized = np.array(
            [
                previous_command[0] / self.max_speed,
                previous_command[1] / self.max_speed,
                previous_command[2] / self.max_angular_speed,
            ],
            dtype=np.float32,
        )

        goal_relative = path[-1] - pose[:2]
        cosine = np.cos(pose[2])
        sine = np.sin(pose[2])
        local_goal = np.array(
            [
                cosine * goal_relative[0] + sine * goal_relative[1],
                -sine * goal_relative[0] + cosine * goal_relative[1],
            ],
            dtype=np.float32,
        )
        local_goal = np.clip(local_goal / self.path_normalization_range, -1.0, 1.0)
        normalized_spin = np.array(
            [spin_velocity / self.max_angular_speed],
            dtype=np.float32,
        )

        observation = np.concatenate(
            (
                rays,
                local_path,
                velocity_normalized,
                command_normalized,
                local_goal,
                normalized_spin,
            )
        ).astype(np.float32, copy=False)
        observation = np.clip(observation, -1.0, 1.0)
        if observation.shape != (self.observation_dim,):
            raise RuntimeError(f"观测维度错误: {observation.shape}, 期望 {(self.observation_dim,)}")
        return observation
