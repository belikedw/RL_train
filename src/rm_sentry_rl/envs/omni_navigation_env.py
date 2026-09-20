from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from rm_sentry_rl.map_loader import GridMap, load_ros_map
from rm_sentry_rl.observation import ObservationBuilder
from rm_sentry_rl.path_planner import TrainingPathPlanner


class OmniNavigationEnv(gym.Env[np.ndarray, np.ndarray]):
    """基于固定ROS地图的轻量二维全向底盘导航环境。"""

    metadata = {"render_modes": []}

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        self.environment_config = config["environment"]
        self.robot_config = config["robot"]
        self.path_config = config["path"]
        self.reward_config = config["reward"]
        self.termination_config = config["termination"]

        self.dt = float(self.environment_config["dt"])
        self.max_steps = int(round(float(self.environment_config["max_episode_seconds"]) / self.dt))
        self.robot_radius = float(self.robot_config["radius"])
        self.max_speed = float(self.robot_config["max_speed"])
        self.max_acceleration = float(self.robot_config["max_acceleration"])
        self.max_angular_speed = float(self.robot_config["max_angular_speed"])

        self.grid_map: GridMap = load_ros_map(config["map"])
        self.path_planner = TrainingPathPlanner(self.grid_map, self.robot_radius, self.path_config)
        self.observation_builder = ObservationBuilder(
            self.grid_map,
            config["observation"],
            self.robot_config,
        )

        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.observation_builder.observation_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        self.pose = np.zeros(3, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.previous_command = np.zeros(3, dtype=np.float32)
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.path = np.zeros((2, 2), dtype=np.float32)
        self.path_arc_length = np.zeros(2, dtype=np.float32)
        self.spin_velocity = 0.0
        self.actuator_tau = 0.1
        self.step_count = 0
        self.previous_progress = 0.0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        initial_pose = None
        if "start" in options or "goal" in options:
            initial_pose = np.asarray(options.get("start"), dtype=np.float32)
            if initial_pose.shape != (3,) or not np.all(np.isfinite(initial_pose)):
                raise ValueError("start必须为有限的[x, y, yaw]")
            new_path = self.path_planner.plan(initial_pose[:2], options.get("goal"))
        else:
            new_path = self.path_planner.sample(self.np_random)
        requested_spin = options.get("spin_velocity")
        if requested_spin is not None and (not np.isfinite(requested_spin)
                                          or abs(requested_spin) > self.max_angular_speed):
            raise ValueError("spin_velocity必须在角速度范围内")
        self.path = new_path
        segment_lengths = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self.path_arc_length = np.concatenate(
            (np.zeros(1, dtype=np.float32), np.cumsum(segment_lengths, dtype=np.float32))
        )

        start_direction = self.path[1] - self.path[0]
        path_yaw = float(np.arctan2(start_direction[1], start_direction[0]))
        initial_yaw_offset = float(self.np_random.uniform(-np.pi, np.pi))
        self.pose[:] = (self.path[0, 0], self.path[0, 1], path_yaw + initial_yaw_offset)
        if initial_pose is not None:
            self.pose[:] = initial_pose
        self.velocity.fill(0.0)
        self.previous_command.fill(0.0)
        self.previous_action.fill(0.0)

        spins = np.asarray(self.robot_config.get("spin_velocities", [0.0]), dtype=np.float32)
        self.spin_velocity = float(self.np_random.choice(spins))
        if requested_spin is not None:
            self.spin_velocity = float(requested_spin)
        self.actuator_tau = float(
            self.np_random.uniform(
                float(self.robot_config["actuator_tau_min"]),
                float(self.robot_config["actuator_tau_max"]),
            )
        )
        self.step_count = 0
        self.previous_progress = self._path_progress()

        observation = self._observation()
        info = self._info(collision=False, goal_reached=False)
        return observation, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError("动作必须为有限的二维向量")
        clipped_action = np.clip(action, -1.0, 1.0)
        desired_velocity = clipped_action * self.max_speed

        response = np.clip(self.dt / max(self.actuator_tau, self.dt), 0.0, 1.0)
        lagged_delta = response * (desired_velocity - self.velocity[:2])
        maximum_delta = self.max_acceleration * self.dt
        velocity_delta = np.clip(lagged_delta, -maximum_delta, maximum_delta)
        self.velocity[:2] += velocity_delta

        speed = float(np.linalg.norm(self.velocity[:2]))
        if speed > self.max_speed:
            self.velocity[:2] *= self.max_speed / speed

        self.velocity[2] = np.clip(
            self.spin_velocity,
            -self.max_angular_speed,
            self.max_angular_speed,
        )

        cosine = float(np.cos(self.pose[2]))
        sine = float(np.sin(self.pose[2]))
        world_vx = cosine * self.velocity[0] - sine * self.velocity[1]
        world_vy = sine * self.velocity[0] + cosine * self.velocity[1]
        old_position = self.pose[:2].copy()
        self.pose[0] += world_vx * self.dt
        self.pose[1] += world_vy * self.dt
        self.pose[2] = np.float32((self.pose[2] + self.velocity[2] * self.dt + np.pi) % (2.0 * np.pi) - np.pi)

        self.previous_command[:] = (
            desired_velocity[0],
            desired_velocity[1],
            self.spin_velocity,
        )

        collision = not self.grid_map.segment_is_free(old_position, self.pose[:2], self.robot_radius)
        if collision:
            self.pose[:2] = old_position
            self.velocity.fill(0.0)
        progress = self._path_progress()
        path_error = self._path_error()
        clearance = self.grid_map.clearance_at(float(self.pose[0]), float(self.pose[1]))
        goal_reached = not collision and self._goal_reached()

        reward = self._reward(
            progress=progress,
            path_error=path_error,
            clearance=clearance,
            action=clipped_action,
            collision=collision,
            goal_reached=goal_reached,
        )

        self.previous_progress = progress
        self.previous_action = clipped_action.copy()
        self.step_count += 1

        terminated = bool(collision or goal_reached)
        truncated = bool(self.step_count >= self.max_steps)
        observation = self._observation()
        info = self._info(collision=collision, goal_reached=goal_reached)
        return observation, float(reward), terminated, truncated, info

    def _nearest_path_index(self) -> int:
        return self.observation_builder.nearest_path_index(self.pose[:2], self.path)

    def _path_progress(self) -> float:
        return float(self.path_arc_length[self._nearest_path_index()])

    def _path_error(self) -> float:
        index = self._nearest_path_index()
        return float(np.linalg.norm(self.path[index] - self.pose[:2]))

    def _goal_reached(self) -> bool:
        tolerance = float(self.termination_config["goal_tolerance"])
        return bool(np.linalg.norm(self.path[-1] - self.pose[:2]) <= tolerance)

    def _reward(
        self,
        *,
        progress: float,
        path_error: float,
        clearance: float,
        action: np.ndarray,
        collision: bool,
        goal_reached: bool,
    ) -> float:
        config = self.reward_config
        clearance_threshold = float(config["clearance_threshold"])
        clearance_violation = max(0.0, clearance_threshold - clearance)

        reward = float(config["progress"]) * (progress - self.previous_progress)
        reward -= float(config["path_error"]) * path_error * path_error * self.dt
        reward -= float(config["action_change"]) * float(np.sum((action - self.previous_action) ** 2))
        reward -= float(config["control_effort"]) * float(np.sum(action**2)) * self.dt
        reward -= float(config["clearance"]) * clearance_violation * clearance_violation * self.dt
        reward -= float(config["time"]) * self.dt

        if collision:
            reward -= float(config["collision"])
        if goal_reached:
            reward += float(config["goal"])
        return reward

    def _observation(self) -> np.ndarray:
        return self.observation_builder.build(
            pose=self.pose,
            velocity=self.velocity,
            previous_command=self.previous_command,
            path=self.path,
            spin_velocity=self.spin_velocity,
        )

    def _info(self, *, collision: bool, goal_reached: bool) -> dict[str, Any]:
        return {
            "collision": collision,
            "goal_reached": goal_reached,
            "path_error": self._path_error(),
            "clearance": self.grid_map.clearance_at(float(self.pose[0]), float(self.pose[1])),
            "progress": self._path_progress(),
            "map_width": self.grid_map.width,
            "map_height": self.grid_map.height,
            "map_resolution": self.grid_map.resolution,
        }
