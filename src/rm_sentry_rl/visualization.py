from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from rm_sentry_rl.config import environment_fingerprint
from rm_sentry_rl.envs.omni_navigation_env import OmniNavigationEnv
from rm_sentry_rl.visualization_transport import VisualizationChannel


def load_scenario(path: str | Path, config: dict) -> dict:
    """场景文件只保存起终点，不导入外部全局路径。"""
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if value.get("environment_fingerprint") != environment_fingerprint(config):
        raise ValueError("场景与当前地图/环境配置不匹配")
    return {key: value[key] for key in ("start", "goal", "spin_velocity")}


class RvizEnvironment(gym.Wrapper):
    """在环境主线程处理控制请求，ROS回调不直接修改仿真状态。"""

    def __init__(self, env: OmniNavigationEnv, session: str | Path, *, mode: str,
                 speed: float = 0.0, paused: bool = False, scenario: dict | None = None):
        super().__init__(env)
        if not np.isfinite(speed) or speed < 0:
            raise ValueError("speed必须为非负有限数，0表示不限速")
        self.base = env
        self.channel = VisualizationChannel(session, producer=True)
        self.mode, self.speed, self.paused = mode, speed, paused
        self.options = scenario
        self.selected_start: list[float] | None = None
        self.reset_requested = False
        self.single_steps = 0
        self.message = "就绪；可设置起点和终点"
        self.display_message = "Set a start pose and a goal with the toolbar."
        self.episode = 0
        self.run_id = uuid.uuid4().hex
        self.reward_sum = 0.0
        self.trajectory: deque = deque(maxlen=1000)
        self.last_info: dict[str, Any] = {}
        self.last_result: dict[str, Any] = {}
        self.published_episode_id = None
        self.last_poll = self.last_publish = 0.0
        self.last_step = time.monotonic()
        self.episode_data: dict = {}
        grid = env.grid_map
        occupancy = np.where(grid.occupied, 100, 0).astype(np.int8)
        occupancy[grid.unknown] = -1
        self.channel.put(map={
            "run_id": self.run_id, "width": grid.width, "height": grid.height,
            "resolution": grid.resolution, "origin": [grid.origin_x, grid.origin_y],
            "data": occupancy.ravel().tolist(),
        })

    def reset(self, *, seed=None, options=None):
        chosen = self.options if options is None else options
        if chosen is None and self.mode in ("plan", "follow"):
            chosen = {"spin_velocity": 0.0}
        observation, info = self.env.reset(seed=seed, options=chosen)
        self.episode += 1
        self.reward_sum = 0.0
        self.reset_requested = False
        self.trajectory.clear()
        self.trajectory.append(self.base.pose[:2].tolist())
        self.last_info = info
        self.episode_data = {
            "run_id": self.run_id,
            "id": f"{self.run_id}:{self.episode}", "number": self.episode,
            "path": self.base.path.tolist(), "start": self.base.pose.tolist(),
            "goal": self.base.path[-1].tolist(), "spin_velocity": self.base.spin_velocity,
        }
        self.last_step = time.monotonic()
        self.publish(force=True)
        return observation, info

    def _handle(self, command: dict) -> None:
        if time.time() - command["created"] > 30:
            raise ValueError("控制命令已过期，请重新发送")
        kind, value = command["kind"], command["value"]
        if kind == "start":
            pose = np.asarray(value, dtype=np.float64)
            if pose.shape != (3,) or not np.all(np.isfinite(pose)):
                raise ValueError("起点必须是有限的[x, y, yaw]")
            if self.base.grid_map.clearance_at(*pose[:2]) <= self.base.path_planner.required_clearance:
                raise ValueError("起点越界或安全距离不足")
            self.selected_start = pose.tolist()
            self.message = "起点已设置，请选择终点"
            self.display_message = "Start selected. Choose a goal."
        elif kind == "goal":
            start = self.selected_start or self.episode_data["start"]
            self.base.path_planner.plan(np.asarray(start[:2]), np.asarray(value))
            self.options = {"start": list(start), "goal": value, "spin_velocity": 0.0}
            self.reset_requested = True
            self.message = "新路径已验证，将在回合边界切换"
            self.display_message = "Goal accepted. Switching at episode boundary."
        elif kind == "pause":
            self.paused = bool(value)
            self.single_steps = 0
            self.message = "已暂停" if self.paused else "继续运行"
            self.display_message = "Paused." if self.paused else "Running."
        elif kind == "step":
            self.paused = True
            self.single_steps = min(self.single_steps + 1, 10)
            self.message = "单步运行"
            self.display_message = "Single step; stays paused afterwards."
        elif kind in ("reset", "random"):
            if kind == "random":
                self.options = None
                self.selected_start = None
            else:
                self.options = {key: self.episode_data[key]
                                for key in ("start", "goal", "spin_velocity")}
            self.reset_requested = True
            self.message = "等待回合重置"
            self.display_message = "Episode reset requested."
        elif kind == "speed":
            speed = float(value)
            if not np.isfinite(speed) or speed < 0:
                raise ValueError("speed必须为非负有限数")
            self.speed = speed
            self.message = f"运行倍率：{speed:g}（0为不限速）"
            self.display_message = f"Speed factor: {speed:g} (0 = unlimited)."
        elif kind == "save":
            target = self.channel.directory / "scenario.json"
            values = {key: self.episode_data[key] for key in ("start", "goal", "spin_velocity")}
            values["environment_fingerprint"] = environment_fingerprint(self.base.config)
            # 保存的是当前生效场景；使用临时文件替换，避免读取到半份JSON。
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(values, indent=2), encoding="utf-8")
            temporary.replace(target)
            self.message = f"已保存当前场景：{target}"
            self.display_message = "Current scene saved to scenario.json."
        else:
            raise ValueError(f"未知命令：{kind}")

    def poll(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_poll < 0.05:
            return
        self.last_poll = now
        try:
            commands = self.channel.drain()
        except sqlite3.OperationalError:
            return  # 可视化通道短暂繁忙，不影响仿真推进。
        for command in commands:
            try:
                self._handle(command)
            except (ValueError, TypeError, KeyError, OSError) as error:
                self.message = f"请求失败：{error}"
                self.display_message = "Request rejected. See bridge terminal for details."
        self.publish(force=bool(commands))

    def publish(self, *, force: bool = False, finished: bool = False) -> None:
        now = time.monotonic()
        if not self.episode_data or (not force and now - self.last_publish < 0.1):
            return
        self.last_publish = now
        position = self.base.pose[:2].tolist()
        if not self.trajectory or position != self.trajectory[-1]:
            self.trajectory.append(position)
        state = {
            "episode_id": self.episode_data["id"], "mode": self.mode,
            "pose": self.base.pose.tolist(), "velocity": self.base.velocity.tolist(),
            "radius": self.base.robot_radius, "trajectory": list(self.trajectory),
            "selected_start": self.selected_start, "reward": self.reward_sum,
            "step": self.base.step_count, "sim_time": self.base.step_count * self.base.dt,
            "info": self.last_info, "paused": self.paused, "speed": self.speed,
            "message": self.message, "finished": finished, "wall_time": time.time(),
            "last_result": self.last_result,
            "display_message": self.display_message,
        }
        try:
            if self.published_episode_id != self.episode_data["id"]:
                self.channel.put(episode=self.episode_data, state=state)
                self.published_episode_id = self.episode_data["id"]
            else:
                self.channel.put(state=state)
        except sqlite3.OperationalError:
            pass

    def step(self, action):
        # PPO已经根据旧观测选好动作；切换场景只能在真实step后truncated。
        if self.mode == "train":
            self.poll()
            while self.paused and self.single_steps == 0 and not self.reset_requested:
                time.sleep(0.02)
                self.poll()
        result = self.env.step(action)
        observation, reward, terminated, truncated, info = result
        if self.reset_requested:
            truncated = True
            info = {**info, "interactive_reset": True}
        self.reward_sum += reward
        self.last_info = info
        if terminated or truncated:
            self.last_result = {"episode": self.episode, "reward": self.reward_sum,
                                "goal": info["goal_reached"], "collision": info["collision"]}
        self.single_steps = max(0, self.single_steps - 1)
        self.publish(force=terminated or truncated)
        if self.speed > 0:
            remaining = self.base.dt / self.speed - (time.monotonic() - self.last_step)
            # 小片段等待，倍率很低时暂停/换速命令也能及时响应。
            while remaining > 0:
                time.sleep(min(remaining, 0.02))
                self.poll()
                if self.paused or self.reset_requested or self.speed == 0:
                    break
                remaining = self.base.dt / self.speed - (time.monotonic() - self.last_step)
        self.last_step = time.monotonic()
        return observation, reward, terminated, truncated, info

    def close(self):
        try:
            self.publish(force=True, finished=True)
        finally:
            self.channel.close()
            self.env.close()


def make_training_environment(config: dict, rank: int, monitor_dir: str,
                              session: str | None = None, speed: float = 0.0,
                              scenario: dict | None = None):
    from stable_baselines3.common.monitor import Monitor

    env = OmniNavigationEnv(config)
    if session is not None and rank == 0:
        env = RvizEnvironment(env, session, mode="train", speed=speed, scenario=scenario)
    elif scenario is not None:
        env = FixedScenarioEnvironment(env, scenario)
    return Monitor(env, str(Path(monitor_dir) / str(rank)))


class FixedScenarioEnvironment(gym.Wrapper):
    def __init__(self, env, options):
        super().__init__(env)
        self.options = options

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed, options=self.options if options is None else options)
