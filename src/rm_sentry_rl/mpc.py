from __future__ import annotations

import os
import select
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from rm_sentry_rl.envs.omni_navigation_env import OmniNavigationEnv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BRIDGE_NAME = "rm_mpc_teacher_bridge"


def find_bridge_binary(explicit_path: str | Path | None = None) -> Path:
    if explicit_path is not None:
        path = Path(explicit_path).expanduser().resolve()
        if path.is_file() and os.access(path, os.X_OK):
            return path
        raise FileNotFoundError(f"指定的C++ MPC桥接程序不可执行: {path}")
    candidates = [
        os.environ.get("RM_MPC_TEACHER_BRIDGE"),
        PROJECT_ROOT / "install/costmap_2d/lib/costmap_2d" / BRIDGE_NAME,
        PROJECT_ROOT / "build/costmap_2d" / BRIDGE_NAME,
        shutil.which(BRIDGE_NAME),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate).expanduser().resolve()
        if path.is_file() and os.access(path, os.X_OK):
            return path
    raise FileNotFoundError(
        "未找到C++ MPC桥接程序。先在仓库根目录执行 "
        "colcon build --packages-up-to costmap_2d --symlink-install"
    )


class OmniMpcTeacher:
    """通过常驻子进程直接调用仓库现有的C++ OmniMpcController。"""

    def __init__(self, env: OmniNavigationEnv, bridge_binary: str | Path | None = None):
        self.env = env
        self.binary = find_bridge_binary(bridge_binary)
        self.horizon = int(env.config["mpc"]["N"])
        self.timeout = float(env.config["mpc"].get("bridge_timeout_seconds", 10.0))
        if self.horizon < 1 or self.horizon > 500 or not np.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("MPC预测步数或桥接超时配置无效")

        self._temporary = tempfile.TemporaryDirectory(prefix="rm_mpc_teacher_")
        self._log_path = Path(self._temporary.name) / "bridge.log"
        self._map_path = Path(self._temporary.name) / "map.bin"
        self._process: subprocess.Popen[str] | None = None
        self._log = None
        try:
            self._write_map_snapshot()
            self._log = self._log_path.open("w+", encoding="utf-8")
        except Exception:
            self.close()
            raise

    def _start_process(self) -> None:
        child_env = os.environ.copy()
        libraries = [
            str(PROJECT_ROOT / "install/costmap_2d/lib"),
            "/opt/ros/humble/lib",
            "/usr/local/lib",
            child_env.get("LD_LIBRARY_PATH", ""),
        ]
        child_env["LD_LIBRARY_PATH"] = os.pathsep.join(part for part in libraries if part)
        self._process = subprocess.Popen(
            [str(self.binary), str(self._map_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=child_env,
        )

    def _write_map_snapshot(self) -> None:
        grid = self.env.grid_map
        with self._map_path.open("wb") as stream:
            stream.write(b"RMPC1")
            stream.write(
                struct.pack(
                    "<ii4d",
                    grid.width,
                    grid.height,
                    grid.resolution,
                    grid.origin_x,
                    grid.origin_y,
                    self.env.robot_radius,
                )
            )
            stream.write(np.asarray(grid.clearance, dtype="<f4", order="C").tobytes())

    def _exchange(self, message: str) -> list[str]:
        process = self._process
        if process is None or process.stdin is None or process.stdout is None:
            raise RuntimeError("C++ MPC桥接程序已关闭")
        try:
            process.stdin.write(message + "\n")
            process.stdin.flush()
        except BrokenPipeError as error:
            raise RuntimeError(f"C++ MPC桥接程序提前退出: {self._bridge_log()}") from error
        ready, _, _ = select.select([process.stdout], [], [], self.timeout)
        if not ready:
            raise TimeoutError(f"C++ MPC计算超过{self.timeout:g}秒")
        response = process.stdout.readline()
        if not response:
            raise RuntimeError(f"C++ MPC桥接程序提前退出: {self._bridge_log()}")
        return response.split()

    def _bridge_log(self) -> str:
        if self._log is not None:
            self._log.flush()
        if self._log_path.exists():
            return self._log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        return "无错误日志"

    def reset(self, seed: int | None = None) -> None:
        del seed  # C++ QP求解器为确定性控制器。
        if self.env.path.shape[0] < 2:
            raise RuntimeError("调用MPC前必须先重置训练环境")
        self._stop_process()
        if self._log is not None:
            self._log.seek(0)
            self._log.truncate()
        self._start_process()
        path_coordinates = " ".join(f"{x:.17g} {y:.17g}" for x, y in self.env.path)
        message = (
            f"RESET {self.env.dt:.17g} {self.horizon} {self.env.max_speed:.17g} "
            f"{self.env.max_acceleration:.17g} {self.env.spin_velocity:.17g} "
            f"{len(self.env.path)} {path_coordinates}"
        )
        if self._exchange(message) != ["OK"]:
            raise RuntimeError(f"C++ MPC重置失败: {self._bridge_log()}")

    def act(self) -> np.ndarray:
        pose = self.env.pose
        velocity = self.env.velocity
        response = self._exchange(
            f"STEP {pose[0]:.17g} {pose[1]:.17g} {pose[2]:.17g} "
            f"{velocity[0]:.17g} {velocity[1]:.17g}"
        )
        if len(response) != 5 or response[0] != "ACTION":
            raise RuntimeError(f"C++ MPC响应格式错误: {response}")
        vx, vy, spin = map(float, response[1:4])
        if response[4] != "1":
            raise RuntimeError(f"C++ MPC求解失败，已拒绝将回退动作写入示范: {self._bridge_log()}")
        if not np.all(np.isfinite([vx, vy, spin])):
            raise RuntimeError("C++ MPC返回非有限速度")
        if abs(spin - self.env.spin_velocity) > 1e-5:
            raise RuntimeError("C++ MPC角速度与当前训练回合配置不一致")
        return np.clip([vx / self.env.max_speed, vy / self.env.max_speed], -1.0, 1.0).astype(
            np.float32
        )

    def _stop_process(self) -> None:
        process = self._process
        self._process = None
        if process is not None:
            if process.poll() is None:
                try:
                    if process.stdin is not None:
                        process.stdin.write("QUIT\n")
                        process.stdin.flush()
                    process.wait(timeout=2.0)
                except (BrokenPipeError, subprocess.TimeoutExpired):
                    process.terminate()
                    try:
                        process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()

    def close(self) -> None:
        self._stop_process()
        if self._log is not None:
            self._log.close()
            self._log = None
        self._temporary.cleanup()

    def __enter__(self) -> OmniMpcTeacher:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
