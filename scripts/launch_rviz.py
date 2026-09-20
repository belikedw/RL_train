#!/usr/bin/env python3
"""管理训练Python、ROS2桥接与RViz的进程生命周期。"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="启动无雷达RViz导航与训练；先source ROS2 setup.bash")
    parser.add_argument("--mode", choices=("plan", "follow", "evaluate", "train"), default="plan")
    parser.add_argument("--session", type=Path, default=PROJECT_ROOT / "artifacts/rviz")
    parser.add_argument("--env-config", type=Path, default=PROJECT_ROOT / "configs/env.yaml")
    parser.add_argument("--ros-python", default="/usr/bin/python3")
    parser.add_argument("--train-python", default=str(PROJECT_ROOT / ".venv/bin/python"))
    parser.add_argument("--speed", type=float, default=None)
    parser.add_argument("--no-rviz", action="store_true", help="仅启动环境与桥接，用于无显示器测试")
    args, extra = parser.parse_known_args()
    if not Path(args.train_python).is_file():
        parser.error("找不到训练Python，请先按照README创建.venv或传--train-python")
    if not os.environ.get("ROS_DISTRO"):
        parser.error("请先source /opt/ros/humble/setup.bash（或Jazzy）")
    rviz = shutil.which("rviz2")
    if not args.no_rviz and rviz is None:
        parser.error("未找到rviz2，请安装并加载ROS2环境")
    ros_environment = os.environ.copy()
    ros_environment["PYTHONNOUSERSITE"] = "1"
    train_environment = ros_environment.copy()
    train_environment.pop("PYTHONPATH", None)
    base = ["--env-config", str(args.env_config)]
    if args.speed is not None:
        base += ["--speed", str(args.speed)]
    if args.mode == "train":
        command = [args.train_python, str(PROJECT_ROOT / "scripts/train.py"), "--rviz",
                   "--rviz-session", str(args.session), *base, *extra]
    else:
        command = [args.train_python, str(PROJECT_ROOT / "scripts/run_navigation.py"),
                   "--mode", args.mode, "--session", str(args.session), *base, *extra]
    processes: list[subprocess.Popen] = []
    try:
        processes.append(subprocess.Popen(command, env=train_environment, start_new_session=True))
        processes.append(subprocess.Popen(
            [args.ros_python, str(PROJECT_ROOT / "scripts/rviz_bridge.py"), "--session", str(args.session)],
            env=ros_environment, start_new_session=True,
        ))
        view = None
        if not args.no_rviz:
            view = subprocess.Popen([rviz, "-d", str(PROJECT_ROOT / "configs/navigation.rviz")],
                                    env=ros_environment, start_new_session=True)
            processes.append(view)
        while processes[0].poll() is None:
            if processes[1].poll() is not None:
                raise RuntimeError(f"ROS2桥接退出：{processes[1].returncode}")
            if view is not None and view.poll() is not None:
                print("RViz窗口已关闭，环境继续运行；Ctrl+C结束。", flush=True)
                view = None
            time.sleep(0.2)
        if processes[0].returncode:
            raise RuntimeError(f"环境进程退出：{processes[0].returncode}")
    except KeyboardInterrupt:
        pass
    finally:
        for process in processes:
            if process.poll() is None:
                # 先通知主进程，由训练端收尾子环境，避免同时中断PPO worker。
                process.send_signal(signal.SIGINT)
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


if __name__ == "__main__":
    main()
