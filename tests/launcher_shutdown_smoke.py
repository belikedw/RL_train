#!/usr/bin/env python3
"""验证暂停中的双进程PPO可由启动器正常中断、保存和退出。"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rm_sentry_rl.visualization_transport import VisualizationChannel


def main():
    with tempfile.TemporaryDirectory(prefix="rl_launcher_test_") as directory:
        root = Path(directory)
        environment = os.environ.copy()
        environment.update(ROS_DOMAIN_ID="89", ROS_LOCALHOST_ONLY="1",
                           ROS_LOG_DIR=str(root / "logs"), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with (root / "output.log").open("w+") as log:
            process = subprocess.Popen(
                ["/usr/bin/python3", str(ROOT / "scripts/launch_rviz.py"), "--mode", "train",
                 "--no-rviz", "--num-envs", "2", "--timesteps", "100000", "--speed", "1",
                 "--session", str(root / "session"), "--run-name", str(root / "model")],
                env=environment, stdout=log, stderr=subprocess.STDOUT,
            )
            try:
                with VisualizationChannel(root / "session") as channel:
                    deadline = time.monotonic() + 30
                    while "state" not in channel.read() and time.monotonic() < deadline:
                        if process.poll() is not None:
                            break
                        time.sleep(0.1)
                    assert "state" in channel.read(), "训练环境未就绪"
                    channel.send("pause", True)
                    deadline = time.monotonic() + 5
                    while not channel.read()["state"]["paused"] and time.monotonic() < deadline:
                        time.sleep(0.1)
                    assert channel.read()["state"]["paused"]
                    process.send_signal(signal.SIGINT)
                    process.wait(timeout=12)
                    assert process.returncode == 0
                    assert (root / "model/interrupted_model.zip").is_file(), "中断模型未保存"
                    assert channel.read()["state"]["finished"], "训练worker未正常关闭"
                print("启动器退出测试通过：双worker暂停中断、模型保存、环境关闭")
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                    process.wait(timeout=12)
                log.seek(0)
                print(log.read())


if __name__ == "__main__":
    main()
