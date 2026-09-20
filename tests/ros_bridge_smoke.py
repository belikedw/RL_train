#!/usr/bin/env python3
"""使用ROS2系统Python运行的跨进程集成测试，不由普通pytest收集。"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Path as PathMessage
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_srvs.srv import SetBool, Trigger
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import MarkerArray

from rviz_bridge import RvizBridge


def main():
    with tempfile.TemporaryDirectory(prefix="rl_rviz_test_") as directory:
        session = Path(directory)
        os.environ.setdefault("ROS_LOG_DIR", str(session / "logs"))
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        log = (session / "environment.log").open("w+")
        process = subprocess.Popen(
            [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/run_navigation.py"),
             "--mode", "follow", "--paused", "--speed", "0", "--session", directory],
            env=environment, stdout=log, stderr=subprocess.STDOUT,
        )
        bridge = client = executor = None
        try:
            rclpy.init()
            bridge = RvizBridge(session)
            client = Node("rl_rviz_smoke_client")
            executor = SingleThreadedExecutor()
            executor.add_node(bridge)
            executor.add_node(client)
            received = {}
            qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            for kind, topic, profile in (
                (OccupancyGrid, "map", qos), (PathMessage, "global_path", qos),
                (PathMessage, "trajectory", 1), (MarkerArray, "markers", 1),
            ):
                client.create_subscription(kind, f"/rl_training/{topic}",
                                           lambda message, key=topic: received.update({key: message}), profile)
            client.create_subscription(TFMessage, "/tf", lambda message: received.update(tf=message), 10)
            start_pub = client.create_publisher(PoseWithCovarianceStamped, "/rl_training/initialpose", 10)
            goal_pub = client.create_publisher(PoseStamped, "/rl_training/goal_pose", 10)

            def wait(predicate, timeout=10.0):
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    executor.spin_once(timeout_sec=0.05)
                    if predicate():
                        return
                    if process.poll() is not None:
                        log.seek(0)
                        raise RuntimeError(log.read())
                raise AssertionError("等待ROS2消息/控制结果超时")

            wait(lambda: len(received) == 5)
            assert received["map"].info.width == 240
            assert received["map"].info.height == 200
            assert len(received["map"].data) == 48000
            assert any(t.child_frame_id == "rl_base_link" for t in received["tf"].transforms)
            wait(lambda: start_pub.get_subscription_count() > 0 and goal_pub.get_subscription_count() > 0)
            start = PoseWithCovarianceStamped()
            start.header.frame_id = "map"
            start.pose.pose.position.x = start.pose.pose.position.y = 2.0
            start.pose.pose.orientation.w = 1.0
            start_pub.publish(start)
            wait(lambda: bridge.channel.read().get("state", {}).get("selected_start") == [2.0, 2.0, 0.0])
            goal = PoseStamped()
            goal.header.frame_id = "map"
            goal.pose.position.x, goal.pose.position.y, goal.pose.orientation.w = 3.5, 2.0, 1.0
            goal_pub.publish(goal)
            wait(lambda: bridge.channel.read().get("episode", {}).get("goal") == [3.5, 2.0])
            wait(lambda: received["global_path"].poses[-1].pose.position.x == 3.5)
            state = bridge.channel.read()["state"]
            assert state["step"] == 0 and state["paused"]

            step_client = client.create_client(Trigger, "/rl_training/step")
            wait(lambda: step_client.service_is_ready())
            future = step_client.call_async(Trigger.Request())
            wait(future.done)
            assert future.result().success
            wait(lambda: bridge.channel.read()["state"]["step"] == 1)
            # 暂停状态下必须只推进一帧。
            until = time.monotonic() + 0.3
            while time.monotonic() < until:
                executor.spin_once(timeout_sec=0.05)
            assert bridge.channel.read()["state"]["step"] == 1

            pause_client = client.create_client(SetBool, "/rl_training/pause")
            wait(lambda: pause_client.service_is_ready())
            future = pause_client.call_async(SetBool.Request(data=False))
            wait(future.done)
            wait(lambda: bridge.channel.read()["state"]["step"] > 1)
            print("ROS2集成通过：地图、路径、轨迹、Marker、TF、鼠标起终点协议、暂停与单步")
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            if executor is not None:
                executor.shutdown()
            if client is not None:
                client.destroy_node()
            if bridge is not None:
                bridge.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            log.close()


if __name__ == "__main__":
    main()
