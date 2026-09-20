#!/usr/bin/env python3
"""使用ROS2系统Python启动；通过本机快照读取独立训练环境。"""
from __future__ import annotations

import argparse
import math
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Path as PathMessage
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import SetBool, Trigger
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from rm_sentry_rl.visualization_transport import VisualizationChannel


class RvizBridge(Node):
    def __init__(self, session: Path):
        super().__init__("rl_rviz_bridge")
        self.channel = VisualizationChannel(session)
        self.declare_parameter("speed", -1.0)
        self.add_on_set_parameters_callback(self.on_parameters)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub = self.create_publisher(OccupancyGrid, "/rl_training/map", latched)
        self.path_pub = self.create_publisher(PathMessage, "/rl_training/global_path", latched)
        self.track_pub = self.create_publisher(PathMessage, "/rl_training/trajectory", 1)
        self.marker_pub = self.create_publisher(MarkerArray, "/rl_training/markers", 1)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(PoseWithCovarianceStamped, "/rl_training/initialpose", self.on_start, 10)
        self.create_subscription(PoseStamped, "/rl_training/goal_pose", self.on_goal, 10)
        self.create_service(SetBool, "/rl_training/pause", self.on_pause)
        for name in ("step", "reset", "random", "save"):
            self.create_service(Trigger, f"/rl_training/{name}", self.trigger_callback(name))
        self.map_id = self.episode_id = self.last_message = None
        self.grid = None
        self.create_timer(0.1, self.tick)
        self.get_logger().info(f"等待训练环境：{session}")

    def send(self, kind, value=None):
        try:
            self.channel.send(kind, value)
            return True, "请求已入队；实际执行结果见RViz状态与桥接日志"
        except (RuntimeError, ValueError, sqlite3.Error) as error:
            return False, str(error)

    def on_parameters(self, parameters):
        for parameter in parameters:
            if parameter.name == "speed":
                value = parameter.value
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    return SetParametersResult(successful=False, reason="speed应>=0，0表示不限速")
                success, message = self.send("speed", value)
                if not success:
                    return SetParametersResult(successful=False, reason=message)
        return SetParametersResult(successful=True)

    def on_pause(self, request, response):
        response.success, response.message = self.send("pause", request.data)
        return response

    def trigger_callback(self, kind):
        def callback(request, response):
            response.success, response.message = self.send(kind)
            return response
        return callback

    def check_pose(self, header, pose):
        if header.frame_id != "map":
            raise ValueError("起终点必须位于map坐标系，请检查RViz Fixed Frame")
        q = pose.orientation
        values = (pose.position.x, pose.position.y, q.x, q.y, q.z, q.w)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("位姿包含非有限数值")
        norm = math.sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w)
        if norm < 1e-6:
            raise ValueError("位姿四元数无效")
        x, y, z, w = (v / norm for v in (q.x, q.y, q.z, q.w))
        yaw = math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
        return [pose.position.x, pose.position.y, yaw]

    def on_start(self, message):
        try:
            success, detail = self.send("start", self.check_pose(message.header, message.pose.pose))
            if not success:
                self.get_logger().warning(detail)
        except ValueError as error:
            self.get_logger().warning(str(error))

    def on_goal(self, message):
        try:
            success, detail = self.send("goal", self.check_pose(message.header, message.pose)[:2])
            if not success:
                self.get_logger().warning(detail)
        except ValueError as error:
            self.get_logger().warning(str(error))

    @staticmethod
    def path_message(points, stamp):
        message = PathMessage()
        message.header.frame_id = "map"
        message.header.stamp = stamp
        for x, y in points:
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x, pose.pose.position.y = float(x), float(y)
            pose.pose.orientation.w = 1.0
            message.poses.append(pose)
        return message

    @staticmethod
    def marker(index, kind, stamp, color):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = stamp
        marker.ns, marker.id, marker.type = "rl_training", index, kind
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        return marker

    def tick(self):
        try:
            snapshot = self.channel.read(("episode", "state"))
        except sqlite3.Error:
            return
        if not all(key in snapshot for key in ("episode", "state")):
            return
        episode, state = (snapshot[key] for key in ("episode", "state"))
        if episode["run_id"] != self.map_id:
            try:
                self.grid = self.channel.read(("map",)).get("map")
            except sqlite3.Error:
                return
        grid = self.grid
        if grid is None or grid["run_id"] != episode["run_id"]:
            return
        stamp = self.get_clock().now().to_msg()
        if grid["run_id"] != self.map_id:
            message = OccupancyGrid()
            message.header.frame_id, message.header.stamp = "map", stamp
            message.info.resolution = grid["resolution"]
            message.info.width, message.info.height = grid["width"], grid["height"]
            message.info.origin.position.x, message.info.origin.position.y = grid["origin"]
            message.info.origin.orientation.w = 1.0
            message.data = grid["data"]
            self.map_pub.publish(message)
            self.map_id = grid["run_id"]
        if episode["id"] != self.episode_id:
            self.path_pub.publish(self.path_message(episode["path"], stamp))
            self.episode_id = episode["id"]
        self.track_pub.publish(self.path_message(state["trajectory"], stamp))
        x, y, yaw = state["pose"]
        transform = TransformStamped()
        transform.header.frame_id, transform.header.stamp = "map", stamp
        transform.child_frame_id = "rl_base_link"
        transform.transform.translation.x, transform.transform.translation.y = x, y
        transform.transform.rotation.z, transform.transform.rotation.w = math.sin(yaw/2), math.cos(yaw/2)
        self.tf.sendTransform(transform)

        collided = state["info"].get("collision", False)
        body = self.marker(0, Marker.CYLINDER, stamp, (1.0, 0.1, 0.1, 0.8) if collided
                           else (0.1, 0.7, 1.0, 0.7))
        body.pose.position.x, body.pose.position.y, body.pose.position.z = x, y, 0.08
        body.scale.x = body.scale.y = 2 * state["radius"]
        body.scale.z = 0.16
        arrow = self.marker(1, Marker.ARROW, stamp, (1.0, 0.8, 0.0, 1.0))
        arrow.pose = body.pose
        arrow.pose.orientation = transform.transform.rotation
        arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.65, 0.08, 0.08
        goal = self.marker(2, Marker.SPHERE, stamp, (0.1, 1.0, 0.2, 0.9))
        goal.pose.position.x, goal.pose.position.y = episode["goal"]
        goal.scale.x = goal.scale.y = goal.scale.z = 0.25
        start = self.marker(3, Marker.ARROW, stamp, (0.8, 0.3, 1.0, 1.0))
        sx, sy, syaw = state["selected_start"] or episode["start"]
        start.pose.position.x, start.pose.position.y = sx, sy
        start.pose.orientation.z, start.pose.orientation.w = math.sin(syaw/2), math.cos(syaw/2)
        start.scale.x, start.scale.y, start.scale.z = 0.7, 0.1, 0.1
        text = self.marker(4, Marker.TEXT_VIEW_FACING, stamp, (1.0, 1.0, 1.0, 1.0))
        text.pose.position.x = grid["origin"][0] + grid["width"] * grid["resolution"] / 2
        text.pose.position.y = grid["origin"][1] + grid["height"] * grid["resolution"] + 0.6
        text.scale.z = 0.22
        status = "FINISHED" if state["finished"] else "PAUSED" if state["paused"] else "RUNNING"
        if time.time() - state["wall_time"] > 3 and not state["finished"]:
            status = "WAITING FOR ENVIRONMENT"
        vx, vy, _ = state["velocity"]
        previous = state.get("last_result", {})
        previous_text = (f"Last #{previous['episode']}: goal={previous['goal']} collision={previous['collision']}"
                         if previous else "No completed episodes yet")
        # RViz默认字体不包含中文；画面用英文，完整中文原因写入桥接日志。
        text.text = (f"{state['mode'].upper()} | {status} | Episode {episode['number']} | Step {state['step']}\n"
                     f"Reward {state['reward']:.2f} | Speed {math.hypot(vx, vy):.2f} m/s | "
                     f"Path error {state['info'].get('path_error', 0):.2f} m\n"
                     f"Goal {state['info'].get('goal_reached', False)} | Collision {collided}\n"
                     f"{previous_text}\n{state.get('display_message', '')}")
        self.marker_pub.publish(MarkerArray(markers=[body, arrow, goal, start, text]))
        if state["message"] != self.last_message:
            self.get_logger().info(state["message"])
            self.last_message = state["message"]

    def destroy_node(self):
        self.channel.close()
        return super().destroy_node()


def main():
    parser = argparse.ArgumentParser(description="RViz训练桥接（请使用ROS2系统Python）")
    parser.add_argument("--session", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts/rviz")
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = None
    try:
        node = RvizBridge(args.session)
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
