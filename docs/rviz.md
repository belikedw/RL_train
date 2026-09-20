# 无雷达RViz导航与训练

## 数据流

OmniNavigationEnv负责动力学、地图障碍观测、碰撞、奖励和终止条件。RvizEnvironment在环境主线程中处理控制请求，将状态写入本机SQLite WAL快照。ROS2系统Python运行rviz_bridge.py，读取快照并发布消息，训练虚拟环境无需ROS2 Python依赖。

地图仅在启动或更换运行实例时发布；路径每回合发布；状态以10Hz发布。轨迹最多保留1000个显示采样点，命令队列最多64条，超过30秒未执行的命令拒绝执行。一个session只允许一个环境生产者。SQLite目录应放在本机磁盘；此接口不提供网络服务、外部地图或外部全局路径输入。

## 操作流程

1. 运行 `python3 scripts/launch_rviz.py --mode plan`，显示地图和一条随机可达路径。
2. 使用2D Pose Estimate点击并拖动以设置起点和朝向，紫色箭头表示选中的起点。
3. 使用2D Goal Pose选择终点。绿色路径由内部A*生成，目标yaw不作为全向底盘的终点约束。
4. 起终点越界、碰撞或不可达时保留原场景，桥接终端和RViz状态提示错误。
5. 选择 `--mode follow` 查看简单跟踪运动，或 `--mode evaluate --model <zip>` 查看训练模型。

plan只规划，不推进机器人。follow默认不自旋；训练和随机策略评估沿用env.yaml的自旋随机化。手动点击创建的场景默认不自旋。机器人位姿由环境直接提供，不运行SLAM、AMCL或里程计估计，也不发布或订阅LaserScan。97维观测中的64个方向距离直接从已知静态地图计算。

## 暂停、重置与保存

在已加载ROS2环境、且ROS_DOMAIN_ID与启动器相同的终端执行：

```bash
# 暂停 / 继续
ros2 service call /rl_training/pause std_srvs/srv/SetBool '{data: true}'
ros2 service call /rl_training/pause std_srvs/srv/SetBool '{data: false}'

# 暂停并推进一次环境动作，完成后仍然暂停
ros2 service call /rl_training/step std_srvs/srv/Trigger '{}'

# 重置当前场景 / 恢复随机起终点
ros2 service call /rl_training/reset std_srvs/srv/Trigger '{}'
ros2 service call /rl_training/random std_srvs/srv/Trigger '{}'

# 实时倍率；0.0表示不限速
ros2 param set /rl_rviz_bridge speed 1.0
ros2 param set /rl_rviz_bridge speed 0.0

# 保存到session/scenario.json，再次调用会更新该文件
ros2 service call /rl_training/save std_srvs/srv/Trigger '{}'
```

服务成功仅表示请求入队，实际执行结果由环境处理后显示。save保存当前生效的起终点和自旋速度，不保存尚未应用的目标、模型、优化器或随机数状态；这是场景复现文件，不是训练断点。

```bash
python3 scripts/launch_rviz.py --mode follow --scenario artifacts/rviz/scenario.json
python3 scripts/launch_rviz.py --mode train --scenario artifacts/rviz/scenario.json --num-envs 1 --run-name fixed_scene
```

场景文件绑定地图与环境配置。多环境训练中，启动时的scenario应用到所有训练环境；运行中的RViz目标只影响第0号环境。评估回调依然使用随机场景。

训练中不在ROS回调内reset环境。已有观测对应的动作先完成一次有效step，再返回truncated=True，由SB3保存终态观测并自动reset到新场景。暂停只暂停环境采样，正在进行的一次策略优化可能继续到结束；同步向量采样会等待暂停的环境。

## 独立启动

训练核心：

```bash
env -u PYTHONPATH .venv/bin/python scripts/run_navigation.py --mode follow --paused
# 或
env -u PYTHONPATH .venv/bin/python scripts/train.py --rviz --num-envs 1 --speed 1 --run-name rviz_train
```

桥接与RViz分别运行：

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 scripts/rviz_bridge.py --session artifacts/rviz
rviz2 -d configs/navigation.rviz
```

桥接重启或关闭RViz不会改变训练状态。重新打开RViz后，地图和路径使用Transient Local QoS补发。默认固定坐标系为map，TF为map到rl_base_link，不发送底盘控制话题。

| 话题 | 类型 | 用途 |
| --- | --- | --- |
| `/rl_training/map` | nav_msgs/OccupancyGrid | 静态地图 |
| `/rl_training/global_path` | nav_msgs/Path | 当前A*路径 |
| `/rl_training/trajectory` | nav_msgs/Path | 最近的实际轨迹 |
| `/rl_training/markers` | visualization_msgs/MarkerArray | 机器人、起终点、状态 |
| `/rl_training/initialpose` | geometry_msgs/PoseWithCovarianceStamped | RViz起点输入 |
| `/rl_training/goal_pose` | geometry_msgs/PoseStamped | RViz终点输入 |

显示消息与TF使用桥接进程的ROS时间，以便训练加速、暂停和回合重置时仍可正常显示。不发布/clock；默认不要给RViz或桥接启用use_sim_time。每回合仿真时间由步数乘以dt计算，与显示时间分开。

## 实现与性能

规划使用完整栅格的八邻域A*和欧氏启发式；距离场结合机器人半径和安全余量划定可通行区域，对角扩展禁止切角。设栅格总数为G，最坏时间复杂度O(G log G)、空间O(G)。astar_stride只用于随机起终点采样，不降低搜索分辨率。

重采样保留拐点，不另加平滑。每次运动按不超过半栅格的间隔检查足迹，并加入覆盖采样间隙的检查余量；碰撞时拒绝该步平移并结束回合。该方法保守，适合静态地图与圆形全向底盘。

显示读写限频，短暂数据库繁忙时允许丢弃显示帧；地图与路径使用独立快照。不限速训练时RViz仅显示抽样帧，可能跳过快速完成的回合；状态保留最近一次回合结果。慢速观察使用 `--num-envs 1 --speed 1`。

地图与碰撞逻辑变更会改变训练分布。本版本观测维度仍为97，但路径采样协议版本为2，旧模型需要重新训练。
