# RM Sentry RL Training

独立的二维全向底盘强化学习工程。A*生成全局路径，PPO根据局部路径、地图障碍距离和速度状态输出二维底盘期望速度。支持RViz交互规划、运动演示、策略评估及训练观察，不需要雷达、Gazebo或实车。

所有命令均在本仓库根目录执行。ROS2仅用于可选的可视化桥接，训练核心不导入rclpy。

## 安装

推荐Ubuntu 22.04 + ROS2 Humble；Python支持3.10～3.12。桥接使用系统Python，训练使用独立虚拟环境。Jazzy接口兼容，但尚未在Jazzy实机验证。

```bash
python3 -m venv .venv
env -u PYTHONPATH .venv/bin/python -m pip install --upgrade pip setuptools wheel
env -u PYTHONPATH .venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
env -u PYTHONPATH .venv/bin/python -m pip install -e '.[train,test]'
```

RViz功能需要ROS2的rviz2、rclpy、nav_msgs、geometry_msgs、visualization_msgs、std_srvs和tf2_ros。ROS2桌面安装通常已具备这些包；不需要构建colcon工作空间。

## 启动RViz

```bash
source /opt/ros/humble/setup.bash
python3 scripts/launch_rviz.py --mode plan
```

默认加载仓库内的 `maps/demo.yaml`：12m × 10m，0.05m分辨率。用工具栏的 **2D Pose Estimate** 选择起点和朝向，再用 **2D Goal Pose** 选择终点。绿色为全局路径，橙色为实际轨迹，蓝色圆形为机器人碰撞轮廓。

无需已有模型即可查看运动：

```bash
python3 scripts/launch_rviz.py --mode follow
```

follow使用简单前视跟踪器验证运动显示，不是MPC教师，也不保证所有路径都成功通过。碰撞、到达或超时后保留终态；设置新目标或调用重置服务开始下一回合。加 `--auto-reset` 可自动重置。

在RViz中观察训练：

```bash
python3 scripts/launch_rviz.py --mode train --num-envs 1 --timesteps 100000 --run-name rviz_train --speed 1
```

`--speed 1` 按真实时间推进；`--speed 0` 不限速，适合正式训练。多进程训练只显示第0号环境；RViz交互也只影响这个环境。同步PPO中，暂停它会让整个向量采样等待。

模型评估：

```bash
python3 scripts/launch_rviz.py --mode evaluate --model artifacts/rviz_train/final_model.zip
```

关闭RViz窗口不会停止环境；在启动终端按Ctrl+C结束整组进程，训练中断时保存interrupted_model.zip。支持 `--no-rviz` 仅运行环境和桥接，或用 `--session artifacts/rviz_other` 隔离通信文件。多套ROS显示同时运行时还应分别设置不同的ROS_DOMAIN_ID。

完整控制命令、独立启动方式及接口说明见 [RViz使用说明](docs/rviz.md)。

## 更换地图与场景

修改 `configs/env.yaml` 的 `map.yaml_path`。相对路径基于该配置文件父目录的父目录解析；默认配置对应本仓库根目录，也可以填写绝对路径。未知区域默认不可通行，地图原点暂不支持旋转。

第一版只支持本地地图和内部A*，不提供外部全局路径输入。可保存当前起终点，然后在相同地图与环境配置下加载：

```bash
ros2 service call /rl_training/save std_srvs/srv/Trigger '{}'
python3 scripts/launch_rviz.py --mode follow --scenario artifacts/rviz/scenario.json
```

手动场景会在后续回合重复使用；调用random服务恢复随机起终点。正式训练应覆盖多组场景，避免只记住一条路径。

## 无界面训练与验证

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests
env -u PYTHONPATH .venv/bin/python scripts/check_env.py
env -u PYTHONPATH .venv/bin/python scripts/train.py --timesteps 2048 --num-envs 1 --run-name smoke_test
env -u PYTHONPATH .venv/bin/python scripts/evaluate.py artifacts/smoke_test/final_model.zip --episodes 10
```

正式训练不指定timesteps时读取 `configs/ppo.yaml`。输出位于 `artifacts/<run-name>/`，包含模型、监控CSV、TensorBoard及周期性评估结果。短训练不足评估周期时只有final_model.zip，没有最佳模型。

```bash
env -u PYTHONPATH .venv/bin/tensorboard --logdir artifacts/rviz_train/tensorboard
```

ROS2集成检查使用系统Python，并在独立DDS域运行：

```bash
source /opt/ros/humble/setup.bash
ROS_DOMAIN_ID=87 ROS_LOCALHOST_ONLY=1 /usr/bin/python3 tests/ros_bridge_smoke.py
/usr/bin/python3 tests/launcher_shutdown_smoke.py
```

## MPC示范与ONNX

MPC教师仍需单独构建原导航工程中的C++桥接程序。本仓库不包含该程序，也不需要它来运行RViz或纯PPO训练。

```bash
env -u PYTHONPATH .venv/bin/python scripts/collect_mpc.py --bridge-binary /absolute/path/to/rm_mpc_teacher_bridge --episodes 20
env -u PYTHONPATH .venv/bin/python scripts/train.py --mpc-dataset artifacts/mpc_demonstrations.npz --run-name ppo_mpc_bc
env -u PYTHONPATH .venv/bin/python scripts/export_onnx.py artifacts/ppo_mpc_bc/final_model.zip
```

如桥接依赖原工程动态库，启动前加载其ROS2安装环境。示范仅用于行为克隆初始化，之后PPO继续与二维环境交互。示范默认只保存成功回合，配置变更后应重新采集。

本次更新采用全分辨率A*、保守障碍净空和沿途碰撞检测；局部路径改为按弧长插值，观测仍为97维，但observation_version升级为2，环境指纹也已更新。旧示范会被拒绝，旧模型应重新训练；部署端也必须使用相同的观测协议。拆分前旧地图上的历史训练指标不适用于当前版本。

导出包含policy.onnx、model_manifest.yaml与golden_io.npz。二维地图训练结果只能证明该环境内的行为，实车部署仍需独立验证。
