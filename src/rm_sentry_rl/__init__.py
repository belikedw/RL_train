"""RM哨兵强化学习训练环境。"""

__all__ = ["OmniNavigationEnv"]


def __getattr__(name: str):
    # ROS2桥接仅使用标准库传输层，不应隐式加载Gym、Torch等训练依赖。
    if name == "OmniNavigationEnv":
        from rm_sentry_rl.envs.omni_navigation_env import OmniNavigationEnv

        return OmniNavigationEnv
    raise AttributeError(name)
