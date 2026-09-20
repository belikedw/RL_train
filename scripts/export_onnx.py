#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import yaml
from stable_baselines3 import PPO

from rm_sentry_rl.config import load_config
from rm_sentry_rl.envs.omni_navigation_env import OmniNavigationEnv
from rm_sentry_rl.observation import ObservationBuilder


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DeterministicPolicy(torch.nn.Module):
    def __init__(self, policy: torch.nn.Module):
        super().__init__()
        self.policy = policy

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        actions, _, _ = self.policy(observation, deterministic=True)
        return torch.clamp(actions, -1.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="导出PPO策略并验证ONNX输出")
    parser.add_argument("model", type=Path)
    parser.add_argument("--env-config", type=Path, default=PROJECT_ROOT / "configs/env.yaml")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/exported")
    args = parser.parse_args()

    config = load_config(args.env_config)
    env = OmniNavigationEnv(config)
    observation_dim = env.observation_space.shape[0]
    model = PPO.load(args.model, device="cpu")
    wrapper = DeterministicPolicy(model.policy).eval()

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "policy.onnx"
    dummy_input = torch.zeros((1, observation_dim), dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        dummy_input,
        model_path,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    rng = np.random.default_rng(2026)
    golden_input = rng.uniform(-1.0, 1.0, size=(1, observation_dim)).astype(np.float32)
    with torch.no_grad():
        torch_output = wrapper(torch.from_numpy(golden_input)).numpy()

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    onnx_output = session.run(["action"], {"observation": golden_input})[0]
    max_error = float(np.max(np.abs(torch_output - onnx_output)))
    if max_error > 1e-4:
        raise RuntimeError(f"ONNX输出校验失败，最大误差: {max_error}")

    np.savez(
        args.output / "golden_io.npz",
        observation=golden_input,
        action=onnx_output,
    )
    manifest = {
        "model_version": 1,
        "observation_version": ObservationBuilder.VERSION,
        "observation_dim": int(observation_dim),
        "action_dim": 2,
        "control_dt": float(config["environment"]["dt"]),
        "coordinate_frame": "base_link",
        "action_type": "normalized_desired_velocity",
        "max_speed": float(config["robot"]["max_speed"]),
        "onnx_opset": 17,
        "validation_max_abs_error": max_error,
    }
    with (args.output / "model_manifest.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(manifest, stream, allow_unicode=True, sort_keys=False)

    env.close()
    print(f"ONNX导出完成: {model_path}")
    print(f"PyTorch/ONNX最大误差: {max_error:.8f}")


if __name__ == "__main__":
    main()
