#!/usr/bin/env python3
"""Export tau models to static ONNX and benchmark fp32 inference."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import statistics
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from mltau.models.MixerTau import MixerTau
from mltau.models.MultiParTau import ParTau as MultiParTau
from mltau.models.SingleParTau import ParTau as SingleParTau


class SingleTaskExportWrapper(nn.Module):
    """Expose the single tensor returned by SingleParTau."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        cand_features: torch.Tensor,
        cand_kinematics_pxpypze: torch.Tensor,
        cand_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(cand_features, cand_kinematics_pxpypze, cand_mask)[0]


class MixerExportWrapper(nn.Module):
    """Expose MixerTau without its unused kinematics argument."""

    def __init__(self, model: MixerTau):
        super().__init__()
        self.model = model

    def forward(
        self,
        cand_features: torch.Tensor,
        cand_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            cand_features, cand_kinematics_pxpypze=None, cand_mask=cand_mask
        )[0]


class MultiTaskExportWrapper(nn.Module):
    """Flatten MultiParTau's four task outputs into one benchmark tensor."""

    def __init__(self, model: MultiParTau):
        super().__init__()
        self.model = model

    def forward(
        self,
        cand_features: torch.Tensor,
        cand_kinematics_pxpypze: torch.Tensor,
        cand_mask: torch.Tensor,
    ) -> torch.Tensor:
        output = self.model(
            cand_features, cand_kinematics_pxpypze, cand_mask
        )
        return torch.cat(
            (
                output["is_tau"],
                output["charge"].unsqueeze(-1),
                output["decay_mode"],
                output["kinematics"],
            ),
            dim=-1,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a fixed-shape fp32 ONNX graph and benchmark it with "
            "ONNX Runtime."
        )
    )
    parser.add_argument(
        "model",
        choices=("singlepartau", "multipartau", "mixer", "all"),
        help=(
            "Model to benchmark. "
            "'all' benchmarks SingleParTau, MultiParTau, and Mixer."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--input-dim", type=int, default=17)
    parser.add_argument("--num-particles", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--task",
        default="is_tau",
        choices=("is_tau", "charge", "decay_mode", "kinematics"),
    )
    parser.add_argument("--num-dm-classes", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument(
        "--devices",
        nargs="+",
        choices=("cpu", "gpu"),
        default=["cpu", "gpu"],
        help="Runtime targets to benchmark (default: cpu gpu).",
    )
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--seed", type=int, default=12345)

    # ParTau architecture defaults match the training modules.
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-cls-layers", type=int, default=2)
    parser.add_argument(
        "--embed-dims", type=int, nargs="+", default=[256, 512, 256]
    )
    parser.add_argument(
        "--pair-embed-dims", type=int, nargs="+", default=[64, 64, 64]
    )
    parser.add_argument("--num-heads", type=int, default=8)

    # MLP-Mixer backbone setting.
    parser.add_argument("--mixer-embed-dim", type=int, default=128)
    return parser.parse_args()


def make_model_and_inputs(
    args: argparse.Namespace,
    model_name: str,
) -> tuple[nn.Module, tuple[torch.Tensor, ...], list[str]]:
    shape = (args.batch_size, args.input_dim, args.num_particles)
    features = torch.randn(shape, dtype=torch.float32)
    mask = torch.ones(
        args.batch_size, 1, args.num_particles, dtype=torch.bool
    )

    kinematics = torch.randn(
        args.batch_size, 4, args.num_particles, dtype=torch.float32
    )
    # Keep E positive and away from singular kinematic configurations.
    kinematics[:, 3, :] = (
        torch.linalg.vector_norm(kinematics[:, :3, :], dim=1) + 1.0
    )

    if model_name == "singlepartau":
        model = SingleParTau(
            input_dim=args.input_dim,
            task=args.task,
            num_dm_classes=args.num_dm_classes,
            embed_dims=args.embed_dims,
            pair_embed_dims=args.pair_embed_dims,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            num_cls_layers=args.num_cls_layers,
            use_pre_activation_pair=False,
            for_inference=True,
            use_amp=False,
            metric="theta-phi",
        )
        model = SingleTaskExportWrapper(model)
    elif model_name == "multipartau":
        model = MultiParTau(
            input_dim=args.input_dim,
            num_dm_classes=args.num_dm_classes,
            embed_dims=args.embed_dims,
            pair_embed_dims=args.pair_embed_dims,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            num_cls_layers=args.num_cls_layers,
            use_pre_activation_pair=False,
            for_inference=True,
            use_amp=False,
            metric="theta-phi",
        )
        model = MultiTaskExportWrapper(model)
    elif model_name == "mixer":
        model = MixerTau(
            input_dim=args.input_dim,
            task=args.task,
            n_constituents=args.num_particles,
            num_dm_classes=args.num_dm_classes,
            embed_dim=args.mixer_embed_dim,
        )
        model = MixerExportWrapper(model)
        inputs = (features, mask)
        names = ["cand_features", "cand_mask"]
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if model_name != "mixer":
        inputs = (features, kinematics, mask)
        names = [
            "cand_features",
            "cand_kinematics_pxpypze",
            "cand_mask",
        ]

    return model.eval(), inputs, names


def load_checkpoint(model: nn.Module, checkpoint: Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    target_keys = set(model.state_dict())
    prefixes = (
        "",
        "model.",
        "ParTau.",
        "model.ParTau.",
        "backbone.",
        "ParTau.backbone.",
        "model.ParTau.backbone.",
    )

    best_state: dict[str, torch.Tensor] = {}
    for prefix in prefixes:
        candidate = {}
        for key, value in state.items():
            if not key.startswith(prefix):
                continue
            stripped = key[len(prefix):]
            for target_key in (stripped, f"model.{stripped}"):
                if target_key in target_keys:
                    candidate[target_key] = value
                    break
        if len(candidate) > len(best_state):
            best_state = candidate

    if not best_state:
        raise RuntimeError(
            f"No checkpoint tensors in {checkpoint} match the selected model."
        )
    missing, unexpected = model.load_state_dict(best_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not exactly match the selected architecture. "
            f"Missing keys: {missing}; unexpected keys: {unexpected}"
        )


def export_onnx(
    model: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    input_names: list[str],
    output_path: Path,
    opset: int,
) -> None:
    import onnx

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            inputs,
            output_path,
            input_names=input_names,
            output_names=["output"],
            opset_version=opset,
            dynamo=False,
            do_constant_folding=True,
        )
    onnx.checker.check_model(output_path)


def _shape(value_info) -> tuple[int, ...] | None:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    dims = []
    for dim in tensor_type.shape.dim:
        if not dim.HasField("dim_value"):
            return None
        dims.append(dim.dim_value)
    return tuple(dims)


def count_onnx_macs(path: Path) -> tuple[int, dict[str, int]]:
    """Count multiply-accumulates in static MatMul, Gemm, and Conv nodes."""
    import onnx

    model = onnx.shape_inference.infer_shapes(onnx.load(path))
    shapes = {}
    for value in (
        list(model.graph.input)
        + list(model.graph.value_info)
        + list(model.graph.output)
    ):
        shape = _shape(value)
        if shape is not None:
            shapes[value.name] = shape
    for initializer in model.graph.initializer:
        shapes[initializer.name] = tuple(initializer.dims)

    by_op: dict[str, int] = {"MatMul": 0, "Gemm": 0, "Conv": 0}
    for node in model.graph.node:
        if node.op_type == "MatMul":
            a, b = shapes.get(node.input[0]), shapes.get(node.input[1])
            out = shapes.get(node.output[0])
            if a and b and out and len(a) >= 2 and len(b) >= 2:
                by_op["MatMul"] += math.prod(out) * a[-1]
        elif node.op_type == "Gemm":
            a, b = shapes.get(node.input[0]), shapes.get(node.input[1])
            out = shapes.get(node.output[0])
            if a and b and out and len(a) == 2 and len(b) == 2:
                attrs = {
                    attr.name: onnx.helper.get_attribute_value(attr)
                    for attr in node.attribute
                }
                k = a[0] if attrs.get("transA", 0) else a[1]
                by_op["Gemm"] += math.prod(out) * k
        elif node.op_type == "Conv":
            weight = shapes.get(node.input[1])
            out = shapes.get(node.output[0])
            if weight and out and len(weight) >= 3:
                by_op["Conv"] += math.prod(out) * math.prod(weight[1:])

    return sum(by_op.values()), by_op


def _latency_stats(
    samples_ms: list[float], batch_size: int
) -> dict[str, float]:
    samples_ms.sort()
    mean_ms = statistics.fmean(samples_ms)
    return {
        "mean_ms": mean_ms,
        "median_ms": statistics.median(samples_ms),
        "p90_ms": samples_ms[math.ceil(0.90 * len(samples_ms)) - 1],
        "p99_ms": samples_ms[math.ceil(0.99 * len(samples_ms)) - 1],
        "throughput_jets_per_s": batch_size * 1000.0 / mean_ms,
    }


def _session_options(single_thread: bool) -> object:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if single_thread:
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
    return options


def benchmark_cpu(
    path: Path,
    inputs: Sequence[torch.Tensor],
    warmup: int,
    iterations: int,
) -> tuple[np.ndarray, dict[str, object]]:
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(path),
        sess_options=_session_options(single_thread=True),
        providers=["CPUExecutionProvider"],
    )
    arrays = {
        ort_input.name: tensor.detach().cpu().numpy()
        for ort_input, tensor in zip(session.get_inputs(), inputs)
    }

    for _ in range(warmup):
        output = session.run(None, arrays)

    samples_ms = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        output = session.run(None, arrays)
        samples_ms.append((time.perf_counter_ns() - start) / 1e6)

    result = {
        "provider": "CPUExecutionProvider",
        "threads": {"intra_op": 1, "inter_op": 1},
        "latency": _latency_stats(
            samples_ms, arrays[session.get_inputs()[0].name].shape[0]
        ),
    }
    return output[0], result


def onnxruntime_gpu_version() -> str:
    try:
        return importlib.metadata.version("onnxruntime-gpu")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "This benchmark requires the 'onnxruntime-gpu' package. "
            "Uninstall 'onnxruntime' and install 'onnxruntime-gpu'."
        ) from exc


def require_onnxruntime_gpu(device_id: int) -> None:
    import onnxruntime as ort

    version = onnxruntime_gpu_version()
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(
            f"onnxruntime-gpu {version} is installed, but CUDAExecutionProvider "
            "is unavailable. Check the CUDA/cuDNN libraries and GPU visibility."
        )
    try:
        ort.OrtValue.ortvalue_from_numpy(
            np.zeros(1, dtype=np.float32), "cuda", device_id
        )
    except Exception as exc:
        raise RuntimeError(
            f"onnxruntime-gpu {version} cannot allocate on CUDA device "
            f"{device_id}. Check GPU visibility and CUDA driver compatibility."
        ) from exc


def benchmark_gpu(
    path: Path,
    inputs: Sequence[torch.Tensor],
    warmup: int,
    iterations: int,
    device_id: int,
) -> tuple[np.ndarray, dict[str, object]]:
    import onnxruntime as ort

    require_onnxruntime_gpu(device_id)
    providers = [
        ("CUDAExecutionProvider", {"device_id": device_id}),
        "CPUExecutionProvider",
    ]
    session = ort.InferenceSession(
        str(path),
        sess_options=_session_options(single_thread=True),
        providers=providers,
    )
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(
            "ONNX Runtime did not activate CUDAExecutionProvider."
    )

    io_binding = session.io_binding()
    input_values = []
    for ort_input, tensor in zip(session.get_inputs(), inputs):
        value = ort.OrtValue.ortvalue_from_numpy(
            tensor.detach().cpu().numpy(), "cuda", device_id
        )
        input_values.append(value)
        io_binding.bind_ortvalue_input(ort_input.name, value)
    for output in session.get_outputs():
        io_binding.bind_output(output.name, "cuda", device_id)

    io_binding.synchronize_inputs()
    for _ in range(warmup):
        session.run_with_iobinding(io_binding)
        io_binding.synchronize_outputs()

    samples_ms = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        session.run_with_iobinding(io_binding)
        io_binding.synchronize_outputs()
        samples_ms.append((time.perf_counter_ns() - start) / 1e6)

    output = io_binding.copy_outputs_to_cpu()[0]
    result = {
        "provider": "CUDAExecutionProvider",
        "device_id": device_id,
        "latency_scope": (
            "Inference with inputs and outputs resident on GPU; host/device "
            "transfer time is excluded."
        ),
        "latency": _latency_stats(samples_ms, inputs[0].shape[0]),
    }
    return output, result


def benchmark_model(
    args: argparse.Namespace,
    model_name: str,
    ort_gpu_version: str | None,
) -> dict[str, object]:
    torch.manual_seed(args.seed)
    model, inputs, input_names = make_model_and_inputs(args, model_name)
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint)

    output_path = args.output or Path(f"{model_name}_static_fp32.onnx")
    export_onnx(model, inputs, input_names, output_path, args.opset)

    with torch.inference_mode():
        torch_output = model(*inputs).detach().cpu().numpy()
    runtimes = {}
    max_abs_diff = {}
    if "cpu" in args.devices:
        cpu_output, runtimes["cpu"] = benchmark_cpu(
            output_path, inputs, args.warmup, args.iterations
        )
        np.testing.assert_allclose(
            cpu_output, torch_output, rtol=1e-2, atol=1e-2
        )
        max_abs_diff["cpu"] = float(
            np.max(np.abs(torch_output - cpu_output))
        )
    if "gpu" in args.devices:
        gpu_output, runtimes["gpu"] = benchmark_gpu(
            output_path,
            inputs,
            args.warmup,
            args.iterations,
            args.gpu_device_id,
        )
        np.testing.assert_allclose(
            gpu_output, torch_output, rtol=1e-2, atol=1e-2
        )
        max_abs_diff["gpu"] = float(
            np.max(np.abs(torch_output - gpu_output))
        )

    macs, macs_by_op = count_onnx_macs(output_path)
    median_ms = {
        device: runtime["latency"]["median_ms"]
        for device, runtime in runtimes.items()
    }
    return {
        "model": model_name,
        "summary": {
            "latency_median_ms": median_ms,
            "estimated_macs": macs,
        },
        "onnx_path": str(output_path),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "dtype": "fp32",
        "onnxruntime_package": {
            "name": (
                "onnxruntime-gpu" if ort_gpu_version is not None
                else "onnxruntime"
            ),
            "version": ort_gpu_version,
        },
        "static_shapes": {
            name: list(tensor.shape)
            for name, tensor in zip(input_names, inputs)
        },
        "warmup": args.warmup,
        "iterations": args.iterations,
        "runtimes": runtimes,
        "macs_matmul_gemm_conv": macs,
        "macs_by_op": macs_by_op,
        "estimated_flops_2x_macs": 2 * macs,
        "flop_note": (
            "2 FLOPs per MAC; excludes normalization, activations, softmax, "
            "masking, and other elementwise operations."
        ),
        "pytorch_onnx_max_abs_diff": max_abs_diff,
    }


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("--warmup must be non-negative and --iterations positive")
    if args.batch_size <= 0 or args.num_particles <= 0 or args.input_dim <= 0:
        raise ValueError("Static input dimensions must be positive")
    if args.model == "all" and args.output is not None:
        raise ValueError("--output cannot be used with model='all'")
    if args.model == "all" and args.checkpoint is not None:
        raise ValueError("--checkpoint cannot be used with model='all'")

    torch.set_num_threads(1)
    ort_gpu_version = (
        onnxruntime_gpu_version() if "gpu" in args.devices else None
    )
    if args.model == "all":
        model_names = ("singlepartau", "multipartau", "mixer")
    else:
        model_names = (
            args.model,
        )
    results = [
        benchmark_model(args, model_name, ort_gpu_version)
        for model_name in model_names
    ]
    print(json.dumps(results[0] if len(results) == 1 else results, indent=2))


if __name__ == "__main__":
    main()
