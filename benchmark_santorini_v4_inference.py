"""Benchmark frozen V4 checkpoints at realistic dense-inference batch sizes."""

import argparse
import hashlib
import json
import math
import os
import platform
import time

import numpy as np
import torch

from santorini.SantoriniGame import SantoriniGame
from santorini.V4Supervised import prepare_corpus
from santorini.pytorch.V4NNet import export_v4_model, load_v4_checkpoint


DEFAULT_CHECKPOINTS = (
    "ordinary13=temp/santorini_v4_p1b_screen_ordinary_global8/ordinary_13_global_blend.pth.tar",
    "candidate_a=temp/santorini_v4_p1b_screen/equivariant_13_stage_blend.pth.tar",
    "candidate_b=temp/santorini_v4_p1b_screen_bc/equivariant_b_13_stage_blend.pth.tar",
    "candidate_c=temp/santorini_v4_p1b_screen_c_targets/equivariant_c_13_global_blend.pth.tar",
    "candidate_d=temp/santorini_v4_p1b_screen_d/equivariant_d_13_stage_blend.pth.tar",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", action="append", dest="checkpoints", metavar="NAME=PATH",
        help="Checkpoint to benchmark; repeat for multiple models.",
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=(1, 8, 32, 64, 128, 192)
    )
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--examples-per-case", type=int, default=2_048)
    parser.add_argument("--minimum-iterations", type=int, default=20)
    parser.add_argument("--agreement-examples", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--output-dir", default="temp/santorini_v4_inference_benchmark")
    parser.add_argument("--json-out")
    parser.add_argument(
        "--engine-corpus", default="temp/santorini_v4_pilot_branch_010/corpus.npz"
    )
    parser.add_argument(
        "--run13-component", default="temp/santorini_v4_mixed_pilot/run13-component.npz"
    )
    parser.add_argument(
        "--selection-plan", default="temp/santorini_v4_mixed_pilot/selection-plan-3k.npz"
    )
    return parser.parse_args()


def _device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(name)


def _checkpoint_specs(values):
    specs = []
    for value in values or DEFAULT_CHECKPOINTS:
        if "=" not in value:
            raise ValueError("Checkpoint arguments must use NAME=PATH.")
        name, path = value.split("=", 1)
        if not name or not path or any(existing[0] == name for existing in specs):
            raise ValueError("Checkpoint names and paths must be nonempty and names unique.")
        if not os.path.isfile(path):
            if values:
                raise FileNotFoundError("Missing checkpoint: {}".format(path))
            continue
        specs.append((name, os.path.abspath(path)))
    if not specs:
        raise FileNotFoundError("No benchmark checkpoints were found.")
    return specs


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_exported(path, game, trace_input):
    model, config, checkpoint = load_v4_checkpoint(path, game)
    trace_input = trace_input[:, :int(config["planes"])]
    learned_parameters = sum(parameter.numel() for parameter in model.parameters())
    exported = export_v4_model(model, config)
    with torch.inference_mode():
        eager_output = exported(trace_input)
        scripted = torch.jit.freeze(torch.jit.trace(exported, trace_input))
        scripted_output = scripted(trace_input)
    export_validation = {}
    for name, expected, actual in zip(
        ("policy", "value"), eager_output, scripted_output
    ):
        difference = torch.abs(expected - actual)
        export_validation[name + "_max_absolute_difference"] = float(difference.max())
        export_validation[name + "_mean_absolute_difference"] = float(difference.mean())
        if not torch.allclose(expected, actual, atol=1e-5, rtol=1e-5):
            raise RuntimeError("Frozen export does not match eager inference.")
    return scripted, config, checkpoint, learned_parameters, export_validation


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _autocast(device, enabled):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=enabled,
    )


def _forward(model, inputs, device, fp16):
    with torch.inference_mode(), _autocast(device, fp16):
        return model(inputs)


def _measure_case(model, inputs, device, fp16, warmups, iterations):
    for _ in range(warmups):
        _forward(model, inputs, device, fp16)
    _sync(device)
    started = time.perf_counter()
    for _ in range(iterations):
        _forward(model, inputs, device, fp16)
    _sync(device)
    elapsed = time.perf_counter() - started
    examples = iterations * len(inputs)
    return {
        "batch_size": len(inputs),
        "iterations": iterations,
        "examples": examples,
        "elapsed_seconds": elapsed,
        "examples_per_second": examples / elapsed,
        "milliseconds_per_batch": 1_000.0 * elapsed / iterations,
    }


def _predictions(model, boards, batch_size, device, fp16):
    policies = []
    values = []
    for start in range(0, len(boards), batch_size):
        inputs = torch.from_numpy(
            np.ascontiguousarray(boards[start:start + batch_size])
        ).to(device)
        policy, value = _forward(model, inputs, device, fp16)
        policies.append(policy.float().cpu().numpy())
        values.append(value[:, 0].float().cpu().numpy())
    return np.concatenate(policies), np.concatenate(values)


def _agreement(model, boards, batch_size, device):
    fp32_policy, fp32_value = _predictions(model, boards, batch_size, device, False)
    fp16_policy, fp16_value = _predictions(model, boards, batch_size, device, True)
    fp32_probability = np.exp(fp32_policy)
    fp16_probability = np.exp(fp16_policy)
    midpoint = 0.5 * (fp32_probability + fp16_probability)
    epsilon = 1e-12
    js = 0.5 * np.sum(
        fp32_probability * np.log((fp32_probability + epsilon) / (midpoint + epsilon)),
        axis=1,
    ) + 0.5 * np.sum(
        fp16_probability * np.log((fp16_probability + epsilon) / (midpoint + epsilon)),
        axis=1,
    )
    return {
        "examples": len(boards),
        "policy_top1_agreement": float(np.mean(
            np.argmax(fp32_policy, axis=1) == np.argmax(fp16_policy, axis=1)
        )),
        "policy_mean_jensen_shannon": float(np.mean(js)),
        "policy_max_jensen_shannon": float(np.max(js)),
        "value_mean_absolute_difference": float(np.mean(np.abs(fp32_value - fp16_value))),
        "value_max_absolute_difference": float(np.max(np.abs(fp32_value - fp16_value))),
    }


def _atomic_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
    os.replace(temporary, path)


def main():
    args = parse_args()
    if any(batch < 1 for batch in args.batch_sizes):
        raise ValueError("Batch sizes must be positive.")
    if args.warmup_iterations < 0 or args.examples_per_case < 1 or args.minimum_iterations < 1:
        raise ValueError("Benchmark iteration counts are invalid.")
    device = _device(args.device)
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError("The target benchmark requires CUDA; use --allow-cpu for diagnostics.")
    checkpoints = _checkpoint_specs(args.checkpoints)
    corpus = prepare_corpus(
        args.engine_corpus,
        args.run13_component,
        args.selection_plan,
        expected_split=1,
    )
    max_examples = max(max(args.batch_sizes), args.agreement_examples, 1)
    if max_examples > len(corpus.encoded_boards):
        raise ValueError("The selection corpus is smaller than the requested benchmark sample.")
    benchmark_boards = corpus.encoded_boards[:max_examples]
    trace_input = torch.from_numpy(np.ascontiguousarray(benchmark_boards[:2]))
    os.makedirs(args.output_dir, exist_ok=True)
    game = SantoriniGame(5, sequential_placement=True)
    results = []
    for name, path in checkpoints:
        print("Exporting {}...".format(name), flush=True)
        model, config, checkpoint, learned_parameters, export_validation = _load_exported(
            path, game, trace_input
        )
        planes = int(config["planes"])
        scripted_path = os.path.join(args.output_dir, name + ".frozen.pt")
        torch.jit.save(model, scripted_path)
        # Frozen parameters are constants rather than normal module parameters;
        # reload with an explicit map location instead of relying on `.to()` to
        # migrate those constants.
        model = torch.jit.load(scripted_path, map_location=device).eval()
        cases = []
        precisions = ("fp32", "autocast_fp16") if device.type == "cuda" else ("fp32",)
        for precision in precisions:
            fp16 = precision == "autocast_fp16"
            for batch_size in args.batch_sizes:
                inputs = torch.from_numpy(
                    np.ascontiguousarray(benchmark_boards[:batch_size, :planes])
                ).to(device)
                iterations = max(
                    args.minimum_iterations,
                    int(math.ceil(args.examples_per_case / batch_size)),
                )
                metrics = _measure_case(
                    model, inputs, device, fp16, args.warmup_iterations, iterations
                )
                metrics["precision"] = precision
                cases.append(metrics)
                print(
                    "  {} batch {:>3}: {:>9.1f} examples/sec".format(
                        precision, batch_size, metrics["examples_per_second"]
                    ),
                    flush=True,
                )
        agreement = None
        if device.type == "cuda" and args.agreement_examples:
            agreement = _agreement(
                model,
                benchmark_boards[:args.agreement_examples, :planes],
                max(args.batch_sizes),
                device,
            )
        results.append({
            "name": name,
            "checkpoint": path,
            "checkpoint_sha256": _sha256(path),
            "checkpoint_schema_version": int(checkpoint.get("schema_version", 0)),
            "checkpoint_role": checkpoint.get("checkpoint_role", "legacy_final_epoch"),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "config": config,
            "learned_parameters": learned_parameters,
            "export_validation": export_validation,
            "frozen_torchscript": os.path.abspath(scripted_path),
            "frozen_torchscript_bytes": os.path.getsize(scripted_path),
            "cases": cases,
            "fp16_agreement": agreement,
        })
    payload = {
        "schema_version": 1,
        "type": "santorini_v4_frozen_inference_benchmark",
        "hardware": {
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "batch_sizes": list(args.batch_sizes),
        "warmup_iterations": args.warmup_iterations,
        "examples_per_case": args.examples_per_case,
        "minimum_iterations": args.minimum_iterations,
        "agreement_examples": args.agreement_examples,
        "selection_examples_available": len(corpus.encoded_boards),
        "final_test_touched": False,
        "results": results,
    }
    output_path = args.json_out or os.path.join(args.output_dir, "results.json")
    _atomic_json(output_path, payload)
    print("Results: {}".format(os.path.abspath(output_path)))
if __name__ == "__main__":
    main()
