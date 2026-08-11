"""Compare matched V4 winner-only and global-blend checkpoints on selection data."""

import argparse
import hashlib
import json
import os

import numpy as np
import torch

from santorini.V4BootstrapCorpus import SOURCE_NAMES, STAGE_NAMES
from santorini.V4Supervised import (
    DEFAULT_ALPHA_BOOT,
    DEFAULT_STAGE_RELIABILITY,
    GLOBAL_SCORE_TEMPERATURE,
    StreamingPreparedV4Corpus,
)
from santorini.SantoriniGame import SantoriniGame
from santorini.pytorch.V4NNet import load_v4_checkpoint


PRIMARY_MARGIN = 0.01
POLICY_WEIGHT = 0.25


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--winner-checkpoint", required=True)
    parser.add_argument("--blend-checkpoint", required=True)
    parser.add_argument("--engine-corpus", required=True)
    parser.add_argument("--run13-component", required=True)
    parser.add_argument("--selection-plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--policy-epsilon", type=float, default=0.05)
    parser.add_argument("--alpha-boot", type=float, default=DEFAULT_ALPHA_BOOT)
    parser.add_argument("--score-temperature", type=float, default=GLOBAL_SCORE_TEMPERATURE)
    parser.add_argument(
        "--stage-reliability", type=float, nargs=3,
        default=DEFAULT_STAGE_RELIABILITY,
        metavar=("EARLY", "MIDDLE", "LATE"),
    )
    parser.add_argument("--noninferiority-margin", type=float, default=PRIMARY_MARGIN)
    return parser.parse_args()


def _device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(name)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_matched(winner_config, blend_config, winner_payload, blend_payload):
    if winner_config.get("target") != "winner":
        raise ValueError("The winner checkpoint does not declare target=winner.")
    if blend_config.get("target") != "global_blend":
        raise ValueError("The blend checkpoint does not declare target=global_blend.")
    config_fields = ("architecture", "planes", "channels", "residual_blocks")
    for field in config_fields:
        if winner_config.get(field) != blend_config.get(field):
            raise ValueError("Checkpoint architecture mismatch in {}.".format(field))
    if winner_payload.get("seed") != blend_payload.get("seed"):
        raise ValueError("Checkpoint initialization seeds do not match.")
    contract_fields = (
        "epochs", "batch_size", "learning_rate", "weight_decay",
        "policy_weight", "policy_epsilon", "alpha_boot", "score_temperature",
        "stage_reliability", "train_plan", "selection_plan",
    )
    winner_contract = winner_payload.get("training_contract", {})
    blend_contract = blend_payload.get("training_contract", {})
    for field in contract_fields:
        winner_value = winner_contract.get(field)
        blend_value = blend_contract.get(field)
        if field in ("train_plan", "selection_plan"):
            winner_value = os.path.basename(str(winner_value))
            blend_value = os.path.basename(str(blend_value))
        if winner_value != blend_value:
            raise ValueError("Checkpoint training-contract mismatch in {}.".format(field))


def _evaluate(model, corpus, planes, batch_size, device):
    model = model.to(device).eval()
    metrics = {
        "policy_ce": [],
        "policy_top1": [],
        "winner_squared_error": [],
        "global_blend_squared_error": [],
    }
    with torch.inference_mode():
        for start in range(0, len(corpus), batch_size):
            indices = np.arange(start, min(start + batch_size, len(corpus)))
            batch = corpus.batch(indices, planes)
            inputs = torch.from_numpy(
                np.ascontiguousarray(batch.encoded_boards)
            ).to(device)
            log_policy, value = model(inputs)
            log_policy = log_policy.float().cpu().numpy()
            value = value[:, 0].float().cpu().numpy()
            metrics["policy_ce"].append(
                -np.sum(batch.policies * log_policy, axis=1)
            )
            metrics["policy_top1"].append(
                np.argmax(log_policy, axis=1) == np.argmax(batch.policies, axis=1)
            )
            metrics["winner_squared_error"].append(
                (value - batch.winner_values) ** 2
            )
            metrics["global_blend_squared_error"].append(
                (value - batch.global_blended_values) ** 2
            )
    metrics = {name: np.concatenate(parts) for name, parts in metrics.items()}
    metrics["handoff_objective"] = (
        POLICY_WEIGHT * metrics["policy_ce"] + metrics["winner_squared_error"]
    )
    metrics["teacher_objective"] = (
        POLICY_WEIGHT * metrics["policy_ce"]
        + metrics["global_blend_squared_error"]
    )
    return metrics


def _means(metrics, mask=None):
    if mask is None:
        mask = slice(None)
    return {
        name: float(np.mean(values[mask]))
        for name, values in metrics.items()
    }


def _paired_interval(values, samples, rng):
    values = np.asarray(values, dtype=np.float64)
    means = []
    remaining = int(samples)
    while remaining:
        count = min(500, remaining)
        indices = rng.randint(len(values), size=(count, len(values)))
        means.append(values[indices].mean(axis=1))
        remaining -= count
    return list(map(float, np.quantile(np.concatenate(means), (0.025, 0.975))))


def _write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
    os.replace(temporary, path)


def main():
    args = parse_args()
    if args.batch_size < 1 or args.bootstrap_samples < 1:
        raise ValueError("Batch size and bootstrap samples must be positive.")
    if args.noninferiority_margin < 0:
        raise ValueError("The noninferiority margin cannot be negative.")
    device = _device(args.device)
    game = SantoriniGame(5, sequential_placement=True)
    winner_model, winner_config, winner_payload = load_v4_checkpoint(
        args.winner_checkpoint, game
    )
    blend_model, blend_config, blend_payload = load_v4_checkpoint(
        args.blend_checkpoint, game
    )
    _assert_matched(
        winner_config, blend_config, winner_payload, blend_payload
    )
    corpus = StreamingPreparedV4Corpus(
        engine_path=args.engine_corpus,
        run13_path=args.run13_component,
        plan_path=args.selection_plan,
        expected_split=1,
        policy_epsilon=args.policy_epsilon,
        alpha_boot=args.alpha_boot,
        stage_reliability=args.stage_reliability,
        temperature=args.score_temperature,
    )
    winner_metrics = _evaluate(
        winner_model, corpus, winner_config["planes"], args.batch_size, device
    )
    blend_metrics = _evaluate(
        blend_model, corpus, blend_config["planes"], args.batch_size, device
    )
    differences = {
        name: winner_metrics[name].astype(np.float64)
        - blend_metrics[name].astype(np.float64)
        for name in (
            "policy_ce", "winner_squared_error",
            "global_blend_squared_error", "handoff_objective", "teacher_objective",
        )
    }
    rng = np.random.RandomState(args.seed)
    paired = {
        name: {
            "winner_minus_global_blend_mean": float(np.mean(values)),
            "winner_minus_global_blend_95_interval": _paired_interval(
                values, args.bootstrap_samples, rng
            ),
        }
        for name, values in differences.items()
    }
    primary_upper = paired["handoff_objective"][
        "winner_minus_global_blend_95_interval"
    ][1]
    by_stage = {}
    by_source = {}
    for destination, ids, names in (
        (by_stage, corpus.stage_ids, STAGE_NAMES),
        (by_source, corpus.source_ids, SOURCE_NAMES),
    ):
        for index, name in enumerate(names):
            mask = ids == index
            destination[name] = {
                "examples": int(np.sum(mask)),
                "winner": _means(winner_metrics, mask),
                "global_blend": _means(blend_metrics, mask),
            }
    result = {
        "schema_version": 1,
        "type": "santorini_v4_1m_value_target_comparison",
        "device": str(device),
        "torch_version": torch.__version__,
        "selection_examples": len(corpus),
        "checkpoints": {
            "winner": {
                "path": os.path.abspath(args.winner_checkpoint),
                "sha256": _sha256(args.winner_checkpoint),
                "epoch": winner_payload.get("epoch"),
                "role": winner_payload.get("checkpoint_role"),
                "config": winner_config,
            },
            "global_blend": {
                "path": os.path.abspath(args.blend_checkpoint),
                "sha256": _sha256(args.blend_checkpoint),
                "epoch": blend_payload.get("epoch"),
                "role": blend_payload.get("checkpoint_role"),
                "config": blend_config,
            },
        },
        "decision_contract": {
            "primary_metric": "0.25 * policy_ce + winner_squared_error",
            "paired_difference": "winner_only_minus_global_blend",
            "noninferiority_margin": args.noninferiority_margin,
            "supervised_noninferiority_rule": (
                "upper endpoint of paired position-bootstrap 95% interval <= margin"
            ),
            "arena_veto_rule": (
                "global blend combined standard/full seed-cluster 95% score "
                "interval lies entirely above 0.5"
            ),
            "preference_if_both_pass": "winner_only",
            "arena_games": 80,
            "arena_seed": 20260814,
            "optional_stopping": False,
            "final_test_touched": False,
        },
        "winner": _means(winner_metrics),
        "global_blend": _means(blend_metrics),
        "paired": paired,
        "supervised_noninferior": bool(
            primary_upper <= args.noninferiority_margin
        ),
        "by_stage": by_stage,
        "by_source": by_source,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "final_test_touched": False,
    }
    corpus.close()
    _write_json(args.output, result)
    print(json.dumps({
        "winner": result["winner"],
        "global_blend": result["global_blend"],
        "paired": result["paired"],
        "supervised_noninferior": result["supervised_noninferior"],
    }, indent=2, sort_keys=True))
    print("Results: {}".format(os.path.abspath(args.output)))


if __name__ == "__main__":
    main()
