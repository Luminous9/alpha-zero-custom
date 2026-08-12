"""Validate and optionally migrate the frozen P1c checkpoint into P2."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from MCTS import MCTS
from pit_santorini import search_args
from santorini.D4Canonical import canonicalize_board_policy
from santorini.SantoriniGame import SantoriniGame
from santorini.SantoriniOracle import anonymous_board_key
from santorini.V4SeamTelemetry import load_seam_telemetry_suite
from santorini.pytorch.NNet import args as nnet_args, build_nnet
from santorini.pytorch.V4NNet import V4InferenceWrapper


SCHEMA_VERSION = 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seam-suite", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--migrated-checkpoint")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--positions", type=int, default=4)
    parser.add_argument("--inference-positions", type=int, default=64)
    parser.add_argument("--simulations", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--policy-tolerance", type=float, default=5e-7)
    parser.add_argument("--value-tolerance", type=float, default=1e-7)
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(name)


def immediate_win_board():
    board = np.zeros((2, 5, 5), dtype=np.int8)
    board[0, 2, 2] = 1
    board[0, 0, 0] = 2
    board[0, 4, 4] = -1
    board[0, 4, 3] = -2
    board[1, 2, 2] = 2
    board[1, 2, 3] = 3
    return board


def transformed_boards(game, board):
    dummy = np.zeros(game.getActionSize(), dtype=np.float32)
    return [item[0] for item in game.getSymmetries(board, dummy)]


def run_search(game, nnet, board, simulations):
    args = search_args(
        simulations,
        search_mode="gumbel",
        gumbel_max_considered_actions=16,
        gumbel_scale=0.0,
        gumbel_placement_scale=0.0,
        search_symmetry_evaluation=False,
        root_symmetry_samples=1,
        placement_root_symmetry_samples=1,
        inference_deduplication=True,
        inference_cache_size=4096,
    )
    mcts = MCTS(game, nnet, args)
    action_policy = np.asarray(
        mcts.getActionProb(
            board,
            temp=0,
            num_simulations=simulations,
            add_root_noise=False,
        ),
        dtype=np.float64,
    )
    training_policy = np.asarray(
        mcts.getTrainingPolicyFromTree(board, temp=1), dtype=np.float64
    )
    tactical = mcts.prepareTacticalRoot(board)
    return action_policy, training_policy, None if tactical is None else tactical["kind"]


def projected_policy(game, board, policy):
    _, canonical_policy, key = canonicalize_board_policy(game, board, policy)
    return canonical_policy, key


def validate_search_position(game, nnet, board, simulations, tolerance, label):
    baseline_action, baseline_training, tactical = run_search(
        game, nnet, board, simulations
    )
    baseline_action_projected, key = projected_policy(game, board, baseline_action)
    baseline_training_projected, _ = projected_policy(
        game, board, baseline_training
    )
    anonymous = anonymous_board_key(board)
    stabilizer_size = sum(
        anonymous_board_key(item) == anonymous
        for item in transformed_boards(game, board)
    )
    maximum_action_difference = 0.0
    maximum_training_difference = 0.0
    raw_asymmetric_action_match = True
    raw_asymmetric_training_difference = 0.0

    for transform_index, transformed in enumerate(transformed_boards(game, board)):
        action_policy, training_policy, transformed_tactical = run_search(
            game, nnet, transformed, simulations
        )
        if transformed_tactical != tactical:
            raise AssertionError(
                "Tactical classification changed under transform for {}.".format(label)
            )
        action_projected, transformed_key = projected_policy(
            game, transformed, action_policy
        )
        training_projected, _ = projected_policy(
            game, transformed, training_policy
        )
        if transformed_key != key:
            raise AssertionError("Canonical search key changed under D4 transform.")
        maximum_action_difference = max(
            maximum_action_difference,
            float(np.max(np.abs(action_projected - baseline_action_projected))),
        )
        maximum_training_difference = max(
            maximum_training_difference,
            float(np.max(np.abs(training_projected - baseline_training_projected))),
        )

        if stabilizer_size == 1:
            rotations = transform_index // 2
            flip = bool(transform_index % 2)
            expected_action = game._transform_policy_array(
                baseline_action, rotations, flip
            )
            expected_training = game._transform_policy_array(
                baseline_training, rotations, flip
            )
            raw_asymmetric_action_match &= bool(
                np.allclose(action_policy, expected_action, atol=tolerance, rtol=0)
            )
            raw_asymmetric_training_difference = max(
                raw_asymmetric_training_difference,
                float(np.max(np.abs(training_policy - expected_training))),
            )

    passed = (
        maximum_action_difference <= tolerance
        and maximum_training_difference <= tolerance
        and (stabilizer_size > 1 or raw_asymmetric_action_match)
        and (stabilizer_size > 1 or raw_asymmetric_training_difference <= tolerance)
    )
    return {
        "label": label,
        "stabilizer_size": int(stabilizer_size),
        "tactical_kind": tactical,
        "projected_action_policy_max_abs_difference": maximum_action_difference,
        "projected_training_policy_max_abs_difference": maximum_training_difference,
        "raw_asymmetric_action_match": bool(raw_asymmetric_action_match),
        "raw_asymmetric_training_policy_max_abs_difference": (
            raw_asymmetric_training_difference
        ),
        "passed": bool(passed),
    }


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main():
    args = parse_args()
    if args.positions < 1 or args.inference_positions < 1 or args.simulations < 1:
        raise ValueError("Position and simulation counts must be positive.")
    device = resolve_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    game = SantoriniGame(5, sequential_placement=True)
    suite = load_seam_telemetry_suite(args.seam_suite)
    inference_boards = np.asarray(suite["boards"][:args.inference_positions])
    reference = V4InferenceWrapper(
        game,
        args.checkpoint,
        device=device,
        autocast_fp16=False,
        freeze_torchscript=True,
        canonicalize_d4=True,
    )
    nnet_args.cuda = device.type == "cuda"
    nnet_args.optimizer = "adamw"
    nnet_args.v4_freeze_torchscript = True
    nnet_args.v4_autocast_fp16 = False
    trainable = build_nnet(game, "v4")
    metadata = trainable.load_checkpoint(
        os.path.dirname(os.path.abspath(args.checkpoint)),
        os.path.basename(args.checkpoint),
        load_optimizer=False,
    )

    expected_policies, expected_values = reference.predict_batch(inference_boards)
    actual_policies, actual_values = trainable.predict_batch(inference_boards)
    policy_difference = float(np.max(np.abs(expected_policies - actual_policies)))
    value_difference = float(np.max(np.abs(expected_values - actual_values)))
    inference_passed = (
        policy_difference <= args.policy_tolerance
        and value_difference <= args.value_tolerance
    )

    search_boards = list(suite["boards"][:args.positions])
    search_labels = ["seam_{}".format(index) for index in range(len(search_boards))]
    search_boards.extend((game.getInitBoard(), immediate_win_board()))
    search_labels.extend(("empty_symmetric_placement", "immediate_win_tactical"))
    search_results = [
        validate_search_position(
            game,
            trainable,
            board,
            args.simulations,
            args.policy_tolerance,
            label,
        )
        for board, label in zip(search_boards, search_labels)
    ]
    search_passed = all(item["passed"] for item in search_results)

    result = {
        "schema_version": SCHEMA_VERSION,
        "type": "santorini_v4_p2_handoff_validation",
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "seam_suite": os.path.abspath(args.seam_suite),
        "seam_suite_sha256": file_sha256(args.seam_suite),
        "device": str(device),
        "cuda_device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "seed": int(args.seed),
        "simulations": int(args.simulations),
        "inference_positions": int(len(inference_boards)),
        "inference_policy_max_abs_difference": policy_difference,
        "inference_value_max_abs_difference": value_difference,
        "inference_passed": bool(inference_passed),
        "source_training_metadata": metadata,
        "search_results": search_results,
        "search_passed": bool(search_passed),
        "passed": bool(inference_passed and search_passed),
        "migrated_checkpoint": None,
        "migrated_checkpoint_sha256": None,
    }
    if not result["passed"]:
        atomic_json(args.json_out, result)
        raise AssertionError("V4 P2 handoff validation failed; see {}.".format(args.json_out))

    if args.migrated_checkpoint:
        migrated = os.path.abspath(args.migrated_checkpoint)
        trainable.save_checkpoint(
            os.path.dirname(migrated),
            os.path.basename(migrated),
            include_optimizer=True,
            metadata={
                "iteration": 0,
                "training_mode": "latest",
                "p2_handoff_source_sha256": result["checkpoint_sha256"],
                "p2_handoff_validation_schema": SCHEMA_VERSION,
                "p2_handoff_seed": int(args.seed),
            },
        )
        result["migrated_checkpoint"] = migrated
        result["migrated_checkpoint_sha256"] = file_sha256(migrated)

    atomic_json(args.json_out, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
