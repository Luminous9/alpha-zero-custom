"""Fine-tune a V3 Santorini checkpoint on oracle policy targets with rehearsal."""

import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch

from santorini.OracleResearch import stage_for_builds
from santorini.SantoriniGame import SantoriniGame
from santorini.pytorch.NNet import args as nnet_args
from santorini.pytorch.NNet import build_nnet


DEFAULT_CHECKPOINT = "./temp/santorini_v3_run13_gumbel/latest.pth.tar"
DEFAULT_TEACHER_REPLAY = "./temp/run13_oracle_teacher_5k.examples.npz"
DEFAULT_REHEARSAL_REPLAY = "./temp/santorini_v3_run13_gumbel/latest.examples.npz"
DEFAULT_OUTPUT_FOLDER = "./temp/santorini_v3_run13_oracle_distilled"
REHEARSAL_STAGES = ("placement", "early", "middle", "late")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a copy of a Santorini V3 checkpoint using oracle-blended "
            "policies plus phase-stratified source rehearsal."
        )
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--teacher-replay", default=DEFAULT_TEACHER_REPLAY)
    parser.add_argument("--rehearsal-replay", default=DEFAULT_REHEARSAL_REPLAY)
    parser.add_argument("--output-folder", default=DEFAULT_OUTPUT_FOLDER)
    parser.add_argument("--output-file", default="best.pth.tar")
    parser.add_argument("--metadata-file", default="finetune_metadata.json")
    parser.add_argument("--teacher-symmetry-multiplier", type=int, default=8)
    parser.add_argument("--max-teacher-positions", type=int)
    parser.add_argument("--rehearsal-positions", type=int, default=5_000)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--max-train-steps", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument(
        "--rehearsal-loss-tolerance",
        type=float,
        default=0.05,
        help="Maximum held-out rehearsal policy-loss increase for checkpoint selection.",
    )
    parser.add_argument(
        "--rehearsal-value-loss-tolerance",
        type=float,
        help=(
            "Optional maximum held-out rehearsal value-loss increase for checkpoint "
            "selection. Omit to preserve the legacy policy-only gate."
        ),
    )
    parser.add_argument(
        "--teacher-value-target",
        choices=("replay", "source-prediction"),
        default="replay",
        help=(
            "Use replay outcomes or preserve the source checkpoint's value prediction "
            "on teacher positions."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--trainable-scope",
        choices=("full", "policy-head"),
        default="full",
        help="Train the full network or only the final spatial policy projection.",
    )
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def compact_replay_info(path):
    with np.load(path, allow_pickle=False) as payload:
        return {
            "format_version": int(payload["format_version"][0]),
            "action_size": int(payload["action_size"][0]),
            "example_count": int(len(payload["boards"])),
            "history_lengths": payload["history_lengths"].astype(int).tolist(),
        }


def decode_compact_examples(path, indices):
    examples = []
    with np.load(path, allow_pickle=False) as payload:
        action_size = int(payload["action_size"][0])
        offsets = payload["policy_offsets"]
        sparse_indices = payload["policy_indices"]
        sparse_values = payload["policy_values"]
        for example_index in (int(index) for index in indices):
            start = int(offsets[example_index])
            end = int(offsets[example_index + 1])
            policy = np.zeros(action_size, dtype=np.float32)
            policy[sparse_indices[start:end].astype(np.int64)] = sparse_values[start:end]
            examples.append((
                payload["boards"][example_index].astype(int),
                policy,
                float(payload["values"][example_index]),
            ))
    return examples


def replace_values_with_source_predictions(nnet, examples, batch_size=256):
    """Keep teacher policies while anchoring values to the source checkpoint."""
    if int(batch_size) < 1:
        raise ValueError("Prediction batch size must be positive.")
    replaced = []
    for start in range(0, len(examples), int(batch_size)):
        batch = examples[start:start + int(batch_size)]
        _, values = nnet.predict_batch([example[0] for example in batch])
        replaced.extend(
            (board, policy, float(value))
            for (board, policy, _), value in zip(batch, values)
        )
    return replaced


def split_indices(indices, validation_fraction, rng):
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) < 2:
        raise ValueError("Need at least two positions for a train/validation split.")
    if not 0 < float(validation_fraction) < 1:
        raise ValueError("--validation-fraction must be between zero and one.")
    shuffled = rng.permutation(indices)
    validation_count = max(1, int(round(len(indices) * float(validation_fraction))))
    validation_count = min(validation_count, len(indices) - 1)
    return shuffled[validation_count:].tolist(), shuffled[:validation_count].tolist()


def teacher_base_indices(path, symmetry_multiplier=8, max_positions=None):
    info = compact_replay_info(path)
    symmetry_multiplier = int(symmetry_multiplier)
    if symmetry_multiplier < 1:
        raise ValueError("--teacher-symmetry-multiplier must be positive.")
    if info["example_count"] % symmetry_multiplier:
        raise ValueError(
            "Teacher replay count {} is not divisible by symmetry multiplier {}.".format(
                info["example_count"], symmetry_multiplier
            )
        )
    indices = list(range(0, info["example_count"], symmetry_multiplier))
    if max_positions is not None:
        if int(max_positions) < 2:
            raise ValueError("--max-teacher-positions must be at least two.")
        indices = indices[:int(max_positions)]
    return indices


def replay_stage(board):
    if int(np.count_nonzero(board[0])) < 4:
        return "placement"
    return stage_for_builds(int(np.sum(board[1])))


def select_rehearsal_indices(path, count, rng):
    count = int(count)
    if count < len(REHEARSAL_STAGES) * 2:
        raise ValueError("--rehearsal-positions must allow at least two examples per phase.")
    pools = {stage: [] for stage in REHEARSAL_STAGES}
    with np.load(path, allow_pickle=False) as payload:
        for index, board in enumerate(payload["boards"]):
            pools[replay_stage(board)].append(int(index))

    base, remainder = divmod(count, len(REHEARSAL_STAGES))
    selected = []
    for stage_index, stage in enumerate(REHEARSAL_STAGES):
        quota = base + (1 if stage_index < remainder else 0)
        if len(pools[stage]) < quota:
            raise ValueError(
                "Rehearsal replay has only {} {} examples; {} requested.".format(
                    len(pools[stage]), stage, quota
                )
            )
        selected.extend(
            int(index) for index in rng.choice(pools[stage], size=quota, replace=False)
        )
    rng.shuffle(selected)
    return selected


def stage_counts(examples):
    counts = {stage: 0 for stage in REHEARSAL_STAGES}
    for board, _, _ in examples:
        counts[replay_stage(board)] += 1
    return counts


def evaluation_metrics(nnet, examples):
    # Use the same phase-aware held-out evaluator as normal V3 training.
    return nnet._validation_metrics(examples)


def configure_trainable_scope(nnet, scope):
    if scope == "full":
        for parameter in nnet.nnet.parameters():
            parameter.requires_grad = True
    elif scope == "policy-head":
        for parameter in nnet.nnet.parameters():
            parameter.requires_grad = False
        for parameter in nnet.nnet.policy_conv.parameters():
            parameter.requires_grad = True
    else:
        raise ValueError("Unknown trainable scope: {}".format(scope))
    trainable = [name for name, parameter in nnet.nnet.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("Trainable scope selected no parameters.")
    return trainable


def write_json_atomic(path, payload):
    path = os.path.abspath(path)
    temporary_path = path + ".tmp"
    with open(temporary_path, "w") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
    os.replace(temporary_path, path)


def main():
    args = parse_args()
    for path in (args.checkpoint, args.teacher_replay, args.rehearsal_replay):
        if not os.path.isfile(path):
            raise FileNotFoundError("Required input not found: {}".format(path))
    if args.eval_interval < 1 or args.max_train_steps < 1 or args.batch_size < 1:
        raise ValueError("Eval interval, max train steps, and batch size must be positive.")
    if args.rehearsal_loss_tolerance < 0:
        raise ValueError("--rehearsal-loss-tolerance must be non-negative.")
    if (
        args.rehearsal_value_loss_tolerance is not None
        and args.rehearsal_value_loss_tolerance < 0
    ):
        raise ValueError("--rehearsal-value-loss-tolerance must be non-negative.")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Learning rate must be positive and weight decay non-negative.")

    os.makedirs(args.output_folder, exist_ok=True)
    output_checkpoint = os.path.join(args.output_folder, args.output_file)
    metadata_path = os.path.join(args.output_folder, args.metadata_file)
    if not args.overwrite:
        existing = [path for path in (output_checkpoint, metadata_path) if os.path.exists(path)]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing output: {}. Use --overwrite or choose "
                "another --output-folder.".format(", ".join(existing))
            )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    use_cuda = bool(torch.cuda.is_available() and not args.cpu)
    nnet_args.cuda = use_cuda
    nnet_args.optimizer = "adamw"
    nnet_args.lr = float(args.learning_rate)
    nnet_args.lr_schedule = []
    nnet_args.weight_decay = float(args.weight_decay)
    nnet_args.epochs = 1
    nnet_args.batch_size = int(args.batch_size)
    nnet_args.max_train_steps = int(args.max_train_steps)
    nnet_args.replay_reuse = None
    nnet_args.on_the_fly_symmetry = True
    nnet_args.symmetry_consistency_fraction = 0.0
    nnet_args.symmetry_consistency_policy_weight = 0.0
    nnet_args.symmetry_consistency_value_weight = 0.0
    nnet_args.freeze_batch_norm = True
    nnet_args.quiet = False

    game = SantoriniGame(5, sequential_placement=True)
    nnet = build_nnet(game, "v3")
    checkpoint_folder, checkpoint_file = os.path.split(os.path.abspath(args.checkpoint))
    nnet.load_checkpoint(checkpoint_folder, checkpoint_file, load_optimizer=False)
    trainable_parameters = configure_trainable_scope(nnet, args.trainable_scope)

    teacher_indices = teacher_base_indices(
        args.teacher_replay, args.teacher_symmetry_multiplier, args.max_teacher_positions
    )
    teacher_train_indices, teacher_validation_indices = split_indices(
        teacher_indices, args.validation_fraction, rng
    )
    rehearsal_indices = select_rehearsal_indices(
        args.rehearsal_replay, args.rehearsal_positions, rng
    )
    rehearsal_train_indices, rehearsal_validation_indices = split_indices(
        rehearsal_indices, args.validation_fraction, rng
    )

    print("Loading selected teacher and rehearsal examples...")
    teacher_train = decode_compact_examples(args.teacher_replay, teacher_train_indices)
    teacher_validation = decode_compact_examples(args.teacher_replay, teacher_validation_indices)
    if args.teacher_value_target == "source-prediction":
        print("Replacing teacher outcomes with source-checkpoint value predictions...")
        teacher_train = replace_values_with_source_predictions(
            nnet, teacher_train, batch_size=args.batch_size
        )
        teacher_validation = replace_values_with_source_predictions(
            nnet, teacher_validation, batch_size=args.batch_size
        )
    rehearsal_train = decode_compact_examples(args.rehearsal_replay, rehearsal_train_indices)
    rehearsal_validation = decode_compact_examples(
        args.rehearsal_replay, rehearsal_validation_indices
    )
    training_examples = teacher_train + rehearsal_train

    print("Measuring source checkpoint on held-out targets...")
    baseline_teacher = evaluation_metrics(nnet, teacher_validation)
    baseline_rehearsal = evaluation_metrics(nnet, rehearsal_validation)
    baseline_teacher_kl = float(baseline_teacher["standard_validation_policy_kl"])
    baseline_rehearsal_loss = float(baseline_rehearsal["validation_policy_loss"])
    baseline_rehearsal_value_loss = float(baseline_rehearsal["validation_value_loss"])
    best_step = 0
    best_teacher_kl = baseline_teacher_kl
    best_rehearsal_loss = baseline_rehearsal_loss
    best_rehearsal_value_loss = baseline_rehearsal_value_loss
    best_state = copy.deepcopy(nnet.nnet.state_dict())
    selection_history = [{
        "step": 0,
        "eligible": True,
        "selected": True,
        "teacher_validation": baseline_teacher,
        "rehearsal_validation": baseline_rehearsal,
    }]
    started = time.perf_counter()
    completed_steps = 0
    training_metrics = None
    while completed_steps < int(args.max_train_steps):
        segment_steps = min(int(args.eval_interval), int(args.max_train_steps) - completed_steps)
        nnet_args.max_train_steps = segment_steps
        training_metrics = nnet.train(training_examples)
        completed_steps += int(training_metrics["training_steps"])
        teacher_metrics = evaluation_metrics(nnet, teacher_validation)
        rehearsal_metrics = evaluation_metrics(nnet, rehearsal_validation)
        teacher_kl = float(teacher_metrics["standard_validation_policy_kl"])
        rehearsal_loss = float(rehearsal_metrics["validation_policy_loss"])
        rehearsal_value_loss = float(rehearsal_metrics["validation_value_loss"])
        policy_eligible = rehearsal_loss <= (
            baseline_rehearsal_loss + float(args.rehearsal_loss_tolerance)
        )
        value_eligible = (
            args.rehearsal_value_loss_tolerance is None
            or rehearsal_value_loss <= (
                baseline_rehearsal_value_loss
                + float(args.rehearsal_value_loss_tolerance)
            )
        )
        eligible = bool(policy_eligible and value_eligible)
        selected = bool(eligible and teacher_kl < best_teacher_kl)
        if selected:
            best_step = completed_steps
            best_teacher_kl = teacher_kl
            best_rehearsal_loss = rehearsal_loss
            best_rehearsal_value_loss = rehearsal_value_loss
            best_state = copy.deepcopy(nnet.nnet.state_dict())
        selection_history.append({
            "step": completed_steps,
            "eligible": bool(eligible),
            "selected": selected,
            "teacher_validation": teacher_metrics,
            "rehearsal_validation": rehearsal_metrics,
            "training_metrics": training_metrics,
        })
        print(
            "Step {}: teacher KL {:.4f}; rehearsal policy/value loss {:.4f}/{:.4f}; "
            "eligible={}; best={}.".format(
                completed_steps, teacher_kl, rehearsal_loss, rehearsal_value_loss,
                eligible, best_step
            )
        )
    training_seconds = time.perf_counter() - started
    nnet.nnet.load_state_dict(best_state)
    selected_teacher = evaluation_metrics(nnet, teacher_validation)
    selected_rehearsal = evaluation_metrics(nnet, rehearsal_validation)

    metadata = {
        "source_checkpoint": os.path.abspath(args.checkpoint),
        "teacher_replay": os.path.abspath(args.teacher_replay),
        "rehearsal_replay": os.path.abspath(args.rehearsal_replay),
        "output_checkpoint": os.path.abspath(output_checkpoint),
        "architecture": "v3",
        "seed": int(args.seed),
        "cuda": use_cuda,
        "optimizer": "adamw",
        "trainable_scope": args.trainable_scope,
        "trainable_parameters": trainable_parameters,
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "eval_interval": int(args.eval_interval),
        "rehearsal_loss_tolerance": float(args.rehearsal_loss_tolerance),
        "rehearsal_value_loss_tolerance": (
            None if args.rehearsal_value_loss_tolerance is None
            else float(args.rehearsal_value_loss_tolerance)
        ),
        "teacher_value_target": args.teacher_value_target,
        "max_train_steps": int(args.max_train_steps),
        "batch_size": int(args.batch_size),
        "on_the_fly_symmetry": True,
        "freeze_batch_norm": True,
        "teacher_symmetry_multiplier": int(args.teacher_symmetry_multiplier),
        "teacher_base_positions": len(teacher_indices),
        "teacher_train_positions": len(teacher_train),
        "teacher_validation_positions": len(teacher_validation),
        "rehearsal_positions": len(rehearsal_indices),
        "rehearsal_train_positions": len(rehearsal_train),
        "rehearsal_validation_positions": len(rehearsal_validation),
        "rehearsal_train_stage_counts": stage_counts(rehearsal_train),
        "rehearsal_validation_stage_counts": stage_counts(rehearsal_validation),
        "training_seconds": float(training_seconds),
        "baseline_teacher_validation": baseline_teacher,
        "baseline_rehearsal_validation": baseline_rehearsal,
        "best_step": int(best_step),
        "best_teacher_policy_kl": float(best_teacher_kl),
        "best_rehearsal_policy_loss": float(best_rehearsal_loss),
        "best_rehearsal_value_loss": float(best_rehearsal_value_loss),
        "selection_history": selection_history,
        "selected_teacher_validation": selected_teacher,
        "selected_rehearsal_validation": selected_rehearsal,
        "generated_at_unix": time.time(),
    }
    nnet.save_checkpoint(
        args.output_folder,
        args.output_file,
        include_optimizer=False,
        metadata={
            "distilled_from": os.path.abspath(args.checkpoint),
            "oracle_teacher_replay": os.path.abspath(args.teacher_replay),
            "distillation_steps": int(best_step),
            "distillation_attempted_steps": int(completed_steps),
        },
    )
    write_json_atomic(metadata_path, metadata)
    print(json.dumps({
        "training_seconds": metadata["training_seconds"],
        "attempted_training_steps": completed_steps,
        "selected_training_step": best_step,
        "baseline_teacher_policy_kl": baseline_teacher.get("standard_validation_policy_kl"),
        "selected_teacher_policy_kl": selected_teacher.get("standard_validation_policy_kl"),
        "baseline_rehearsal_policy_loss": baseline_rehearsal.get("validation_policy_loss"),
        "selected_rehearsal_policy_loss": selected_rehearsal.get("validation_policy_loss"),
        "baseline_rehearsal_value_loss": baseline_rehearsal.get("validation_value_loss"),
        "selected_rehearsal_value_loss": selected_rehearsal.get("validation_value_loss"),
    }, indent=2, sort_keys=True))
    print("Distilled checkpoint: {}".format(os.path.abspath(output_checkpoint)))
    print("Metadata: {}".format(os.path.abspath(metadata_path)))


if __name__ == "__main__":
    main()
