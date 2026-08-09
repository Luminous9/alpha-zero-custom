"""Run matched P1b supervised screens on the mixed V4 pilot corpus."""

import argparse
import json
import os
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from santorini.SantoriniGame import SantoriniGame
from santorini.V4BootstrapCorpus import SOURCE_NAMES, STAGE_NAMES
from santorini.V4Supervised import (
    DEFAULT_ALPHA_BOOT,
    DEFAULT_STAGE_RELIABILITY,
    GLOBAL_SCORE_TEMPERATURE,
    apply_d4_augmentation,
    prepare_corpus,
)
from santorini.pytorch.SantoriniNNet import SantoriniNNet
from santorini.pytorch.V4Prototype import D4RegularNetwork


SCREEN_CONFIGS = (
    {"name": "ordinary_6_stage_blend", "architecture": "ordinary", "planes": 6, "target": "stage_blend"},
    {"name": "ordinary_13_stage_blend", "architecture": "ordinary", "planes": 13, "target": "stage_blend"},
    {"name": "ordinary_13_global_blend", "architecture": "ordinary", "planes": 13, "target": "global_blend"},
    {"name": "equivariant_13_stage_blend", "architecture": "equivariant", "planes": 13, "target": "stage_blend", "candidate": "A", "effective_channels": 96, "residual_blocks": 8},
    {"name": "equivariant_13_global_blend", "architecture": "equivariant", "planes": 13, "target": "global_blend", "candidate": "A", "effective_channels": 96, "residual_blocks": 8},
    {"name": "equivariant_13_winner", "architecture": "equivariant", "planes": 13, "target": "winner", "candidate": "A", "effective_channels": 96, "residual_blocks": 8},
    {"name": "equivariant_b_13_stage_blend", "architecture": "equivariant", "planes": 13, "target": "stage_blend", "candidate": "B", "effective_channels": 128, "residual_blocks": 10},
    {"name": "equivariant_c_13_stage_blend", "architecture": "equivariant", "planes": 13, "target": "stage_blend", "candidate": "C", "effective_channels": 192, "residual_blocks": 6},
    {"name": "equivariant_c_13_global_blend", "architecture": "equivariant", "planes": 13, "target": "global_blend", "candidate": "C", "effective_channels": 192, "residual_blocks": 6},
    {"name": "equivariant_c_13_winner", "architecture": "equivariant", "planes": 13, "target": "winner", "candidate": "C", "effective_channels": 192, "residual_blocks": 6},
    {"name": "equivariant_d_13_stage_blend", "architecture": "equivariant", "planes": 13, "target": "stage_blend", "candidate": "D", "effective_channels": 160, "residual_blocks": 12},
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-corpus", default="temp/santorini_v4_pilot_branch_010/corpus.npz")
    parser.add_argument("--run13-component", default="temp/santorini_v4_mixed_pilot/run13-component.npz")
    parser.add_argument("--train-plan", default="temp/santorini_v4_mixed_pilot/train-plan-10k.npz")
    parser.add_argument("--selection-plan", default="temp/santorini_v4_mixed_pilot/selection-plan-3k.npz")
    parser.add_argument("--output-dir", default="temp/santorini_v4_p1b_screen")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--policy-weight", type=float, default=0.25)
    parser.add_argument("--policy-epsilon", type=float, default=0.05)
    parser.add_argument("--alpha-boot", type=float, default=DEFAULT_ALPHA_BOOT)
    parser.add_argument("--score-temperature", type=float, default=GLOBAL_SCORE_TEMPERATURE)
    parser.add_argument(
        "--stage-reliability", type=float, nargs=3,
        default=DEFAULT_STAGE_RELIABILITY,
        metavar=("EARLY", "MIDDLE", "LATE"),
    )
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--configs", nargs="+", choices=[config["name"] for config in SCREEN_CONFIGS])
    return parser.parse_args()


def _device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(name)


def _ordinary_model(game, planes):
    args = SimpleNamespace(
        input_channels=int(planes),
        num_channels=96,
        num_residual_blocks=8,
        policy_channels=65,
        value_hidden_size=128,
        dropout=0.0,
    )
    return SantoriniNNet(game, args)


def _model(game, config):
    if config["architecture"] == "ordinary":
        return _ordinary_model(game, config["planes"])
    return D4RegularNetwork(
        game,
        input_channels=config["planes"],
        effective_channels=config.get("effective_channels", 96),
        residual_blocks=config.get("residual_blocks", 8),
        value_hidden_size=128,
        dropout=0.0,
    )


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _evaluate(model, corpus, planes, target_kind, batch_size, device):
    model.eval()
    predictions = []
    policies = []
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(corpus.encoded_boards), batch_size):
            end = min(len(corpus.encoded_boards), start + batch_size)
            boards = torch.from_numpy(
                np.ascontiguousarray(corpus.encoded_boards[start:end, :planes])
            ).to(device)
            out_policy, out_value = model(boards)
            policies.append(out_policy.cpu().numpy())
            predictions.append(out_value[:, 0].cpu().numpy())
    _sync(device)
    elapsed = time.perf_counter() - started
    log_policy = np.concatenate(policies)
    prediction = np.concatenate(predictions)
    target_policy = corpus.policies
    selected_target = corpus.value_targets(target_kind)
    result = {
        "policy_loss": float(-np.mean(np.sum(target_policy * log_policy, axis=1))),
        "policy_top1_accuracy": float(np.mean(np.argmax(log_policy, axis=1) == np.argmax(target_policy, axis=1))),
        "selected_value_mse": float(np.mean((prediction - selected_target) ** 2)),
        "winner_value_mse": float(np.mean((prediction - corpus.winner_values) ** 2)),
        "score_value_mse": float(np.mean((prediction - corpus.score_values) ** 2)),
        "stage_blend_value_mse": float(np.mean((prediction - corpus.stage_blended_values) ** 2)),
        "global_blend_value_mse": float(np.mean((prediction - corpus.global_blended_values) ** 2)),
        "winner_sign_accuracy": float(np.mean(np.sign(prediction) == np.sign(corpus.winner_values))),
        "examples_per_second": float(len(prediction) / elapsed),
        "elapsed_seconds": float(elapsed),
        "by_stage": {},
        "by_source": {},
    }
    for label, ids, names in (
        ("by_stage", corpus.stage_ids, STAGE_NAMES),
        ("by_source", corpus.source_ids, SOURCE_NAMES),
    ):
        for index, name in enumerate(names):
            mask = ids == index
            result[label][name] = {
                "examples": int(np.sum(mask)),
                "policy_loss": float(-np.mean(np.sum(target_policy[mask] * log_policy[mask], axis=1))),
                "winner_value_mse": float(np.mean((prediction[mask] - corpus.winner_values[mask]) ** 2)),
                "stage_blend_value_mse": float(np.mean((prediction[mask] - corpus.stage_blended_values[mask]) ** 2)),
            }
    return result


def _train_one(config, train, selection, args, device, game):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = _model(game, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    targets = train.value_targets(config["target"])
    history = []
    selection_history = []
    best_objective = float("inf")
    best_epoch = None
    best_state = None
    optimization_seconds = 0.0
    train_started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        rng = np.random.RandomState(args.seed + epoch)
        order = rng.permutation(len(train.encoded_boards))
        sums = {"policy": 0.0, "value": 0.0, "total": 0.0, "examples": 0}
        optimization_started = time.perf_counter()
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            boards = train.encoded_boards[indices, :config["planes"]]
            policies = train.policies[indices]
            if config["architecture"] == "ordinary":
                symmetry_ids = rng.randint(0, 8, size=len(indices))
                boards, policies = apply_d4_augmentation(
                    boards, policies, symmetry_ids, game
                )
            board_tensor = torch.from_numpy(np.ascontiguousarray(boards)).to(device)
            policy_tensor = torch.from_numpy(np.ascontiguousarray(policies)).to(device)
            value_tensor = torch.from_numpy(targets[indices, None]).to(device)
            optimizer.zero_grad(set_to_none=True)
            output_policy, output_value = model(board_tensor)
            policy_loss = -torch.mean(torch.sum(policy_tensor * output_policy, dim=1))
            value_loss = F.mse_loss(output_value, value_tensor)
            total_loss = args.policy_weight * policy_loss + value_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            count = len(indices)
            sums["policy"] += float(policy_loss.detach()) * count
            sums["value"] += float(value_loss.detach()) * count
            sums["total"] += float(total_loss.detach()) * count
            sums["examples"] += count
        _sync(device)
        optimization_seconds += time.perf_counter() - optimization_started
        history.append({
            "epoch": epoch + 1,
            "policy_loss": sums["policy"] / sums["examples"],
            "value_loss": sums["value"] / sums["examples"],
            "total_loss": sums["total"] / sums["examples"],
        })
        selection_metrics = _evaluate(
            model, selection, config["planes"], config["target"],
            args.batch_size, device,
        )
        selection_objective = (
            args.policy_weight * selection_metrics["policy_loss"]
            + selection_metrics["selected_value_mse"]
        )
        selection_history.append({
            "epoch": epoch + 1,
            "selection_objective": selection_objective,
            "metrics": selection_metrics,
        })
        if selection_objective < best_objective:
            best_objective = selection_objective
            best_epoch = epoch + 1
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        print(
            "  epoch {}: train={:.4f}, selection={:.4f} "
            "(policy={:.4f}, value={:.4f}){}".format(
                epoch + 1,
                history[-1]["total_loss"],
                selection_objective,
                selection_metrics["policy_loss"],
                selection_metrics["selected_value_mse"],
                " best" if best_epoch == epoch + 1 else "",
            ),
            flush=True,
        )
    _sync(device)
    total_seconds = time.perf_counter() - train_started
    final_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    return best_state, final_state, {
        "config": config,
        "train_history": history,
        "selection_history": selection_history,
        "best_epoch": best_epoch,
        "selection_objective_definition": "policy_weight * policy_loss + selected_value_mse",
        "best_selection_objective": best_objective,
        "optimization_seconds": optimization_seconds,
        "total_train_and_selection_seconds": total_seconds,
        "train_examples_per_second": args.epochs * len(train.encoded_boards) / optimization_seconds,
        "learned_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "selection": selection_history[best_epoch - 1]["metrics"],
        "final_selection": selection_history[-1]["metrics"],
    }


def _write_json(path, payload):
    temporary = path + ".tmp"
    with open(temporary, "w") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
    os.replace(temporary, path)


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.policy_weight <= 0:
        raise ValueError("Epochs, batch size, and policy weight must be positive.")
    if len(args.stage_reliability) != 3 or any(
        value < 0 or value > 1 for value in args.stage_reliability
    ):
        raise ValueError("Stage reliability values must be in [0, 1].")
    device = _device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    prepare_started = time.perf_counter()
    prepare_kwargs = dict(
        engine_path=args.engine_corpus,
        run13_path=args.run13_component,
        policy_epsilon=args.policy_epsilon,
        alpha_boot=args.alpha_boot,
        stage_reliability=args.stage_reliability,
        temperature=args.score_temperature,
    )
    train = prepare_corpus(
        plan_path=args.train_plan, expected_split=0, **prepare_kwargs
    )
    selection = prepare_corpus(
        plan_path=args.selection_plan, expected_split=1, **prepare_kwargs
    )
    preparation_seconds = time.perf_counter() - prepare_started
    selected_names = set(args.configs or [config["name"] for config in SCREEN_CONFIGS])
    configs = [config for config in SCREEN_CONFIGS if config["name"] in selected_names]
    game = SantoriniGame(5, sequential_placement=True)
    results = []
    for config in configs:
        print("Training {}...".format(config["name"]), flush=True)
        best_state, final_state, result = _train_one(
            config, train, selection, args, device, game
        )
        checkpoint_path = os.path.join(args.output_dir, config["name"] + ".pth.tar")
        final_checkpoint_path = os.path.join(
            args.output_dir, config["name"] + ".final.pth.tar"
        )
        checkpoint_payload = {
            "schema_version": 2,
            "config": config,
            "seed": args.seed,
            "training_contract": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "policy_weight": args.policy_weight,
                "policy_epsilon": args.policy_epsilon,
                "alpha_boot": args.alpha_boot,
                "score_temperature": args.score_temperature,
                "stage_reliability": list(args.stage_reliability),
                "engine_corpus": os.path.abspath(args.engine_corpus),
                "run13_component": os.path.abspath(args.run13_component),
                "train_plan": os.path.abspath(args.train_plan),
                "selection_plan": os.path.abspath(args.selection_plan),
                "final_test_touched": False,
            },
        }
        for path, state, role, epoch in (
            (checkpoint_path, best_state, "best_selection_objective", result["best_epoch"]),
            (final_checkpoint_path, final_state, "final_epoch", args.epochs),
        ):
            temporary_checkpoint = path + ".tmp"
            payload = dict(checkpoint_payload)
            payload.update({"state_dict": state, "checkpoint_role": role, "epoch": epoch})
            torch.save(payload, temporary_checkpoint)
            os.replace(temporary_checkpoint, path)
        result["checkpoint"] = os.path.abspath(checkpoint_path)
        result["final_checkpoint"] = os.path.abspath(final_checkpoint_path)
        results.append(result)
        _write_json(os.path.join(args.output_dir, "partial-results.json"), {"results": results})
        print(json.dumps(result["selection"], indent=2, sort_keys=True), flush=True)
    output = {
        "schema_version": 1,
        "type": "santorini_v4_p1b_supervised_screen",
        "device": str(device),
        "torch_version": torch.__version__,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "policy_weight": args.policy_weight,
        "policy_epsilon": args.policy_epsilon,
        "alpha_boot": args.alpha_boot,
        "score_temperature": args.score_temperature,
        "stage_reliability": list(args.stage_reliability),
        "train_examples": len(train.encoded_boards),
        "selection_examples": len(selection.encoded_boards),
        "preparation_seconds": preparation_seconds,
        "final_test_touched": False,
        "results": results,
    }
    output_path = os.path.join(args.output_dir, "results.json")
    _write_json(output_path, output)
    print("Results: {}".format(os.path.abspath(output_path)))


if __name__ == "__main__":
    main()
