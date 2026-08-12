"""Run the frozen P1c Gate G1 protocol against Run13."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from arena_santorini_v4_selection import (
    _load_player,
    _standard_selection_openings,
)
from santorini.OracleResearch import file_sha256
from santorini.SantoriniGame import SantoriniGame


SELECTION_SEED = 20260901
SUMMARY_SEED = 20260902
GAMES_PER_GATE = 40
EQUAL_SIMULATION_BUDGETS = (96, 128)
EQUAL_COST_ANCHOR = 128
CALIBRATION_BATCH_SIZE = 8
CALIBRATION_BOARDS = 64
CALIBRATION_WARMUP_BATCHES = 12
CALIBRATION_ROUNDS = 7
CALIBRATION_BATCHES_PER_ROUND = 30


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--run13", required=True)
    parser.add_argument("--engine-corpus", required=True)
    parser.add_argument("--run13-component", required=True)
    parser.add_argument("--selection-plan", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def choose_equal_cost_simulations(candidate_seconds, run13_seconds, anchor=128):
    costs = {
        "p1c": float(candidate_seconds),
        "run13": float(run13_seconds),
    }
    if any(not np.isfinite(value) or value <= 0 for value in costs.values()):
        raise ValueError("Inference calibration costs must be positive and finite.")
    # Run13's evaluation contract averages all eight D4 frames at every root.
    # The nominal simulation count includes one root evaluation, so this costs
    # seven additional neural evaluations per move. Exact canonical V4 uses one.
    root_extras = {"p1c": 0, "run13": 7}
    anchored_costs = {
        name: (int(anchor) + root_extras[name]) * costs[name] for name in costs
    }
    target = min(anchored_costs.values())

    def budget(name, cost):
        raw = target / cost - root_extras[name]
        rounded = int(round(raw / 8.0) * 8)
        return max(16, min(int(anchor), rounded))

    return {name: budget(name, cost) for name, cost in costs.items()}


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _benchmark_round(wrapper, batches, repetitions, device, offset):
    _sync(device)
    started = time.perf_counter()
    for index in range(int(repetitions)):
        wrapper.predict_batch(batches[(offset + index) % len(batches)])
    _sync(device)
    elapsed = time.perf_counter() - started
    examples = int(repetitions) * len(batches[0])
    return {
        "elapsed_seconds": float(elapsed),
        "examples": examples,
        "seconds_per_example": float(elapsed / examples),
        "examples_per_second": float(examples / elapsed),
    }


def calibrate(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("G1 requires CUDA but none is available.")
    game = SantoriniGame(5, sequential_placement=True)
    opening_args = argparse.Namespace(
        games=CALIBRATION_BOARDS * 2,
        seed=SELECTION_SEED,
        selection_plan=args.selection_plan,
        engine_corpus=args.engine_corpus,
        run13_component=args.run13_component,
    )
    boards, records = _standard_selection_openings(opening_args, game)
    batches = [
        np.asarray(boards[start:start + CALIBRATION_BATCH_SIZE])
        for start in range(0, len(boards), CALIBRATION_BATCH_SIZE)
    ]
    if any(len(batch) != CALIBRATION_BATCH_SIZE for batch in batches):
        raise AssertionError("Calibration boards do not make complete batches.")

    candidate = _load_player(
        game, "v4", args.candidate, device, False, canonical_d4=True
    )
    run13 = _load_player(
        game, "v3", args.run13, device, False, canonical_d4=False
    )
    wrappers = {"p1c": candidate, "run13": run13}
    for name in ("p1c", "run13"):
        for index in range(CALIBRATION_WARMUP_BATCHES):
            wrappers[name].predict_batch(batches[index % len(batches)])
    _sync(device)

    measurements = {"p1c": [], "run13": []}
    for round_index in range(CALIBRATION_ROUNDS):
        order = ("p1c", "run13") if round_index % 2 == 0 else ("run13", "p1c")
        for name in order:
            measurements[name].append(_benchmark_round(
                wrappers[name],
                batches,
                CALIBRATION_BATCHES_PER_ROUND,
                device,
                round_index,
            ))

    medians = {
        name: float(np.median([
            item["seconds_per_example"] for item in measurements[name]
        ]))
        for name in measurements
    }
    budgets = choose_equal_cost_simulations(
        medians["p1c"], medians["run13"], EQUAL_COST_ANCHOR
    )
    predicted_costs = {
        "p1c": medians["p1c"] * budgets["p1c"],
        "run13": medians["run13"] * (budgets["run13"] + 7),
    }
    return {
        "schema_version": 1,
        "contract": "santorini_v4_g1_pre_arena_inference_calibration",
        "performed_before_games": True,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "selection_seed": SELECTION_SEED,
        "final_test_touched": False,
        "final_arena_seeds_touched": False,
        "batch_size": CALIBRATION_BATCH_SIZE,
        "boards": len(boards),
        "board_sha256": [record["board_sha256"] for record in records],
        "warmup_batches_per_model": CALIBRATION_WARMUP_BATCHES,
        "rounds": CALIBRATION_ROUNDS,
        "batches_per_round": CALIBRATION_BATCHES_PER_ROUND,
        "measurements": measurements,
        "median_seconds_per_example": medians,
        "p1c_over_run13_cost_ratio": medians["p1c"] / medians["run13"],
        "equal_cost_budget_rule": (
            "At batch eight, compare predicted per-move model time at 128 "
            "simulations, including Run13's seven extra evaluations for its "
            "eight-frame root average. Anchor the cheaper player at 128 and "
            "set the other to the nearest equal-time multiple of eight, "
            "clamped to [16, 128]."
        ),
        "equal_cost_anchor": EQUAL_COST_ANCHOR,
        "equal_cost_simulations": budgets,
        "predicted_model_seconds_per_move": predicted_costs,
        "predicted_cost_mismatch_fraction": abs(
            predicted_costs["p1c"] - predicted_costs["run13"]
        ) / max(predicted_costs.values()),
        "inputs": {
            "candidate_sha256": file_sha256(args.candidate),
            "run13_sha256": file_sha256(args.run13),
            "engine_corpus_sha256": file_sha256(args.engine_corpus),
            "run13_component_sha256": file_sha256(args.run13_component),
            "selection_plan_sha256": file_sha256(args.selection_plan),
        },
    }


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _valid_existing_arena(path, gate, p1_sims, p2_sims, placement_temperature):
    path = Path(path)
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return (
        payload.get("gate") == gate
        and payload.get("games") == GAMES_PER_GATE
        and payload.get("selection_seed") == SELECTION_SEED
        and payload.get("player1_simulations") == int(p1_sims)
        and payload.get("player2_simulations") == int(p2_sims)
        and payload.get("placement_temperature") == float(placement_temperature)
        and payload.get("player1", {}).get("kind") == "v4"
        and payload.get("player1", {}).get("canonical_d4") is True
        and payload.get("player1", {}).get("root_symmetries") == 1
        and payload.get("player1", {}).get("placement_root_symmetries") == 1
        and payload.get("player2", {}).get("kind") == "v3"
        and payload.get("player2", {}).get("canonical_d4") is False
        and payload.get("player2", {}).get("root_symmetries") == 8
        and payload.get("player2", {}).get("placement_root_symmetries") == 8
        and not payload.get("final_test_touched", True)
        and not payload.get("final_arena_seeds_touched", True)
    )


def _run_arena(args, output, gate, p1_sims, p2_sims):
    placement_temperature = 1.0 if gate == "full" else 0.0
    if _valid_existing_arena(
        output, gate, p1_sims, p2_sims, placement_temperature
    ):
        print("Reusing complete arena:", output, flush=True)
        return
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "arena_santorini_v4_selection.py"),
        "--player1", os.path.abspath(args.candidate),
        "--player1-kind", "v4",
        "--player1-name", "p1c",
        "--player2", os.path.abspath(args.run13),
        "--player2-kind", "v3",
        "--player2-name", "run13",
        "--gate", gate,
        "--games", str(GAMES_PER_GATE),
        "--simulations", str(p1_sims),
        "--player1-simulations", str(p1_sims),
        "--player2-simulations", str(p2_sims),
        "--batch-size", "32",
        "--search-mode", "gumbel",
        "--gumbel-scale", "0",
        "--placement-gumbel-scale", "1.5",
        "--placement-temperature", str(placement_temperature),
        "--player1-root-symmetries", "1",
        "--player1-placement-root-symmetries", "1",
        "--player2-root-symmetries", "8",
        "--player2-placement-root-symmetries", "8",
        "--player1-canonical-d4",
        "--inference-cache-size", "4096",
        "--device", args.device,
        "--seed", str(SELECTION_SEED),
        "--engine-corpus", os.path.abspath(args.engine_corpus),
        "--run13-component", os.path.abspath(args.run13_component),
        "--selection-plan", os.path.abspath(args.selection_plan),
        "--output", os.path.abspath(output),
    ]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parent)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = output_dir / "inference-calibration.json"
    if calibration_path.is_file():
        calibration = json.loads(calibration_path.read_text())
        expected = {
            "candidate_sha256": file_sha256(args.candidate),
            "run13_sha256": file_sha256(args.run13),
            "engine_corpus_sha256": file_sha256(args.engine_corpus),
            "run13_component_sha256": file_sha256(args.run13_component),
            "selection_plan_sha256": file_sha256(args.selection_plan),
        }
        if calibration.get("inputs") != expected:
            raise ValueError("Existing G1 calibration belongs to different inputs.")
        print("Reusing pre-arena calibration:", calibration_path, flush=True)
    else:
        calibration = calibrate(args)
        _atomic_json(calibration_path, calibration)
        print(json.dumps(calibration, indent=2, sort_keys=True), flush=True)

    equal_paths = {}
    for budget in EQUAL_SIMULATION_BUDGETS:
        equal_paths[budget] = {}
        for gate in ("standard", "full"):
            path = output_dir / "equal-{}-{}.json".format(budget, gate)
            _run_arena(args, path, gate, budget, budget)
            equal_paths[budget][gate] = path

    budgets = calibration["equal_cost_simulations"]
    equal_cost_paths = {}
    for gate in ("standard", "full"):
        if budgets["p1c"] == budgets["run13"] == EQUAL_COST_ANCHOR:
            path = equal_paths[EQUAL_COST_ANCHOR][gate]
        else:
            path = output_dir / "equal-cost-{}.json".format(gate)
            _run_arena(args, path, gate, budgets["p1c"], budgets["run13"])
        equal_cost_paths[gate] = path

    summary_path = output_dir / "g1-summary.json"
    summary_command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "summarize_santorini_v4_g1.py"),
        "--calibration", str(calibration_path),
        "--equal-96-standard", str(equal_paths[96]["standard"]),
        "--equal-96-full", str(equal_paths[96]["full"]),
        "--equal-128-standard", str(equal_paths[128]["standard"]),
        "--equal-128-full", str(equal_paths[128]["full"]),
        "--equal-cost-standard", str(equal_cost_paths["standard"]),
        "--equal-cost-full", str(equal_cost_paths["full"]),
        "--bootstrap-samples", "10000",
        "--seed", str(SUMMARY_SEED),
        "--output", str(summary_path),
    ]
    subprocess.run(summary_command, check=True, cwd=Path(__file__).resolve().parent)
    contract = {
        "schema_version": 1,
        "contract": "santorini_v4_p1c_gate_g1_job",
        "selection_seed": SELECTION_SEED,
        "summary_seed": SUMMARY_SEED,
        "games_per_gate": GAMES_PER_GATE,
        "equal_simulation_budgets": list(EQUAL_SIMULATION_BUDGETS),
        "placement_temperature": 1.0,
        "p1c_root_symmetries": {"standard": 1, "placement": 1},
        "run13_root_symmetries": {"standard": 8, "placement": 8},
        "final_test_touched": False,
        "final_arena_seeds_touched": False,
        "outputs": sorted(path.name for path in output_dir.iterdir()),
    }
    _atomic_json(output_dir / "job-contract.json", contract)


if __name__ == "__main__":
    main()
