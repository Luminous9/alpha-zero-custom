"""Calibrate the P2-start santorini-ai sparring rung against frozen P1c V4."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from Arena import Arena
from pit_santorini import NetworkMCTSPlayer
from santorini.OracleResearch import file_sha256
from santorini.SantoriniGame import SantoriniGame
from santorini.SantoriniOpeningBook import SantoriniRandomOpeningSampler
from santorini.SantoriniOracle import SantoriniOraclePlayer, SantoriniOracleProcess
from santorini.pytorch.V4NNet import V4InferenceWrapper


DEFAULT_BUDGETS = (5_000, 10_000, 20_000, 50_000, 100_000, 250_000)
DEFAULT_GAMES = 40
DEFAULT_SIMULATIONS = 96
DEFAULT_OPENING_SEED = 20260921
DEFAULT_BOOTSTRAP_SEED = 20260922
TARGET_LOW = 0.35
TARGET_HIGH = 0.50
TARGET_MIDPOINT = 0.425


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--oracle-binary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS)
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES)
    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
    parser.add_argument("--opening-seed", type=int, default=DEFAULT_OPENING_SEED)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--inference-cache-size", type=int, default=4096)
    return parser.parse_args()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(name)


def _board_sha256(board):
    return hashlib.sha256(np.ascontiguousarray(board).tobytes()).hexdigest()


def build_opening_suite(game, pairs, seed):
    sampler = SantoriniRandomOpeningSampler(
        board_size=game.n,
        random_orientation=True,
        rng=np.random.RandomState(int(seed)),
    )
    boards = sampler.sample_distinct_arena_suite(int(pairs))
    if len(boards) != int(pairs):
        raise ValueError("Opening sampler returned the wrong number of boards.")
    if len({_board_sha256(board) for board in boards}) != len(boards):
        raise ValueError("Oracle sweep openings must be exactly distinct.")
    for board in boards:
        if np.count_nonzero(board[0]) != 4 or game.getGameEnded(board, 1) != 0:
            raise ValueError("Oracle sweep requires nonterminal completed placements.")
    return boards


def _contract(args, opening_boards):
    budgets = sorted(set(map(int, args.budgets)))
    return {
        "schema_version": 1,
        "type": "santorini_v4_p2_oracle_sparring_calibration",
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "oracle_binary_sha256": file_sha256(args.oracle_binary),
        "oracle_budgets": budgets,
        "games_per_budget": int(args.games),
        "paired_openings_per_budget": int(args.games // 2),
        "simulations": int(args.simulations),
        "search_mode": "gumbel",
        "gumbel_scale": 0.0,
        "action_temperature": 0.0,
        "root_symmetry_samples": 1,
        "canonical_d4": True,
        "inference_cache_size": int(args.inference_cache_size),
        "fp16": bool(args.fp16),
        "opening_seed": int(args.opening_seed),
        "opening_sha256": [_board_sha256(board) for board in opening_boards],
        "bootstrap_seed": int(args.bootstrap_seed),
        "bootstrap_samples": int(args.bootstrap_samples),
        "target_v4_score_low": TARGET_LOW,
        "target_v4_score_high": TARGET_HIGH,
        "target_v4_score_midpoint": TARGET_MIDPOINT,
        "oracle_reset_policy": "every_game_boundary",
        "final_test_touched": False,
        "final_arena_seeds_touched": False,
    }


def _contract_fingerprint(contract):
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _seed_game(seed, device):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def _play_game(game, first, second, opening, seed, device):
    _seed_game(seed, device)
    return Arena(
        first,
        second,
        game,
        opening_boards=[opening],
    ).playGame(opening_board=opening)


def _paired_statistics(pair_records, samples, seed):
    scores = np.asarray(
        [record["v4_pair_score"] for record in pair_records], dtype=np.float64
    )
    if scores.ndim != 1 or not len(scores) or np.any((scores < 0) | (scores > 2)):
        raise ValueError("Invalid paired V4 scores.")
    rng = np.random.RandomState(int(seed))
    means = []
    remaining = int(samples)
    while remaining:
        count = min(500, remaining)
        indices = rng.randint(len(scores), size=(count, len(scores)))
        means.append(scores[indices].mean(axis=1) / 2.0)
        remaining -= count
    interval = np.quantile(np.concatenate(means), (0.025, 0.975))
    return {
        "pairs": int(len(scores)),
        "v4_pair_wins_2_0": int(np.sum(scores == 2.0)),
        "split_pairs_1_1": int(np.sum(scores == 1.0)),
        "v4_pair_losses_0_2": int(np.sum(scores == 0.0)),
        "v4_score": float(np.mean(scores) / 2.0),
        "cluster_bootstrap_samples": int(samples),
        "cluster_bootstrap_95_low": float(interval[0]),
        "cluster_bootstrap_95_high": float(interval[1]),
    }


def _run_budget(
    game,
    neural,
    oracle_process,
    opening_boards,
    nodes,
    simulations,
    device,
    bootstrap_samples,
    bootstrap_seed,
    opening_seed,
    inference_cache_size,
):
    neural_player = NetworkMCTSPlayer(
        game,
        neural,
        simulations,
        action_temp=0.0,
        search_mode="gumbel",
        gumbel_max_considered_actions=16,
        gumbel_scale=0.0,
        gumbel_placement_scale=0.0,
        search_symmetry_evaluation=False,
        root_symmetry_samples=1,
        placement_root_symmetry_samples=1,
        inference_deduplication=True,
        inference_cache_size=int(inference_cache_size),
    )
    oracle_player = SantoriniOraclePlayer(game, oracle_process, nodes=int(nodes))
    records = []
    v4_wins = oracle_wins = draws = 0
    started = time.perf_counter()
    for pair_index, opening in enumerate(opening_boards):
        games = []
        first_seed = int(opening_seed) + 2 * pair_index
        result = _play_game(
            game, neural_player, oracle_player, opening, first_seed, device
        )
        v4_result = int(result)
        games.append({
            "seat_order": "v4_first",
            "game_seed": first_seed,
            "winner": "v4" if v4_result == 1 else "oracle" if v4_result == -1 else "draw",
        })

        second_seed = first_seed + 1
        result = _play_game(
            game, oracle_player, neural_player, opening, second_seed, device
        )
        v4_result_second = -int(result)
        games.append({
            "seat_order": "oracle_first",
            "game_seed": second_seed,
            "winner": (
                "v4" if v4_result_second == 1
                else "oracle" if v4_result_second == -1 else "draw"
            ),
        })
        results = (v4_result, v4_result_second)
        v4_wins += sum(value == 1 for value in results)
        oracle_wins += sum(value == -1 for value in results)
        draws += sum(value == 0 for value in results)
        pair_score = sum(1.0 if value == 1 else 0.5 if value == 0 else 0.0 for value in results)
        records.append({
            "pair_index": int(pair_index),
            "opening_sha256": _board_sha256(opening),
            "v4_pair_score": float(pair_score),
            "games": games,
        })
        print(
            "oracle {:>7} nodes: pair {}/{} complete (V4 score {:.1f}/2)".format(
                int(nodes), pair_index + 1, len(opening_boards), pair_score
            ),
            flush=True,
        )
    elapsed = time.perf_counter() - started
    paired = _paired_statistics(
        records,
        bootstrap_samples,
        int(bootstrap_seed) ^ int(nodes),
    )
    return {
        "oracle_nodes": int(nodes),
        "games": int(2 * len(opening_boards)),
        "v4_wins": int(v4_wins),
        "oracle_wins": int(oracle_wins),
        "draws": int(draws),
        "v4_score": paired["v4_score"],
        "paired_statistics": paired,
        "pair_records": records,
        "elapsed_seconds": float(elapsed),
        "games_per_second": float(2 * len(opening_boards) / elapsed),
    }


def select_sparring_rung(rows, low=TARGET_LOW, high=TARGET_HIGH, target=TARGET_MIDPOINT):
    """Apply the frozen point-estimate rule and recommend a follow-up if needed."""
    normalized = sorted(
        ({"oracle_nodes": int(row["oracle_nodes"]), "v4_score": float(row["v4_score"])} for row in rows),
        key=lambda row: row["oracle_nodes"],
    )
    if not normalized or len({row["oracle_nodes"] for row in normalized}) != len(normalized):
        raise ValueError("Rung selection requires unique budget rows.")
    if any(not 0.0 <= row["v4_score"] <= 1.0 for row in normalized):
        raise ValueError("V4 scores must be in [0, 1].")
    qualifying = [row for row in normalized if low <= row["v4_score"] <= high]
    if qualifying:
        selected = min(
            qualifying,
            key=lambda row: (abs(row["v4_score"] - target), -row["oracle_nodes"]),
        )
        return {
            "decision": "selected",
            "selected_oracle_nodes": selected["oracle_nodes"],
            "selected_v4_score": selected["v4_score"],
            "recommended_next_budget": None,
            "reason": "closest qualifying score to target midpoint; exact ties prefer the higher budget",
        }

    scores = np.asarray([row["v4_score"] for row in normalized])
    if np.all(scores > high):
        next_budget = normalized[-1]["oracle_nodes"] * 2
        reason = "V4 is above the target band at every tested rung; strengthen the oracle"
    elif np.all(scores < low):
        next_budget = max(1, normalized[0]["oracle_nodes"] // 2)
        reason = "V4 is below the target band at every tested rung; weaken the oracle"
    else:
        next_budget = None
        for left, right in zip(normalized, normalized[1:]):
            if left["v4_score"] > high and right["v4_score"] < low:
                next_budget = int(round(math.sqrt(
                    left["oracle_nodes"] * right["oracle_nodes"]
                ) / 100.0) * 100)
                break
        if next_budget is None:
            reason = "no rung qualifies and sampling noise is non-monotonic; increase paired games"
        else:
            reason = "scores straddle the target band; test the logarithmic midpoint"
    return {
        "decision": "extend_sweep" if next_budget is not None else "inconclusive",
        "selected_oracle_nodes": None,
        "selected_v4_score": None,
        "recommended_next_budget": next_budget,
        "reason": reason,
    }


def _load_existing_budget(path, fingerprint, nodes):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError("Existing oracle sweep result is unreadable: {}".format(path)) from exc
    if (
        payload.get("contract_fingerprint") != fingerprint
        or payload.get("result", {}).get("oracle_nodes") != int(nodes)
        or payload.get("contract", {}).get("games_per_budget")
        != payload.get("result", {}).get("games")
        or payload.get("final_test_touched", True)
        or payload.get("final_arena_seeds_touched", True)
    ):
        raise ValueError("Existing oracle sweep result has a different contract: {}".format(path))
    return payload


def run(args):
    budgets = sorted(set(map(int, args.budgets)))
    if not budgets or any(budget < 1 for budget in budgets):
        raise ValueError("Oracle budgets must be positive.")
    if args.games < 2 or args.games % 2:
        raise ValueError("Games must be a positive even number.")
    if args.simulations < 1 or args.bootstrap_samples < 1:
        raise ValueError("Simulation and bootstrap counts must be positive.")
    if args.inference_cache_size < 0:
        raise ValueError("Inference cache size cannot be negative.")
    if args.fp16 and args.device == "cpu":
        raise ValueError("FP16 calibration requires CUDA.")
    for path in (args.checkpoint, args.oracle_binary):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    device = _resolve_device(args.device)
    game = SantoriniGame(5, sequential_placement=True)
    openings = build_opening_suite(game, args.games // 2, args.opening_seed)
    contract = _contract(args, openings)
    fingerprint = _contract_fingerprint(contract)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "sweep-contract.json"
    if contract_path.is_file():
        existing = json.loads(contract_path.read_text())
        if existing != contract:
            raise ValueError("Output directory already contains a different sweep contract.")
    else:
        _atomic_json(contract_path, contract)

    _seed_game(args.opening_seed, device)
    neural = V4InferenceWrapper(
        game,
        args.checkpoint,
        device=device,
        autocast_fp16=args.fp16,
        freeze_torchscript=True,
        canonicalize_d4=True,
        canonical_cache_size=args.inference_cache_size,
    )
    results = []
    with SantoriniOracleProcess(args.oracle_binary) as oracle_process:
        oracle_version = oracle_process.info.get("version")
        for nodes in budgets:
            path = output_dir / "oracle-{:07d}.json".format(nodes)
            existing = _load_existing_budget(path, fingerprint, nodes)
            if existing is not None:
                print("Reusing complete budget result:", path, flush=True)
                results.append(existing["result"])
                continue
            result = _run_budget(
                game,
                neural,
                oracle_process,
                openings,
                nodes,
                args.simulations,
                device,
                args.bootstrap_samples,
                args.bootstrap_seed,
                args.opening_seed,
                args.inference_cache_size,
            )
            payload = {
                "schema_version": 1,
                "type": "santorini_v4_p2_oracle_budget_result",
                "contract_fingerprint": fingerprint,
                "contract": contract,
                "oracle_version": oracle_version,
                "result": result,
                "final_test_touched": False,
                "final_arena_seeds_touched": False,
            }
            _atomic_json(path, payload)
            results.append(result)

    selection = select_sparring_rung(results)
    summary = {
        "schema_version": 1,
        "type": "santorini_v4_p2_oracle_sweep_summary",
        "contract_fingerprint": fingerprint,
        "contract": contract,
        "oracle_version": oracle_version,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "results": sorted(results, key=lambda row: row["oracle_nodes"]),
        "selection": selection,
        "final_test_touched": False,
        "final_arena_seeds_touched": False,
    }
    _atomic_json(output_dir / "oracle-sweep-summary.json", summary)
    return summary


def main():
    summary = run(parse_args())
    console = dict(summary)
    console["results"] = [
        {
            "oracle_nodes": row["oracle_nodes"],
            "v4_score": row["v4_score"],
            "cluster_bootstrap_95": [
                row["paired_statistics"]["cluster_bootstrap_95_low"],
                row["paired_statistics"]["cluster_bootstrap_95_high"],
            ],
            "elapsed_seconds": row["elapsed_seconds"],
        }
        for row in summary["results"]
    ]
    print(json.dumps(console, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
