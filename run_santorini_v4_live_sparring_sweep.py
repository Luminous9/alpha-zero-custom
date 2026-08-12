"""Recalibrate oracle sparring under the exact live P2 96/32 contract."""

import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from Coach import Coach
from run_santorini_v4_oracle_sweep import _paired_statistics, select_sparring_rung
from santorini.OracleResearch import file_sha256
from santorini.SantoriniGame import SantoriniGame
from santorini.pytorch.NNet import args as nnet_args, build_nnet
from utils import dotdict


DEFAULT_BUDGETS = (2_500, 5_000, 7_500, 10_000, 20_000)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="temp/santorini_v4_p2_handoff/p2-start.pth.tar",
    )
    parser.add_argument(
        "--oracle-binary",
        default="tools/santorini_oracle/target/release/santorini-oracle",
    )
    parser.add_argument(
        "--output-dir",
        default="temp/santorini_v4_p2_live_sparring_sweep",
    )
    parser.add_argument("--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS)
    parser.add_argument("--games", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260922)
    return parser.parse_args()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _coach_args(binary, nodes, workers):
    return dotdict({
        "trainingMode": "latest",
        "numMCTSSims": 96,
        "cpuct": 1.0,
        "tempThreshold": 15,
        "searchMode": "gumbel",
        "gumbelMaxConsideredActions": 16,
        "gumbelScale": 1.0,
        "gumbelPlacementScale": 1.5,
        "placementScaleExplorationProbability": 0.10,
        "placementExplorationGumbelScale": 2.25,
        "policyTargetTemperature": 1.0,
        "playoutCapRandomization": True,
        "playoutCapFullProbability": 0.25,
        "playoutCapFastSims": 32,
        "playoutCapFullPlacement": True,
        "tacticalShortcuts": True,
        "addDirichletNoise": True,
        "dirichletAlpha": 0.30,
        "dirichletEpsilon": 0.25,
        "symmetryAugmentation": "none",
        "searchSymmetryEvaluation": False,
        "rootSymmetrySamples": 1,
        "placementRootSymmetrySamples": 1,
        "inferenceDeduplication": True,
        "inferenceCacheSize": 4096,
        "oracleSparringProbability": 1.0,
        "oracleSparringNodes": int(nodes),
        "oracleSparringWorkers": int(workers),
        "oracleSparringOpeningSeed": 20260921,
        "oracleSparringLadderVersion": 2,
        "oracleBinary": str(binary),
        "telemetryMatchGames": 0,
        "telemetryPlacementGames": 0,
        "quiet": True,
    })


def _configure_network_runtime():
    nnet_args.cuda = False
    nnet_args.optimizer = "adamw"
    nnet_args.lr = 3e-4
    nnet_args.weight_decay = 1e-4
    nnet_args.lr_schedule = [(200, 1e-4), (400, 3e-5)]
    nnet_args.replay_reuse = 16.0
    nnet_args.v4_freeze_torchscript = True
    nnet_args.v4_autocast_fp16 = False


def _run_budget(checkpoint, binary, nodes, games, workers, bootstrap_samples, seed):
    # Loading the resumable handoff with optimizer state deliberately restores
    # the same RNG state that the P100 iteration-1 smoke began from.
    random.seed(20260930)
    np.random.seed(20260930)
    torch.manual_seed(20260930)
    game = SantoriniGame(5, sequential_placement=True)
    network = build_nnet(game, "v4")
    network.load_checkpoint(
        os.path.dirname(checkpoint),
        os.path.basename(checkpoint),
        load_optimizer=True,
    )
    coach = Coach(game, network, _coach_args(binary, nodes, workers))
    started = time.perf_counter()
    try:
        examples = coach.executeOracleSparringEpisodes(games, iteration=1)
        stats = dict(coach._oracle_sparring_stats)
        search = coach._searchSymmetryTelemetry()
        playout = coach._playoutCapTelemetry()
    finally:
        coach.close()
    elapsed = time.perf_counter() - started

    by_pair = {}
    for record in stats["game_records"]:
        by_pair.setdefault(int(record["pair_index"]), []).append(record)
    if len(by_pair) != games // 2 or any(len(records) != 2 for records in by_pair.values()):
        raise RuntimeError("Live sweep did not complete every paired opening.")
    pair_records = []
    for pair_index, records in sorted(by_pair.items()):
        score = sum(
            1.0 if record["neural_result"] == 1
            else 0.5 if record["neural_result"] == 0 else 0.0
            for record in records
        )
        pair_records.append({
            "pair_index": pair_index,
            "opening_sha256": records[0]["opening_hash"],
            "v4_pair_score": score,
            "games": records,
        })
    paired = _paired_statistics(pair_records, bootstrap_samples, seed ^ int(nodes))
    return {
        "oracle_nodes": int(nodes),
        "games": int(games),
        "v4_wins": int(stats["neural_wins"]),
        "oracle_wins": int(stats["oracle_wins"]),
        "draws": int(stats["draws"]),
        "v4_score": paired["v4_score"],
        "stored_examples": int(len(examples)),
        "paired_statistics": paired,
        "pair_records": pair_records,
        "elapsed_seconds": elapsed,
        "games_per_second": float(games / elapsed),
        "search_telemetry": search,
        "playout_cap_telemetry": playout,
    }


def run(args):
    if args.games < 2 or args.games % 2:
        raise ValueError("--games must be a positive even number.")
    if args.workers < 1 or args.bootstrap_samples < 1:
        raise ValueError("Worker and bootstrap counts must be positive.")
    budgets = sorted(set(int(value) for value in args.budgets))
    if not budgets or any(value < 1 for value in budgets):
        raise ValueError("Oracle budgets must be positive.")
    for path in (args.checkpoint, args.oracle_binary):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    _configure_network_runtime()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema_version": 1,
        "type": "santorini_v4_p2_live_sparring_calibration",
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "oracle_binary_sha256": file_sha256(args.oracle_binary),
        "budgets": budgets,
        "games_per_budget": int(args.games),
        "paired_openings_per_budget": int(args.games // 2),
        "workers": int(args.workers),
        "iteration": 1,
        "search_mode": "gumbel",
        "gumbel_scale": 1.0,
        "full_simulations": 96,
        "fast_simulations": 32,
        "full_search_probability": 0.25,
        "policy_target_temperature": 1.0,
        "opening_seed": 20260921,
        "iteration_addressed_opening_seed": 20260921 + 1_000_003,
        "rng_source": "resumable_p2_start_checkpoint",
        "oracle_reset_policy": "every_game_boundary",
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "target_low": 0.35,
        "target_high": 0.50,
        "target_midpoint": 0.425,
        "final_test_touched": False,
        "final_arena_seeds_touched": False,
    }
    _atomic_json(output_dir / "sweep-contract.json", contract)

    results = []
    for nodes in budgets:
        path = output_dir / "oracle-{:07d}.json".format(nodes)
        if path.is_file():
            payload = json.loads(path.read_text())
            if payload.get("contract") != contract:
                raise ValueError("Existing result uses a different contract: {}".format(path))
            result = payload["result"]
            print("Reusing", path, flush=True)
        else:
            print("Running live sparring budget {} nodes...".format(nodes), flush=True)
            result = _run_budget(
                args.checkpoint,
                args.oracle_binary,
                nodes,
                args.games,
                args.workers,
                args.bootstrap_samples,
                args.bootstrap_seed,
            )
            _atomic_json(path, {"contract": contract, "result": result})
        results.append(result)
        print(
            "{} nodes: V4 {}/{} ({:.1%}), paired 95% {:.1%}-{:.1%}".format(
                nodes,
                result["v4_wins"],
                result["games"],
                result["v4_score"],
                result["paired_statistics"]["cluster_bootstrap_95_low"],
                result["paired_statistics"]["cluster_bootstrap_95_high"],
            ),
            flush=True,
        )
    summary = {
        "schema_version": 1,
        "type": "santorini_v4_p2_live_sparring_sweep_summary",
        "contract": contract,
        "results": results,
        "selection": select_sparring_rung(results),
    }
    _atomic_json(output_dir / "live-sparring-sweep-summary.json", summary)
    return summary


def main():
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
