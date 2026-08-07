import argparse
import json
import os
import sys

import numpy as np

import Arena
from pit_santorini import NeuralMCTSPlayer, display_name_from_folder
from santorini.SantoriniGame import SantoriniGame
from santorini.SantoriniOpeningBook import SantoriniRandomOpeningSampler
from santorini.SantoriniOracle import (SantoriniOraclePlayer,
                                       SantoriniOracleProcess)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pit a Santorini neural MCTS checkpoint against the fixed-node Rust oracle."
    )
    parser.add_argument("--checkpoint-folder", default="./temp/santorini_v3_run13_gumbel")
    parser.add_argument("--checkpoint-file", default="latest.pth.tar")
    parser.add_argument("--architecture", choices=["v2", "v3"], default="v3")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--sims", type=int, default=1024)
    parser.add_argument("--search-mode", choices=["puct", "gumbel"], default="gumbel")
    parser.add_argument("--gumbel-max-considered-actions", type=int, default=16)
    parser.add_argument("--gumbel-scale", type=float, default=0.0)
    parser.add_argument("--oracle-nodes", type=int, default=20_000)
    parser.add_argument("--oracle-binary")
    parser.add_argument("--opening-seed", type=int, default=20260721)
    parser.add_argument("--root-symmetry-samples", type=int, default=8)
    parser.add_argument("--json-out")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.games < 2 or args.games % 2:
        raise ValueError("--games must be a positive even number.")
    if args.oracle_nodes < 1 or args.sims < 1:
        raise ValueError("--oracle-nodes and --sims must be positive.")

    checkpoint_path = os.path.join(args.checkpoint_folder, args.checkpoint_file)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))

    # Start after placement because the first bridge deliberately excludes the
    # external engine's joint two-worker placement action.
    game = SantoriniGame(5, sequential_placement=True)
    opening_rng = np.random.RandomState(args.opening_seed)
    sampler = SantoriniRandomOpeningSampler(random_orientation=True, rng=opening_rng)
    opening_boards = sampler.sample_distinct_arena_suite(args.games // 2)

    neural = NeuralMCTSPlayer(
        game,
        args.checkpoint_folder,
        args.checkpoint_file,
        args.sims,
        architecture=args.architecture,
        action_temp=0.0,
        search_mode=args.search_mode,
        gumbel_max_considered_actions=args.gumbel_max_considered_actions,
        gumbel_scale=args.gumbel_scale,
        gumbel_placement_scale=args.gumbel_scale,
        search_symmetry_evaluation=args.architecture == "v3",
        root_symmetry_samples=args.root_symmetry_samples,
        placement_root_symmetry_samples=args.root_symmetry_samples,
        inference_deduplication=args.architecture == "v3",
    )

    with SantoriniOracleProcess(args.oracle_binary) as oracle_process:
        oracle = SantoriniOraclePlayer(game, oracle_process, nodes=args.oracle_nodes)
        arena = Arena.Arena(
            neural,
            oracle,
            game,
            display=SantoriniGame.display,
            opening_boards=opening_boards,
            progress_file=sys.stdout,
        )
        neural_wins, oracle_wins, draws = arena.playGames(args.games, verbose=False)

    result = {
        "architecture": args.architecture,
        "checkpoint_file": args.checkpoint_file,
        "checkpoint_folder": os.path.abspath(args.checkpoint_folder),
        "contestant1_name": display_name_from_folder(args.checkpoint_folder),
        "contestant1_wins": neural_wins,
        "contestant2_name": "santorini-ai",
        "contestant2_wins": oracle_wins,
        "draws": draws,
        "games": args.games,
        "gumbel_max_considered_actions": args.gumbel_max_considered_actions,
        "gumbel_scale": args.gumbel_scale,
        "opening_mode": "sampled_symmetry_distinct_completed",
        "opening_seed": args.opening_seed,
        "oracle_nodes": args.oracle_nodes,
        "search_mode": args.search_mode,
        "sims": args.sims,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json_out:
        output_dir = os.path.dirname(os.path.abspath(args.json_out))
        os.makedirs(output_dir, exist_ok=True)
        with open(args.json_out, "w") as output_file:
            json.dump(result, output_file, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
