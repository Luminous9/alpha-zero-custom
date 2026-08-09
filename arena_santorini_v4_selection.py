"""Run paired P1b selection arenas without touching final-test openings/seeds."""

import argparse
import hashlib
import json
import math
import os
import time

import numpy as np
import torch

from BatchedArena import BatchedMCTSArena
from santorini.SantoriniGame import SantoriniGame
from santorini.pytorch.NNet import V3NNetWrapper, args as legacy_nnet_args
from santorini.pytorch.V4NNet import V4InferenceWrapper
from utils import dotdict


DEFAULT_SELECTION_SEED = 20260814


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player1", required=True, help="Checkpoint path.")
    parser.add_argument("--player2", required=True, help="Checkpoint path.")
    parser.add_argument("--player1-kind", choices=("v4", "v3"), default="v4")
    parser.add_argument("--player2-kind", choices=("v4", "v3"), default="v4")
    parser.add_argument("--player1-name", default="player1")
    parser.add_argument("--player2-name", default="player2")
    parser.add_argument("--gate", choices=("standard", "full"), default="standard")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--simulations", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--search-mode", choices=("puct", "gumbel"), default="gumbel")
    parser.add_argument("--gumbel-scale", type=float, default=0.0)
    parser.add_argument("--placement-gumbel-scale", type=float, default=1.5)
    parser.add_argument("--player1-root-symmetries", type=int, default=1)
    parser.add_argument("--player2-root-symmetries", type=int, default=1)
    parser.add_argument("--inference-cache-size", type=int, default=4096)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SELECTION_SEED)
    parser.add_argument("--output", required=True)
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


def _resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(name)


def _load_player(game, kind, path, device, fp16):
    path = os.path.abspath(path)
    if kind == "v4":
        return V4InferenceWrapper(
            game,
            path,
            device=device,
            autocast_fp16=fp16,
            freeze_torchscript=True,
        )
    if fp16:
        raise ValueError("The legacy V3 adapter does not expose FP16 inference.")
    legacy_nnet_args.cuda = device.type == "cuda"
    wrapper = V3NNetWrapper(game)
    wrapper.load_checkpoint(os.path.dirname(path), os.path.basename(path))
    return wrapper


def _search_args(args, root_symmetries):
    return dotdict({
        "numMCTSSims": int(args.simulations),
        "cpuct": 1.0,
        "searchMode": args.search_mode,
        "gumbelMaxConsideredActions": 16,
        "gumbelScale": float(args.gumbel_scale),
        "gumbelPlacementScale": float(args.placement_gumbel_scale),
        "tacticalShortcuts": True,
        "searchSymmetryEvaluation": int(root_symmetries) > 1,
        "rootSymmetrySamples": int(root_symmetries),
        "placementRootSymmetrySamples": int(root_symmetries),
        "inferenceDeduplication": True,
        "inferenceCacheSize": int(args.inference_cache_size),
    })


def _orient_board(board, symmetry_id):
    rotations, flip = divmod(int(symmetry_id), 2)
    board = np.rot90(board, rotations, axes=(-2, -1))
    if flip:
        board = np.flip(board, axis=-1)
    return np.ascontiguousarray(board)


def _standard_selection_openings(args, game):
    needed = args.games // 2
    rng = np.random.RandomState(args.seed)
    with np.load(args.selection_plan, allow_pickle=False) as plan, np.load(
        args.engine_corpus, allow_pickle=False
    ) as engine, np.load(args.run13_component, allow_pickle=False) as run13:
        if np.any(plan["split_ids"] != 1):
            raise ValueError("Selection arena plan contains a non-selection split.")
        order = rng.permutation(len(plan["position_indices"]))
        boards = []
        records = []
        seen = set()
        for draw_index in order:
            corpus_id = int(plan["corpus_ids"][draw_index])
            position_index = int(plan["position_indices"][draw_index])
            key = (corpus_id, position_index)
            if key in seen:
                continue
            payload = engine if corpus_id == 0 else run13
            if int(payload["split_ids"][position_index]) != 1:
                raise ValueError("Selection plan points outside the selection split.")
            board = payload["boards"][position_index].astype(np.int8)
            if np.count_nonzero(board[0]) != 4 or game.getGameEnded(board, 1) != 0:
                continue
            symmetry_id = int(rng.randint(8))
            board = _orient_board(board, symmetry_id)
            seen.add(key)
            boards.append(board)
            records.append({
                "corpus_id": corpus_id,
                "position_index": position_index,
                "symmetry_id": symmetry_id,
                "board_sha256": hashlib.sha256(
                    np.ascontiguousarray(board).tobytes()
                ).hexdigest(),
            })
            if len(boards) == needed:
                break
    if len(boards) != needed:
        raise ValueError("Not enough distinct standard-play selection openings.")
    return boards, records


def _atomic_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _decisive_interval_and_pvalue(wins, losses):
    decisive = int(wins) + int(losses)
    if not decisive:
        return {
            "decisive_games": 0,
            "decisive_win_rate": None,
            "wilson_95_low": None,
            "wilson_95_high": None,
            "exact_two_sided_binomial_p": None,
        }
    wins = int(wins)
    rate = wins / decisive
    z = 1.959963984540054
    denominator = 1.0 + z * z / decisive
    center = (rate + z * z / (2.0 * decisive)) / denominator
    half_width = z * math.sqrt(
        rate * (1.0 - rate) / decisive + z * z / (4.0 * decisive * decisive)
    ) / denominator
    tail = sum(
        math.comb(decisive, count)
        for count in range(max(wins, decisive - wins), decisive + 1)
    ) / (2.0 ** decisive)
    return {
        "decisive_games": decisive,
        "decisive_win_rate": rate,
        "wilson_95_low": center - half_width,
        "wilson_95_high": center + half_width,
        "exact_two_sided_binomial_p": min(1.0, 2.0 * tail),
    }


def _paired_statistics(records, seed, bootstrap_samples=10_000):
    grouped = {}
    for record in records:
        grouped.setdefault(int(record["pair_index"]), []).append(record)
    pair_scores = []
    pair_records = []
    for pair_index, group in sorted(grouped.items()):
        if len(group) != 2 or {item["seat_order"] for item in group} != {
            "contestant1_first", "contestant2_first"
        }:
            raise ValueError("Arena did not produce one seat-swapped game per pair.")
        score = sum(
            1.0 if item["winner"] == 1 else 0.5 if item["winner"] == 0 else 0.0
            for item in group
        )
        pair_scores.append(score)
        pair_records.append({
            "pair_index": pair_index,
            "game_seed": group[0]["game_seed"],
            "contestant1_score": score,
            "games": sorted(group, key=lambda item: item["seat_order"]),
        })
    pair_scores = np.asarray(pair_scores, dtype=np.float64)
    pair_wins = int(np.sum(pair_scores > 1.0))
    pair_losses = int(np.sum(pair_scores < 1.0))
    pair_ties = int(np.sum(pair_scores == 1.0))
    decisive = _decisive_interval_and_pvalue(pair_wins, pair_losses)
    rng = np.random.RandomState(int(seed) ^ 0x5A17)
    bootstrap = pair_scores[
        rng.randint(len(pair_scores), size=(int(bootstrap_samples), len(pair_scores)))
    ].mean(axis=1) / 2.0
    return {
        "pairs": len(pair_scores),
        "pair_wins": pair_wins,
        "pair_losses": pair_losses,
        "pair_ties": pair_ties,
        "mean_game_score": float(pair_scores.mean() / 2.0),
        "pair_sign_test": decisive,
        "cluster_bootstrap_samples": int(bootstrap_samples),
        "cluster_bootstrap_95_low": float(np.quantile(bootstrap, 0.025)),
        "cluster_bootstrap_95_high": float(np.quantile(bootstrap, 0.975)),
        "records": pair_records,
    }


def main():
    args = parse_args()
    if args.games < 2 or args.games % 2:
        raise ValueError("Selection arena games must be a positive even number.")
    if args.simulations < 1 or args.batch_size < 1:
        raise ValueError("Simulation and batch sizes must be positive.")
    for value in (args.player1_root_symmetries, args.player2_root_symmetries):
        if value < 1 or value > 8:
            raise ValueError("Root symmetry counts must be in [1, 8].")
    device = _resolve_device(args.device)
    game = SantoriniGame(5, sequential_placement=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    player1 = _load_player(
        game, args.player1_kind, args.player1, device, args.fp16
    )
    player2 = _load_player(
        game, args.player2_kind, args.player2, device, args.fp16
    )
    opening_boards = None
    opening_records = None
    if args.gate == "standard":
        opening_boards, opening_records = _standard_selection_openings(args, game)
    game_seeds = [args.seed + index for index in range(args.games // 2)]
    arena = BatchedMCTSArena(
        game,
        player1,
        player2,
        _search_args(args, args.player1_root_symmetries),
        player_args={
            1: _search_args(args, args.player1_root_symmetries),
            -1: _search_args(args, args.player2_root_symmetries),
        },
        batch_size=args.batch_size,
        quiet=False,
        opening_boards=opening_boards,
        placement_temperature=0.0,
        game_seeds=game_seeds,
    )
    started = time.perf_counter()
    player1_wins, player2_wins, draws = arena.playGames(args.games)
    elapsed = time.perf_counter() - started
    payload = {
        "schema_version": 1,
        "type": "santorini_v4_p1b_selection_arena",
        "gate": args.gate,
        "final_test_touched": False,
        "final_arena_seeds_touched": False,
        "selection_seed": args.seed,
        "games": args.games,
        "simulations": args.simulations,
        "batch_size": args.batch_size,
        "search_mode": args.search_mode,
        "gumbel_scale": args.gumbel_scale,
        "placement_gumbel_scale": args.placement_gumbel_scale,
        "device": str(device),
        "fp16": bool(args.fp16),
        "player1": {
            "name": args.player1_name,
            "kind": args.player1_kind,
            "checkpoint": os.path.abspath(args.player1),
            "root_symmetries": args.player1_root_symmetries,
            "wins": player1_wins,
        },
        "player2": {
            "name": args.player2_name,
            "kind": args.player2_kind,
            "checkpoint": os.path.abspath(args.player2),
            "root_symmetries": args.player2_root_symmetries,
            "wins": player2_wins,
        },
        "draws": draws,
        "player1_score": (player1_wins + 0.5 * draws) / args.games,
        "player1_decisive_statistics": _decisive_interval_and_pvalue(
            player1_wins, player2_wins
        ),
        "paired_statistics": _paired_statistics(arena.game_records, args.seed),
        "elapsed_seconds": elapsed,
        "games_per_second": args.games / elapsed,
        "inference": arena.inferenceDiagnostics(),
        "paired_game_seeds": game_seeds,
        "standard_openings": opening_records,
    }
    _atomic_json(args.output, payload)
    console_payload = dict(payload)
    console_payload["standard_openings_count"] = (
        len(opening_records) if opening_records is not None else 0
    )
    console_payload.pop("standard_openings")
    console_payload["paired_statistics"] = dict(payload["paired_statistics"])
    console_payload["paired_statistics"].pop("records")
    print(json.dumps(console_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
