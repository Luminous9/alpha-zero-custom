"""Run the fresh-suite P2 A2 checkpoint arenas.

Iteration 14 is tested against iterations 11 and 1 on the same newly drawn,
seat-paired standard openings.  The suite is frozen in the output before any
game is played and is intentionally distinct from the longitudinal 20260715
suite used to select Arm D and iteration 11.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from BatchedArena import BatchedMCTSArena
from arena_santorini_v4_p2_arm import _arena_payload, _search_args
from santorini.OracleResearch import file_sha256
from santorini.SantoriniGame import SantoriniGame
from santorini.SantoriniOpeningBook import SantoriniRandomOpeningSampler
from santorini.pytorch.V4NNet import V4InferenceWrapper


SCHEMA_VERSION = 1
DEFAULT_SEED = 20260815


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration1", required=True)
    parser.add_argument("--iteration11", required=True)
    parser.add_argument("--iteration14", required=True)
    parser.add_argument("--games", type=int, default=120)
    parser.add_argument("--simulations", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _atomic_json(path, payload):
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(name)


def _board_sha256(board):
    return hashlib.sha256(np.ascontiguousarray(board, dtype=np.int8).tobytes()).hexdigest()


def _freeze_openings(output_dir, games, seed):
    suite_path = output_dir / "a2-openings.npz"
    manifest_path = output_dir / "a2-openings-manifest.json"
    opening_count = games // 2
    contract = {
        "schema_version": SCHEMA_VERSION,
        "type": "santorini_v4_p2_a2_openings",
        "games": int(games),
        "pairs": int(opening_count),
        "seed": int(seed),
        "random_orientation": True,
        "standard_play_only": True,
        "seat_paired": True,
        "previous_longitudinal_seed": 20260715,
        "fresh_relative_to_longitudinal_suite": int(seed) != 20260715,
        "final_arena_seeds_touched": False,
    }
    if suite_path.exists() or manifest_path.exists():
        if not suite_path.is_file() or not manifest_path.is_file():
            raise ValueError("A2 opening suite and manifest must both exist or both be absent.")
        existing = json.loads(manifest_path.read_text())
        existing_contract = dict(existing)
        for key in ("opening_sha256", "suite_sha256"):
            existing_contract.pop(key, None)
        if existing_contract != contract:
            raise ValueError("Existing A2 suite belongs to a different contract.")
        if file_sha256(suite_path) != existing["suite_sha256"]:
            raise ValueError("Existing A2 suite digest does not match its manifest.")
        with np.load(suite_path, allow_pickle=False) as payload:
            boards = payload["boards"].astype(np.int8)
        return boards, existing

    boards = np.asarray(
        SantoriniRandomOpeningSampler(
            board_size=5,
            random_orientation=True,
            rng=np.random.RandomState(seed),
        ).sample_distinct_arena_suite(opening_count),
        dtype=np.int8,
    )
    if len(boards) != opening_count:
        raise RuntimeError("Opening sampler returned the wrong suite size.")
    hashes = [_board_sha256(board) for board in boards]
    if len(set(hashes)) != len(hashes):
        raise RuntimeError("A2 opening suite contains duplicate oriented boards.")
    temporary = str(suite_path) + ".tmp"
    with open(temporary, "wb") as output:
        np.savez_compressed(
            output,
            schema_version=np.asarray([SCHEMA_VERSION], dtype=np.int16),
            boards=boards,
            game_seeds=np.asarray(
                [seed + index for index in range(opening_count)], dtype=np.int64
            ),
        )
    os.replace(temporary, suite_path)
    manifest = dict(contract)
    manifest.update({
        "opening_sha256": hashes,
        "suite_sha256": file_sha256(suite_path),
    })
    _atomic_json(manifest_path, manifest)
    return boards, manifest


def _load_player(game, path, device):
    return V4InferenceWrapper(
        game,
        path,
        device=device,
        autocast_fp16=False,
        freeze_torchscript=True,
        canonicalize_d4=True,
    )


def _run_matchup(game, anchor, current, openings, args, seed):
    arena = BatchedMCTSArena(
        game,
        anchor,
        current,
        _search_args(args.simulations),
        batch_size=args.batch_size,
        quiet=False,
        opening_boards=openings,
        game_seeds=[seed + index for index in range(len(openings))],
    )
    started = time.perf_counter()
    return _arena_payload(arena, args.games, seed, started)


def main():
    args = parse_args()
    if args.games < 2 or args.games % 2:
        raise ValueError("--games must be a positive even number.")
    if args.simulations < 1 or args.batch_size < 1:
        raise ValueError("Simulation and batch sizes must be positive.")
    if args.seed == 20260715:
        raise ValueError("A2 must not reuse the Arm-D/iteration-11 selection suite.")
    checkpoints = {
        "iteration1": os.path.abspath(args.iteration1),
        "iteration11": os.path.abspath(args.iteration11),
        "iteration14": os.path.abspath(args.iteration14),
    }
    for name, path in checkpoints.items():
        if not os.path.isfile(path):
            raise FileNotFoundError("{} checkpoint not found: {}".format(name, path))

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "a2-summary.json"
    if summary_path.exists():
        raise FileExistsError("Refusing to overwrite completed A2 summary: {}".format(summary_path))
    openings, opening_manifest = _freeze_openings(output_dir, args.games, args.seed)
    device = _device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    game = SantoriniGame(5, sequential_placement=True)
    players = {
        name: _load_player(game, path, device) for name, path in checkpoints.items()
    }
    started = time.perf_counter()
    matchups = {}
    for anchor_name in ("iteration11", "iteration1"):
        print("Starting iteration14 vs {}...".format(anchor_name), flush=True)
        payload = _run_matchup(
            game,
            players[anchor_name],
            players["iteration14"],
            openings,
            args,
            args.seed,
        )
        payload.update({
            "anchor": anchor_name,
            "current": "iteration14",
            "opening_suite_sha256": opening_manifest["suite_sha256"],
        })
        matchups["iteration14_vs_{}".format(anchor_name)] = payload
        _atomic_json(
            output_dir / "iteration14-vs-{}.json".format(anchor_name), payload
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "type": "santorini_v4_p2_a2_fresh_checkpoint_arenas",
        "device": str(device),
        "torch_version": torch.__version__,
        "games_per_matchup": int(args.games),
        "simulations": int(args.simulations),
        "batch_size": int(args.batch_size),
        "search_mode": "gumbel",
        "gumbel_scale": 0.0,
        "canonical_d4": True,
        "checkpoints": {
            name: {"path": path, "sha256": file_sha256(path)}
            for name, path in checkpoints.items()
        },
        "opening_suite": opening_manifest,
        "matchups": matchups,
        "elapsed_seconds": time.perf_counter() - started,
        "final_test_touched": False,
        "final_arena_seeds_touched": False,
        "interpretation": [
            "The two matchups share the same fresh opening/seat blocks for comparability.",
            "This suite was not used to select Arm D or iteration 11.",
            "The arenas measure family-relative strength; external transfer remains an A3 question.",
        ],
    }
    _atomic_json(summary_path, summary)
    print(json.dumps({
        "summary": str(summary_path),
        "device": str(device),
        "elapsed_seconds": summary["elapsed_seconds"],
        "scores": {
            name: {
                "iteration14_score": payload["current_score"],
                "paired_95": [
                    payload["paired"]["cluster_bootstrap_95_low"],
                    payload["paired"]["cluster_bootstrap_95_high"],
                ],
            }
            for name, payload in matchups.items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
