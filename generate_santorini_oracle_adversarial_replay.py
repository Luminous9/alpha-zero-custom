"""Generate on-policy Santorini corrections from neural-vs-oracle games."""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import time

import numpy as np
from tqdm import tqdm

from santorini.OracleResearch import (
    ParallelOraclePool,
    STAGES,
    blend_policies,
    canonical_d4_fen,
    confidence_metrics,
    ranked_moves_to_v3_policy,
    stage_for_builds,
)
from pit_santorini import NeuralMCTSPlayer, select_legal_action
from santorini.ReplayBuffer import load_compact_replay, save_compact_replay
from santorini.SantoriniGame import SantoriniGame
from santorini.SantoriniOpeningBook import SantoriniRandomOpeningSampler
from santorini.SantoriniOracle import (
    SantoriniOraclePlayer,
    SantoriniOracleProcess,
    external_actions_to_v3_actions,
)


SCHEMA_VERSION = 1
DEFAULT_CHECKPOINT_FOLDER = "./temp/santorini_v3_run13_gumbel"
DEFAULT_OUTPUT = "./temp/run13_oracle_adversarial_1500.examples.npz"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Play a neural checkpoint against the oracle, label on-trajectory neural "
            "decisions, and retain stable high-margin disagreements."
        )
    )
    parser.add_argument("--checkpoint-folder", default=DEFAULT_CHECKPOINT_FOLDER)
    parser.add_argument("--checkpoint-file", default="latest.pth.tar")
    parser.add_argument("--architecture", choices=("v2", "v3"), default="v3")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--sims", type=int, default=128)
    parser.add_argument("--search-mode", choices=("puct", "gumbel"), default="gumbel")
    parser.add_argument("--gumbel-max-considered-actions", type=int, default=16)
    parser.add_argument("--gumbel-scale", type=float, default=0.0)
    parser.add_argument("--oracle-game-nodes", type=int, default=20_000)
    parser.add_argument("--shallow-nodes-per-move", type=int, default=2_000)
    parser.add_argument("--deep-nodes-per-move", type=int, default=10_000)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--score-temperature", type=float, default=100.0)
    parser.add_argument("--oracle-weight", type=float, default=0.50)
    parser.add_argument("--min-top3-jaccard", type=float, default=0.20)
    parser.add_argument("--min-score-margin", type=int, default=25)
    parser.add_argument("--corrections", type=int, default=1_500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--root-symmetry-samples", type=int, default=8)
    parser.add_argument("--oracle-binary")
    parser.add_argument("--trajectories-out")
    parser.add_argument("--records-out")
    parser.add_argument("--metadata-out")
    parser.add_argument("--no-symmetries", action="store_true")
    return parser.parse_args()


def companion_path(output_path, suffix):
    if output_path.endswith(".examples.npz"):
        return output_path[:-len(".examples.npz")] + suffix
    return output_path + suffix


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_jsonl(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as output:
        output.write(json.dumps(payload, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def load_jsonl(path, metadata, identity_keys, id_key):
    if not os.path.exists(path):
        append_jsonl(path, metadata)
        return []
    with open(path) as source:
        lines = [line for line in source if line.strip()]
    if not lines:
        raise ValueError("Existing resume file is empty: {}".format(path))
    stored = json.loads(lines[0])
    if {key: stored[key] for key in identity_keys} != {
        key: metadata[key] for key in identity_keys
    }:
        raise ValueError("Existing resume metadata does not match this run: {}".format(path))
    records = [json.loads(line) for line in lines[1:]]
    ids = [int(record[id_key]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Resume file contains duplicate {} values.".format(id_key))
    return records


def sparse_policy(policy):
    policy = np.asarray(policy, dtype=np.float32)
    indices = np.flatnonzero(policy)
    return {
        "policy_actions": indices.astype(int).tolist(),
        "policy_probabilities": policy[indices].astype(float).tolist(),
    }


def dense_policy(game, position):
    policy = np.zeros(game.getActionSize(), dtype=np.float32)
    policy[np.asarray(position["policy_actions"], dtype=np.int64)] = np.asarray(
        position["policy_probabilities"], dtype=np.float32
    )
    policy /= policy.sum()
    return policy


def trajectory_metadata(args, checkpoint_path, checkpoint_digest, trajectories_path):
    return {
        "type": "metadata",
        "schema_version": SCHEMA_VERSION,
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest,
        "trajectories_path": os.path.abspath(trajectories_path),
        "games": int(args.games),
        "sims": int(args.sims),
        "search_mode": args.search_mode,
        "gumbel_max_considered_actions": int(args.gumbel_max_considered_actions),
        "gumbel_scale": float(args.gumbel_scale),
        "oracle_game_nodes": int(args.oracle_game_nodes),
        "opening_seed": int(args.seed),
        "root_symmetry_samples": int(args.root_symmetry_samples),
    }


TRAJECTORY_IDENTITY = (
    "schema_version", "checkpoint_path", "checkpoint_sha256", "games", "sims",
    "search_mode", "gumbel_max_considered_actions", "gumbel_scale",
    "oracle_game_nodes", "opening_seed", "root_symmetry_samples",
)


def play_adversarial_game(game, neural, oracle, opening_board, neural_side, game_id):
    board = opening_board.copy()
    current_player = 1
    positions = []
    neural.startGame()
    oracle.startGame()
    ply = 0
    while game.getGameEnded(board, current_player) == 0:
        ply += 1
        canonical = game.getCanonicalForm(board, current_player)
        if current_player == neural_side:
            action_policy = neural.mcts.getActionProb(canonical, temp=0)
            training_policy = neural.mcts.getTrainingPolicyFromTree(canonical, temp=1)
            if float(np.sum(training_policy)) <= 0:
                # Exact tactical shortcuts bypass the normal search tree.
                training_policy = action_policy
            action = select_legal_action(game, canonical, action_policy)
            position = {
                "board": np.asarray(canonical, dtype=np.int8).tolist(),
                "chosen_action": int(action),
                "ply": int(ply),
                "stage": stage_for_builds(int(np.sum(canonical[1]))),
            }
            position.update(sparse_policy(training_policy))
            positions.append(position)
        else:
            action = oracle.play(canonical)
        board, current_player = game.getNextState(board, current_player, action)
        if ply > 500:
            raise RuntimeError("Adversarial game exceeded 500 plies.")

    player_one_result = absolute_player_one_result(game, board, current_player)
    neural_result = float(player_one_result * neural_side)
    for position in positions:
        position["value"] = neural_result
    return {
        "type": "game",
        "game_id": int(game_id),
        "neural_side": int(neural_side),
        "neural_result": neural_result,
        "plies": int(ply),
        "positions": positions,
    }


def absolute_player_one_result(game, terminal_board, current_player):
    """Convert the terminal next-player-relative result to absolute player one."""
    result = float(game.getGameEnded(terminal_board, current_player))
    if result == 0:
        raise ValueError("Expected a terminal result after adversarial game completion.")
    return float(current_player * result)


def collect_candidates(game_records):
    by_fen = {}
    observations = 0
    for game_record in game_records:
        for position in game_record["positions"]:
            observations += 1
            board = np.asarray(position["board"], dtype=int)
            fen = canonical_d4_fen(board)
            candidate = dict(position)
            candidate.update({
                "candidate_id": -1,
                "game_id": int(game_record["game_id"]),
                "neural_side": int(game_record["neural_side"]),
                "d4_fen": fen,
                "observations": 1,
            })
            if fen not in by_fen:
                by_fen[fen] = candidate
            else:
                by_fen[fen]["observations"] += 1
                # Prefer a loss example because its value exposes an adversarial failure.
                if candidate["value"] < by_fen[fen]["value"]:
                    candidate["observations"] = by_fen[fen]["observations"]
                    by_fen[fen] = candidate
    candidates = sorted(by_fen.values(), key=lambda record: record["d4_fen"])
    for candidate_id, candidate in enumerate(candidates):
        candidate["candidate_id"] = int(candidate_id)
    return candidates, observations


def label_metadata(args, checkpoint_digest, trajectories_digest, candidates, records_path):
    return {
        "type": "metadata",
        "schema_version": SCHEMA_VERSION,
        "checkpoint_sha256": checkpoint_digest,
        "trajectories_sha256": trajectories_digest,
        "records_path": os.path.abspath(records_path),
        "candidate_count": len(candidates),
        "candidate_fens": [candidate["d4_fen"] for candidate in candidates],
        "shallow_nodes_per_move": int(args.shallow_nodes_per_move),
        "deep_nodes_per_move": int(args.deep_nodes_per_move),
        "top_k": int(args.top_k),
        "score_temperature": float(args.score_temperature),
    }


LABEL_IDENTITY = (
    "schema_version", "checkpoint_sha256", "trajectories_sha256", "candidate_count",
    "candidate_fens", "shallow_nodes_per_move", "deep_nodes_per_move", "top_k",
    "score_temperature",
)


def analyze_candidate(pool, candidate, args):
    board = np.asarray(candidate["board"], dtype=int)
    analyses = {}
    oracle = pool.oracle()
    for label, nodes in (
        ("shallow", args.shallow_nodes_per_move),
        ("deep", args.deep_nodes_per_move),
    ):
        started = time.perf_counter()
        oracle.reset()
        response = oracle.analyze_root_moves(board, nodes_per_move=nodes, top_k=args.top_k)
        response["elapsed_seconds"] = float(time.perf_counter() - started)
        analyses[label] = response
    confidence = confidence_metrics(
        analyses["shallow"], analyses["deep"], args.score_temperature
    )
    record = dict(candidate)
    record.update({"type": "candidate", "analyses": analyses, "confidence": confidence})
    return record


def top_move_actions(game, record):
    board = np.asarray(record["board"], dtype=int)
    move = record["analyses"]["deep"]["moves"][0]
    return set(external_actions_to_v3_actions(game, board, move["actions"]))


def select_phase_balanced(records, limit):
    """Select high-margin loss-first corrections while keeping phases balanced."""
    limit = min(int(limit), len(records))
    pools = {stage: [] for stage in STAGES}
    for record in records:
        pools[record["stage"]].append(record)
    for stage in STAGES:
        pools[stage].sort(key=lambda record: (
            float(record["value"]),
            -int(record["confidence"]["deep_score_margin"]),
            int(record["candidate_id"]),
        ))

    selected = []
    quota = limit // len(STAGES)
    for stage in STAGES:
        selected.extend(pools[stage][:quota])
        pools[stage] = pools[stage][quota:]
    remaining = limit - len(selected)
    overflow = sorted(
        (record for stage in STAGES for record in pools[stage]),
        key=lambda record: (
            float(record["value"]),
            -int(record["confidence"]["deep_score_margin"]),
            int(record["candidate_id"]),
        ),
    )
    selected.extend(overflow[:remaining])
    return sorted(selected, key=lambda record: int(record["candidate_id"]))


def materialize(game, selected, args):
    examples = []
    for record in selected:
        board = np.asarray(record["board"], dtype=int)
        source = dense_policy(game, record)
        oracle = ranked_moves_to_v3_policy(
            game, board, record["analyses"]["deep"]["moves"], args.score_temperature
        )
        policy = blend_policies(source, oracle, args.oracle_weight)
        valids = game.getValidMoves(board, 1).astype(bool)
        if np.any(policy[~valids] > 1e-7):
            raise ValueError("Adversarial target assigns mass to an illegal action.")
        symmetries = (
            game.getSymmetries(board, policy)
            if not args.no_symmetries else [(board, policy)]
        )
        examples.extend((sym_board, sym_policy, float(record["value"])) for sym_board, sym_policy in symmetries)

    if not examples:
        raise ValueError("No adversarial candidates passed the correction filters.")
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary_path = output_path + ".tmp"
    try:
        save_compact_replay(temporary_path, [deque(examples)])
        loaded = load_compact_replay(temporary_path)
        if len(loaded) != 1 or len(loaded[0]) != len(examples):
            raise ValueError("Adversarial replay failed round-trip validation.")
        os.replace(temporary_path, output_path)
    finally:
        try:
            os.remove(temporary_path)
        except FileNotFoundError:
            pass
    return len(examples)


def write_json_atomic(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
    os.replace(temporary, path)


def validate_args(args):
    positive = (
        args.games, args.sims, args.oracle_game_nodes, args.shallow_nodes_per_move,
        args.deep_nodes_per_move, args.top_k, args.corrections, args.workers,
        args.root_symmetry_samples,
    )
    if any(int(value) < 1 for value in positive):
        raise ValueError("Game, search, correction, and worker counts must be positive.")
    if args.games % 2:
        raise ValueError("--games must be even so every opening is played from both seats.")
    if args.deep_nodes_per_move <= args.shallow_nodes_per_move:
        raise ValueError("Deep nodes per move must exceed shallow nodes per move.")
    if args.score_temperature <= 0 or args.min_score_margin < 0:
        raise ValueError("Temperature must be positive and margin non-negative.")
    if not 0 <= args.oracle_weight <= 1 or not 0 <= args.min_top3_jaccard <= 1:
        raise ValueError("Oracle weight and minimum Jaccard must be between zero and one.")
    if not 1 <= args.root_symmetry_samples <= 8:
        raise ValueError("Root symmetry samples must be between one and eight.")


def main():
    args = parse_args()
    validate_args(args)
    checkpoint_path = os.path.join(args.checkpoint_folder, args.checkpoint_file)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))
    checkpoint_digest = file_sha256(checkpoint_path)
    trajectories_path = args.trajectories_out or companion_path(args.output, ".trajectories.jsonl")
    records_path = args.records_out or companion_path(args.output, ".records.jsonl")
    metadata_path = args.metadata_out or companion_path(args.output, ".metadata.json")

    trajectory_meta = trajectory_metadata(
        args, checkpoint_path, checkpoint_digest, trajectories_path
    )
    games = load_jsonl(
        trajectories_path, trajectory_meta, TRAJECTORY_IDENTITY, "game_id"
    )
    completed_games = {int(record["game_id"]) for record in games}
    print("Adversarial games: resuming with {} of {} complete.".format(len(games), args.games))

    game = SantoriniGame(5, sequential_placement=True)
    opening_rng = np.random.RandomState(args.seed)
    openings = SantoriniRandomOpeningSampler(
        random_orientation=True, rng=opening_rng
    ).sample_distinct_arena_suite(args.games // 2)
    neural = NeuralMCTSPlayer(
        game, args.checkpoint_folder, args.checkpoint_file, args.sims,
        architecture=args.architecture, action_temp=0.0,
        search_mode=args.search_mode,
        gumbel_max_considered_actions=args.gumbel_max_considered_actions,
        gumbel_scale=args.gumbel_scale,
        gumbel_placement_scale=args.gumbel_scale,
        search_symmetry_evaluation=args.architecture == "v3",
        root_symmetry_samples=args.root_symmetry_samples,
        placement_root_symmetry_samples=args.root_symmetry_samples,
        inference_deduplication=args.architecture == "v3",
    )
    gameplay_started = time.perf_counter()
    with SantoriniOracleProcess(args.oracle_binary) as oracle_process:
        oracle_info = dict(oracle_process.info)
        oracle = SantoriniOraclePlayer(game, oracle_process, nodes=args.oracle_game_nodes)
        for game_id in tqdm(range(args.games), desc="Adversarial games"):
            if game_id in completed_games:
                continue
            # Make each game reproducible on its own so resuming does not change
            # MCTS symmetry sampling for all subsequent games.
            np.random.seed(int(args.seed) + int(game_id))
            opening_id = game_id % (args.games // 2)
            neural_side = 1 if game_id < args.games // 2 else -1
            record = play_adversarial_game(
                game, neural, oracle, openings[opening_id], neural_side, game_id
            )
            record["opening_id"] = int(opening_id)
            append_jsonl(trajectories_path, record)
            games.append(record)
    gameplay_wall_seconds = time.perf_counter() - gameplay_started

    games.sort(key=lambda record: int(record["game_id"]))
    candidates, trajectory_observations = collect_candidates(games)
    trajectories_digest = file_sha256(trajectories_path)
    label_meta = label_metadata(
        args, checkpoint_digest, trajectories_digest, candidates, records_path
    )
    records = load_jsonl(records_path, label_meta, LABEL_IDENTITY, "candidate_id")
    completed_candidates = {int(record["candidate_id"]) for record in records}
    pending = [
        candidate for candidate in candidates
        if int(candidate["candidate_id"]) not in completed_candidates
    ]
    print(
        "Captured {} neural decisions ({} D4-unique); labeling {} remaining.".format(
            trajectory_observations, len(candidates), len(pending)
        )
    )

    labeling_started = time.perf_counter()
    pool = ParallelOraclePool(args.oracle_binary)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(analyze_candidate, pool, candidate, args) for candidate in pending]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Oracle corrections"
            ):
                record = future.result()
                append_jsonl(records_path, record)
                records.append(record)
    finally:
        pool.close()
    labeling_wall_seconds = time.perf_counter() - labeling_started

    records.sort(key=lambda record: int(record["candidate_id"]))
    stable = [
        record for record in records
        if record["confidence"]["top1_agreement"]
        and record["confidence"]["top3_jaccard"] >= float(args.min_top3_jaccard)
    ]
    high_margin = [
        record for record in stable
        if record["confidence"]["deep_score_margin"] is not None
        and int(record["confidence"]["deep_score_margin"]) >= int(args.min_score_margin)
    ]
    eligible = [
        record for record in high_margin
        if int(record["chosen_action"]) not in top_move_actions(game, record)
    ]
    selected = select_phase_balanced(eligible, args.corrections)
    augmented_examples = materialize(game, selected, args)
    result = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "checkpoint_sha256": checkpoint_digest,
        "output_path": os.path.abspath(args.output),
        "trajectories_path": os.path.abspath(trajectories_path),
        "records_path": os.path.abspath(records_path),
        "games": len(games),
        "neural_wins": sum(float(record["neural_result"]) == 1 for record in games),
        "oracle_wins": sum(float(record["neural_result"]) == -1 for record in games),
        "draws": sum(float(record["neural_result"]) not in (-1, 1) for record in games),
        "trajectory_observations": trajectory_observations,
        "unique_candidates": len(candidates),
        "stable_candidates": len(stable),
        "high_margin_candidates": len(high_margin),
        "eligible_corrections": len(eligible),
        "eligible_by_stage": {
            stage: sum(record["stage"] == stage for record in eligible) for stage in STAGES
        },
        "selected_corrections": len(selected),
        "selected_by_stage": {
            stage: sum(record["stage"] == stage for record in selected) for stage in STAGES
        },
        "selected_losses": sum(float(record["value"]) < 0 for record in selected),
        "augmented_examples": augmented_examples,
        "sims": int(args.sims),
        "oracle_game_nodes": int(args.oracle_game_nodes),
        "oracle": oracle_info,
        "shallow_nodes_per_move": int(args.shallow_nodes_per_move),
        "deep_nodes_per_move": int(args.deep_nodes_per_move),
        "top_k": int(args.top_k),
        "score_temperature": float(args.score_temperature),
        "oracle_weight": float(args.oracle_weight),
        "min_top3_jaccard": float(args.min_top3_jaccard),
        "min_score_margin": int(args.min_score_margin),
        "workers": int(args.workers),
        "seed": int(args.seed),
        "gameplay_wall_seconds": float(gameplay_wall_seconds),
        "labeling_wall_seconds": float(labeling_wall_seconds),
    }
    write_json_atomic(metadata_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print("Adversarial replay: {}".format(os.path.abspath(args.output)))
    print("Metadata: {}".format(os.path.abspath(metadata_path)))


if __name__ == "__main__":
    main()
