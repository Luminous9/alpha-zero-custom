"""Apply the frozen decision rule to V4 value-target selection outputs."""

import argparse
import json
import os

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--standard-arena", required=True)
    parser.add_argument("--full-arena", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260821)
    return parser.parse_args()


def _load(path):
    with open(path) as source:
        return json.load(source)


def _records_by_seed(arena):
    if arena.get("player1", {}).get("name") != "global_blend":
        raise ValueError("Arena player1 must be named global_blend.")
    if arena.get("player2", {}).get("name") != "winner_only":
        raise ValueError("Arena player2 must be named winner_only.")
    records = arena["paired_statistics"]["records"]
    result = {
        int(record["game_seed"]): float(record["contestant1_score"])
        for record in records
    }
    if len(records) != arena["games"] // 2 or len(result) != len(records):
        raise ValueError("Arena has the wrong number of unique seed clusters.")
    return result


def _write(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
    os.replace(temporary, path)


def summarize(comparison, standard, full, bootstrap_samples=10_000, seed=20260821):
    if bootstrap_samples < 1:
        raise ValueError("Bootstrap samples must be positive.")
    if standard.get("gate") != "standard" or full.get("gate") != "full":
        raise ValueError("Expected one standard and one full arena.")
    for arena in (standard, full):
        if arena.get("selection_seed") != 20260814 or arena.get("games") != 40:
            raise ValueError("Arena does not match the frozen seed/game contract.")
        if arena.get("simulations") != 96 or arena.get("fp16"):
            raise ValueError("Arena does not match the frozen search/precision contract.")
        if arena.get("search_mode") != "gumbel" or arena.get("gumbel_scale") != 0.0:
            raise ValueError("Arena does not match the frozen search-mode contract.")
        for player_name in ("player1", "player2"):
            player = arena.get(player_name, {})
            if not player.get("canonical_d4") or player.get("root_symmetries") != 1:
                raise ValueError("Arena does not use the frozen exact-D4 contract.")
        if arena.get("final_test_touched") or arena.get("final_arena_seeds_touched"):
            raise ValueError("A reserved final input was touched.")
    standard_scores = _records_by_seed(standard)
    full_scores = _records_by_seed(full)
    if set(standard_scores) != set(full_scores):
        raise ValueError("Standard and full arenas do not share seed clusters.")
    seeds = sorted(standard_scores)
    # Each gate contributes two seat-swapped games, so each combined seed score
    # ranges from zero to four and is one bootstrap cluster.
    cluster_scores = np.asarray([
        standard_scores[seed] + full_scores[seed] for seed in seeds
    ], dtype=np.float64)
    rng = np.random.RandomState(seed)
    bootstrap = cluster_scores[
        rng.randint(len(cluster_scores), size=(bootstrap_samples, len(cluster_scores)))
    ].mean(axis=1) / 4.0
    interval = list(map(float, np.quantile(bootstrap, (0.025, 0.975))))
    global_score = float(np.mean(cluster_scores) / 4.0)
    global_clear_arena_win = interval[0] > 0.5
    supervised_noninferior = bool(comparison["supervised_noninferior"])
    select_winner = supervised_noninferior and not global_clear_arena_win
    return {
        "schema_version": 1,
        "type": "santorini_v4_1m_value_target_decision",
        "decision_contract": comparison["decision_contract"],
        "supervised_noninferior": supervised_noninferior,
        "combined_arena": {
            "perspective": "global_blend",
            "seed_clusters": len(cluster_scores),
            "games": int(standard["games"] + full["games"]),
            "score": global_score,
            "cluster_bootstrap_95_interval": interval,
            "global_blend_clear_win": bool(global_clear_arena_win),
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": seed,
            "cluster_records": [
                {
                    "game_seed": seed,
                    "global_blend_score_across_four_games": float(score),
                }
                for seed, score in zip(seeds, cluster_scores)
            ],
        },
        "selected_target": "winner" if select_winner else "global_blend",
        "reason": (
            "winner-only is supervised-noninferior and global blend has no clear arena win"
            if select_winner else
            "winner-only failed supervised noninferiority or global blend won the arena clearly"
        ),
        "optional_stopping": False,
        "final_test_touched": False,
        "final_arena_seeds_touched": False,
    }


def main():
    args = parse_args()
    result = summarize(
        _load(args.comparison),
        _load(args.standard_arena),
        _load(args.full_arena),
        args.bootstrap_samples,
        args.seed,
    )
    _write(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
