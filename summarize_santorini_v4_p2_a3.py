"""Validate and combine the P2 A3 external-oracle milestone anchors."""

import argparse
import json
import os

import numpy as np

from santorini.OracleResearch import file_sha256


SHARED_CONTRACT_KEYS = (
    "oracle_binary_sha256",
    "games_per_budget",
    "simulations",
    "search_mode",
    "gumbel_scale",
    "action_temperature",
    "root_symmetry_samples",
    "canonical_d4",
    "inference_cache_size",
    "fp16",
    "opening_seed",
    "opening_sha256",
    "bootstrap_seed",
    "bootstrap_samples",
    "oracle_reset_policy",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1c", required=True)
    parser.add_argument("--iteration11-20k", required=True)
    parser.add_argument("--iteration11-100k", required=True)
    parser.add_argument("--iteration14", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260923)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _load(path):
    with open(path) as source:
        payload = json.load(source)
    if payload.get("type") != "santorini_v4_p2_oracle_sweep_summary":
        raise ValueError("Not a P2 oracle sweep summary: {}".format(path))
    if payload.get("final_test_touched", True) or payload.get(
        "final_arena_seeds_touched", True
    ):
        raise ValueError("Oracle input touched reserved evaluation data: {}".format(path))
    return payload


def _rows_by_budget(payload):
    return {int(row["oracle_nodes"]): row for row in payload["results"]}


def paired_delta(candidate, reference, samples, seed):
    candidate_records = candidate["pair_records"]
    reference_records = reference["pair_records"]
    if [row["opening_sha256"] for row in candidate_records] != [
        row["opening_sha256"] for row in reference_records
    ]:
        raise ValueError("Paired oracle results use different opening orders.")
    differences = np.asarray([
        float(left["v4_pair_score"]) - float(right["v4_pair_score"])
        for left, right in zip(candidate_records, reference_records)
    ]) / 2.0
    rng = np.random.RandomState(seed)
    draws = differences[
        rng.randint(len(differences), size=(int(samples), len(differences)))
    ].mean(axis=1)
    return {
        "score_delta": float(np.mean(differences)),
        "cluster_bootstrap_95_low": float(np.quantile(draws, 0.025)),
        "cluster_bootstrap_95_high": float(np.quantile(draws, 0.975)),
        "improved_pairs": int(np.sum(differences > 0)),
        "unchanged_pairs": int(np.sum(differences == 0)),
        "worsened_pairs": int(np.sum(differences < 0)),
        "pair_score_differences": differences.tolist(),
    }


def build_summary(paths, samples=10_000, seed=20260923):
    payloads = {name: _load(path) for name, path in paths.items()}
    reference = payloads["p1c"]["contract"]
    for name, payload in payloads.items():
        for key in SHARED_CONTRACT_KEYS:
            if payload["contract"].get(key) != reference.get(key):
                raise ValueError("{} changes shared contract key {}.".format(name, key))

    rows = {
        "p1c": _rows_by_budget(payloads["p1c"]),
        "iteration11": {
            **_rows_by_budget(payloads["iteration11_20k"]),
            **_rows_by_budget(payloads["iteration11_100k"]),
        },
        "iteration14": _rows_by_budget(payloads["iteration14"]),
    }
    required_budgets = (20_000, 100_000)
    for name, checkpoint_rows in rows.items():
        if not set(required_budgets).issubset(checkpoint_rows):
            raise ValueError("{} lacks a required A3 budget.".format(name))

    score_table = {}
    comparisons = {}
    offset = 0
    for budget in required_budgets:
        score_table[str(budget)] = {}
        for name in ("p1c", "iteration11", "iteration14"):
            row = rows[name][budget]
            paired = row["paired_statistics"]
            score_table[str(budget)][name] = {
                "v4_score": float(row["v4_score"]),
                "paired_bootstrap_95": [
                    float(paired["cluster_bootstrap_95_low"]),
                    float(paired["cluster_bootstrap_95_high"]),
                ],
                "pair_wins_2_0": int(paired["v4_pair_wins_2_0"]),
                "pair_splits_1_1": int(paired["split_pairs_1_1"]),
                "pair_losses_0_2": int(paired["v4_pair_losses_0_2"]),
            }
        comparisons[str(budget)] = {}
        for candidate, anchor in (
            ("iteration11", "p1c"),
            ("iteration14", "p1c"),
            ("iteration14", "iteration11"),
        ):
            key = "{}_vs_{}".format(candidate, anchor)
            comparisons[str(budget)][key] = paired_delta(
                rows[candidate][budget], rows[anchor][budget], samples, seed + offset
            )
            offset += 1

    return {
        "schema_version": 1,
        "type": "santorini_v4_p2_a3_external_anchor_summary",
        "shared_contract": {key: reference[key] for key in SHARED_CONTRACT_KEYS},
        "inputs": {
            name: {"path": os.path.abspath(path), "sha256": file_sha256(path)}
            for name, path in paths.items()
        },
        "checkpoint_sha256": {
            "p1c": payloads["p1c"]["contract"]["checkpoint_sha256"],
            "iteration11": payloads["iteration11_20k"]["contract"]["checkpoint_sha256"],
            "iteration14": payloads["iteration14"]["contract"]["checkpoint_sha256"],
        },
        "score_table": score_table,
        "paired_checkpoint_deltas": comparisons,
        "bootstrap": {"samples": int(samples), "seed": int(seed)},
        "final_test_touched": False,
        "final_arena_seeds_touched": False,
        "interpretation": [
            "All checkpoint deltas are paired on identical openings, seats, and oracle budgets.",
            "Iteration 14 improves the 20k point estimate but does not improve on P1c at 100k.",
            "No paired checkpoint-delta interval excludes zero; A3 demonstrates no transferable gain.",
        ],
    }


def _atomic_json(path, payload):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
    os.replace(temporary, path)


def main():
    args = parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("Bootstrap sample count must be positive.")
    paths = {
        "p1c": args.p1c,
        "iteration11_20k": args.iteration11_20k,
        "iteration11_100k": args.iteration11_100k,
        "iteration14": args.iteration14,
    }
    summary = build_summary(paths, args.bootstrap_samples, args.bootstrap_seed)
    _atomic_json(args.output, summary)
    print(json.dumps({
        "output": os.path.abspath(args.output),
        "score_table": summary["score_table"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
