"""Summarize the frozen P1c Gate G1 arenas and apply declared thresholds."""

import argparse
import json
import os

import numpy as np


STOP_SCORE = 0.20
GREEN_SCORE = 0.35
MATERIAL_PHASE_GAP = 0.15


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True)
    for budget in (96, 128):
        parser.add_argument("--equal-{}-standard".format(budget), required=True)
        parser.add_argument("--equal-{}-full".format(budget), required=True)
    parser.add_argument("--equal-cost-standard", required=True)
    parser.add_argument("--equal-cost-full", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _load(path):
    with open(path) as source:
        return json.load(source)


def _validate_arena(payload, gate):
    if payload.get("gate") != gate:
        raise ValueError("Expected a {} arena.".format(gate))
    if payload.get("final_test_touched") or payload.get("final_arena_seeds_touched"):
        raise ValueError("A G1 arena touched reserved evaluation data.")
    if payload.get("player1", {}).get("name") != "p1c":
        raise ValueError("G1 player one must be the P1c candidate.")
    if payload.get("player2", {}).get("name") != "run13":
        raise ValueError("G1 player two must be Run13.")
    if (
        payload["player1"].get("kind") != "v4"
        or payload["player1"].get("canonical_d4") is not True
        or payload["player1"].get("root_symmetries") != 1
        or payload["player1"].get("placement_root_symmetries") != 1
    ):
        raise ValueError("P1c must use canonical D4 inference with 1/1 roots.")
    if (
        payload["player2"].get("kind") != "v3"
        or payload["player2"].get("canonical_d4") is not False
        or payload["player2"].get("root_symmetries") != 8
        or payload["player2"].get("placement_root_symmetries") != 8
    ):
        raise ValueError("Run13 must use its native evaluation 8/8 root settings.")
    expected_temperature = 1.0 if gate == "full" else 0.0
    if payload.get("placement_temperature") != expected_temperature:
        raise ValueError("G1 arena uses the wrong placement sampling temperature.")
    if int(payload.get("games", 0)) < 2:
        raise ValueError("A G1 arena has no paired games.")


def _row(payload):
    paired = payload["paired_statistics"]
    return {
        "candidate_score": float(payload["player1_score"]),
        "cluster_bootstrap_95_low": float(paired["cluster_bootstrap_95_low"]),
        "cluster_bootstrap_95_high": float(paired["cluster_bootstrap_95_high"]),
        "candidate_wins": int(payload["player1"]["wins"]),
        "run13_wins": int(payload["player2"]["wins"]),
        "draws": int(payload["draws"]),
        "pairs": int(paired["pairs"]),
        "pair_wins": int(paired["pair_wins"]),
        "pair_losses": int(paired["pair_losses"]),
        "pair_ties": int(paired["pair_ties"]),
        "candidate_simulations": int(payload["player1_simulations"]),
        "run13_simulations": int(payload["player2_simulations"]),
        "placement_temperature": float(payload["placement_temperature"]),
        "candidate_root_symmetries": {
            "standard": int(payload["player1"]["root_symmetries"]),
            "placement": int(payload["player1"]["placement_root_symmetries"]),
        },
        "run13_root_symmetries": {
            "standard": int(payload["player2"]["root_symmetries"]),
            "placement": int(payload["player2"]["placement_root_symmetries"]),
        },
        "elapsed_seconds": float(payload["elapsed_seconds"]),
        "inference": payload["inference"],
    }


def _pair_scores(payload):
    records = payload["paired_statistics"]["records"]
    return {
        int(record["pair_index"]): float(record["contestant1_score"]) / 2.0
        for record in records
    }


def _phase_gap(standard, full, seed, bootstrap_samples):
    standard_scores = _pair_scores(standard)
    full_scores = _pair_scores(full)
    if set(standard_scores) != set(full_scores):
        raise ValueError("Standard/full G1 arenas do not share paired seed blocks.")
    keys = sorted(standard_scores)
    differences = np.asarray(
        [full_scores[key] - standard_scores[key] for key in keys],
        dtype=np.float64,
    )
    rng = np.random.RandomState(int(seed))
    sampled = differences[
        rng.randint(len(differences), size=(int(bootstrap_samples), len(differences)))
    ].mean(axis=1)
    low = float(np.quantile(sampled, 0.025))
    high = float(np.quantile(sampled, 0.975))
    difference = float(differences.mean())
    material = (
        abs(difference) >= MATERIAL_PHASE_GAP
        and (low > 0.0 or high < 0.0)
    )
    return {
        "full_minus_standard_score": difference,
        "cluster_bootstrap_95_low": low,
        "cluster_bootstrap_95_high": high,
        "absolute_gap_threshold": MATERIAL_PHASE_GAP,
        "material": bool(material),
    }


def _standard_classification(row):
    if row["candidate_score"] < STOP_SCORE:
        return "stop"
    if (
        row["candidate_score"] >= GREEN_SCORE
        and row["cluster_bootstrap_95_low"] > STOP_SCORE
    ):
        return "green"
    return "inconclusive"


def build_summary(calibration, equal_arenas, equal_cost, seed, bootstrap_samples):
    primary = {}
    phase_gaps = {}
    for offset, budget in enumerate((96, 128)):
        standard = equal_arenas[budget]["standard"]
        full = equal_arenas[budget]["full"]
        _validate_arena(standard, "standard")
        _validate_arena(full, "full")
        if any(
            payload["player1_simulations"] != budget
            or payload["player2_simulations"] != budget
            for payload in (standard, full)
        ):
            raise ValueError("A primary G1 arena is not equal-simulation.")
        standard_row = _row(standard)
        full_row = _row(full)
        standard_row["classification"] = _standard_classification(standard_row)
        primary[str(budget)] = {"standard": standard_row, "full": full_row}
        phase_gaps[str(budget)] = _phase_gap(
            standard,
            full,
            int(seed) + offset,
            bootstrap_samples,
        )

    for gate in ("standard", "full"):
        _validate_arena(equal_cost[gate], gate)
        if (
            equal_cost[gate]["player1_simulations"]
            != calibration["equal_cost_simulations"]["p1c"]
            or equal_cost[gate]["player2_simulations"]
            != calibration["equal_cost_simulations"]["run13"]
        ):
            raise ValueError("An equal-cost arena does not match its calibration.")
    equal_cost_rows = {gate: _row(equal_cost[gate]) for gate in ("standard", "full")}

    any_stop_score = any(
        primary[str(budget)][gate]["candidate_score"] < STOP_SCORE
        for budget in (96, 128)
        for gate in ("standard", "full")
    )
    material_gap = any(item["material"] for item in phase_gaps.values())
    standard_green = all(
        primary[str(budget)]["standard"]["classification"] == "green"
        for budget in (96, 128)
    )
    if any_stop_score or material_gap:
        decision = "stop_and_debug"
    elif standard_green:
        decision = "green_light"
    else:
        decision = "inconclusive"

    return {
        "schema_version": 1,
        "contract": "santorini_v4_p1c_gate_g1",
        "decision": decision,
        "decision_reasons": {
            "any_primary_score_below_stop_region": bool(any_stop_score),
            "material_standard_full_discrepancy": bool(material_gap),
            "both_standard_gates_green": bool(standard_green),
        },
        "thresholds": {
            "stop_score": STOP_SCORE,
            "green_standard_score": GREEN_SCORE,
            "green_interval_must_exclude_stop_region": True,
            "material_phase_gap": MATERIAL_PHASE_GAP,
        },
        "equal_simulation": primary,
        "standard_full_phase_gaps": phase_gaps,
        "equal_cost_diagnostic": {
            "budget_rule": calibration["equal_cost_budget_rule"],
            "candidate_simulations": calibration["equal_cost_simulations"]["p1c"],
            "run13_simulations": calibration["equal_cost_simulations"]["run13"],
            "standard": equal_cost_rows["standard"],
            "full": equal_cost_rows["full"],
            "decision_role": "supplementary; does not change frozen G1 thresholds",
        },
        "calibration": calibration,
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(seed),
        "final_test_touched": False,
        "final_arena_seeds_touched": False,
    }


def main():
    args = parse_args()
    calibration = _load(args.calibration)
    equal = {
        budget: {
            gate: _load(getattr(args, "equal_{}_{}".format(budget, gate)))
            for gate in ("standard", "full")
        }
        for budget in (96, 128)
    }
    equal_cost = {
        gate: _load(getattr(args, "equal_cost_{}".format(gate)))
        for gate in ("standard", "full")
    }
    payload = build_summary(
        calibration, equal, equal_cost, args.seed, args.bootstrap_samples
    )
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary = output + ".tmp"
    with open(temporary, "w") as destination:
        json.dump(payload, destination, indent=2, sort_keys=True)
    os.replace(temporary, output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
