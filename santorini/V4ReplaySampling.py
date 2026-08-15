"""Phase-aware replay preparation for Santorini V4 self-play.

Raw replay remains an append-only record of searched positions.  This module
builds a non-destructive training view which prevents the four mandatory
placement prefixes from receiving weight merely because every game traverses
them.
"""

from collections import defaultdict

import numpy as np

from .D4Canonical import canonicalize_board_policies


def placement_ply(game, board):
    """Return the zero-based placement ply, or ``None`` in standard play."""
    if not (
        hasattr(game, "isPlacementPhase") and game.isPlacementPhase(board)
    ):
        return None
    return int(np.count_nonzero(np.asarray(board)[0]))


def _aggregate_window_placements(game, examples, window_index):
    by_ply = defaultdict(list)
    standard = []
    for example in examples:
        ply = placement_ply(game, example[0])
        if ply is None:
            standard.append(example)
        else:
            by_ply[ply].append(example)

    groups = []
    for ply, ply_examples in sorted(by_ply.items()):
        boards = np.asarray([example[0] for example in ply_examples])
        policies = np.asarray([example[1] for example in ply_examples])
        canonical_boards, canonical_policies, keys = canonicalize_board_policies(
            game, boards, policies
        )
        grouped_indices = defaultdict(list)
        for index, key in enumerate(keys):
            grouped_indices[key].append(index)

        for key, indices in grouped_indices.items():
            mean_policy = np.mean(canonical_policies[indices], axis=0)
            policy_mass = float(mean_policy.sum())
            if policy_mass <= 0:
                raise ValueError("Aggregated placement policy has no probability mass.")
            mean_policy = (mean_policy / policy_mass).astype(np.float32)
            mean_value = float(np.mean([
                float(ply_examples[index][2]) for index in indices
            ]))
            source_counts = defaultdict(int)
            for index in indices:
                example = ply_examples[index]
                metadata = example[3] if len(example) >= 4 else None
                source_counts[(metadata or {}).get("source", "self_play")] += 1
            metadata = {
                "source": "placement_replay_aggregate",
                "source_counts": dict(sorted(source_counts.items())),
                "source_window": int(window_index),
                "placement_ply": int(ply),
                "occurrences": int(len(indices)),
            }
            groups.append({
                "example": (
                    canonical_boards[indices[0]].astype(int),
                    mean_policy,
                    mean_value,
                    metadata,
                ),
                "window": int(window_index),
                "ply": int(ply),
                "count": int(len(indices)),
                "key": key,
            })
    return standard, groups


def prepare_v4_replay(
    game,
    training_history,
    placement_fraction=0.15,
    frequency_exponent=0.5,
):
    """Return aggregated examples, normalized draw weights, and diagnostics.

    Sampling is hierarchical:

    * standard positions receive ``1 - placement_fraction`` total mass;
    * the placement mass is divided equally among the available placement plies;
    * a ply's mass is divided among source windows in proportion to the number
      of raw examples that window contributed at that ply;
    * D4-identical states within a window/ply are represented once, with their
      policy and value targets averaged, and receive weight ``count ** exponent``.

    ``frequency_exponent=1`` exactly retains raw duplicate weighting within a
    stratum.  The production value ``0.5`` deliberately softens it.
    """
    placement_fraction = float(placement_fraction)
    frequency_exponent = float(frequency_exponent)
    if not 0.0 <= placement_fraction < 1.0:
        raise ValueError("placement_fraction must be in [0, 1).")
    if not 0.0 <= frequency_exponent <= 1.0:
        raise ValueError("frequency_exponent must be in [0, 1].")

    standard = []
    groups = []
    raw_window_ply = defaultdict(int)
    raw_total = 0
    for window_index, window in enumerate(training_history):
        window_examples = list(window)
        raw_total += len(window_examples)
        window_standard, window_groups = _aggregate_window_placements(
            game, window_examples, window_index
        )
        standard.extend(window_standard)
        groups.extend(window_groups)
        for group in window_groups:
            raw_window_ply[(group["window"], group["ply"])] += group["count"]

    if not standard:
        raise ValueError("Phase-balanced replay requires at least one standard position.")

    examples = list(standard) + [group["example"] for group in groups]
    weights = np.zeros(len(examples), dtype=np.float64)
    effective_placement_fraction = placement_fraction if groups else 0.0
    weights[:len(standard)] = (
        (1.0 - effective_placement_fraction) / len(standard)
    )

    available_plies = sorted({group["ply"] for group in groups})
    group_offset = len(standard)
    if available_plies:
        ply_mass = effective_placement_fraction / len(available_plies)
        for ply in available_plies:
            ply_raw = sum(
                count for (window, item_ply), count in raw_window_ply.items()
                if item_ply == ply
            )
            ply_groups = [
                (index, group) for index, group in enumerate(groups)
                if group["ply"] == ply
            ]
            for window in sorted({group["window"] for _, group in ply_groups}):
                stratum = [
                    (index, group) for index, group in ply_groups
                    if group["window"] == window
                ]
                stratum_mass = ply_mass * raw_window_ply[(window, ply)] / ply_raw
                relative = np.asarray([
                    group["count"] ** frequency_exponent for _, group in stratum
                ], dtype=np.float64)
                relative /= relative.sum()
                for (index, _), probability in zip(stratum, relative):
                    weights[group_offset + index] = stratum_mass * probability

    weights /= weights.sum()
    raw_placement = sum(group["count"] for group in groups)
    metrics = {
        "replay_sampling_mode": "phase-balanced-v1",
        "replay_raw_examples": int(raw_total),
        "replay_raw_standard_examples": int(len(standard)),
        "replay_raw_placement_examples": int(raw_placement),
        "replay_raw_placement_fraction": (
            float(raw_placement / raw_total) if raw_total else None
        ),
        "replay_training_view_examples": int(len(examples)),
        "replay_aggregated_placement_groups": int(len(groups)),
        "replay_placement_sampling_fraction": float(
            weights[len(standard):].sum()
        ),
        "replay_placement_frequency_exponent": float(frequency_exponent),
        "replay_sampling_effective_examples": float(1.0 / np.sum(weights ** 2)),
        "replay_placement_max_group_occurrences": int(
            max((group["count"] for group in groups), default=0)
        ),
    }
    for ply in available_plies:
        selected = [
            index for index, group in enumerate(groups) if group["ply"] == ply
        ]
        metrics.update({
            "replay_placement_ply_{}_raw_examples".format(ply): int(sum(
                groups[index]["count"] for index in selected
            )),
            "replay_placement_ply_{}_groups".format(ply): int(len(selected)),
            "replay_placement_ply_{}_sampling_fraction".format(ply): float(
                weights[group_offset + np.asarray(selected, dtype=np.int64)].sum()
            ),
        })
    return examples, weights, metrics
