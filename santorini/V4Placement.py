"""Shared exhaustive placement coverage and joint-teacher factorization."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .D4Canonical import canonicalize_board, canonicalize_board_policy
from .SantoriniPlayers import parse_coordinate
from .V4Supervised import score_to_value


EXPECTED_ORBITS_BY_WORKER_COUNT = (1, 6, 49, 904)


def enumerate_placement_orbits(game):
    """Return every reachable player-to-move placement prefix, once per D4 orbit."""
    levels = [[game.getInitBoard()]]
    for _ in range(3):
        next_level = {}
        for board in levels[-1]:
            for action in np.flatnonzero(game.getValidMoves(board, 1)):
                next_board, next_player = game.getNextState(board, 1, int(action))
                player_view = game.getCanonicalForm(next_board, next_player)
                representative, _, key = canonicalize_board(player_view)
                next_level[key] = representative
        levels.append([next_level[key] for key in sorted(next_level)])
    counts = tuple(len(level) for level in levels)
    if counts != EXPECTED_ORBITS_BY_WORKER_COUNT:
        raise AssertionError(
            "Reachable placement-orbit count changed: {} != {}.".format(
                counts, EXPECTED_ORBITS_BY_WORKER_COUNT
            )
        )
    return [board for level in levels for board in level]


def joint_boundary_orbits(game):
    """Return the 1+49 V4 states that santorini-ai can search directly."""
    return [
        board for board in enumerate_placement_orbits(game)
        if int(np.count_nonzero(board[0])) in (0, 2)
    ]


def legal_unordered_pairs(game, board):
    locations = [
        divmod(int(action) // game.local_action_size, game.n)
        for action in np.flatnonzero(game.getValidMoves(board, 1))
    ]
    return tuple(combinations(sorted(locations), 2))


def _normalized_pairs(pairs):
    normalized = []
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError("Every joint placement must contain two squares.")
        locations = tuple(sorted(tuple(map(int, location)) for location in pair))
        if locations[0] == locations[1]:
            raise ValueError("A joint placement cannot reuse one square.")
        normalized.append(locations)
    if len(normalized) != len(set(normalized)):
        raise ValueError("Joint placement pairs must be unique.")
    return tuple(normalized)


def pair_softmax(scores, temperature):
    scores = np.asarray(scores, dtype=np.float64)
    temperature = float(temperature)
    if scores.ndim != 1 or not len(scores) or not np.all(np.isfinite(scores)):
        raise ValueError("Joint placement scores must be a finite non-empty vector.")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Joint placement policy temperature must be positive.")
    logits = (scores - np.max(scores)) / temperature
    weights = np.exp(np.clip(logits, -700.0, 0.0))
    return weights / weights.sum()


def symmetrize_joint_pair_scores(game, boundary_board, pairs, scores):
    """Average scores for pair moves equivalent under the parent's D4 stabilizer."""
    pairs = _normalized_pairs(pairs)
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) != len(pairs):
        raise ValueError("Every joint placement pair requires one score.")
    groups = {}
    for index, pair in enumerate(pairs):
        board, player = game.getNextState(
            boundary_board, 1, game.getPlacementAction(pair[0])
        )
        board, player = game.getNextState(
            board, player, game.getPlacementAction(pair[1])
        )
        child = game.getCanonicalForm(board, player)
        _, _, key = canonicalize_board(child)
        groups.setdefault(key, []).append(index)
    projected = scores.copy()
    ranges = []
    for indices in groups.values():
        values = scores[indices]
        projected[indices] = np.mean(values)
        ranges.append(float(np.max(values) - np.min(values)))
    diagnostics = {
        "pair_orbits": len(groups),
        "mean_raw_orbit_score_range": float(np.mean(ranges)),
        "max_raw_orbit_score_range": float(np.max(ranges)),
        "nonzero_raw_orbit_score_ranges": int(np.sum(np.asarray(ranges) > 0)),
    }
    return projected, diagnostics


@dataclass
class PlacementTeacherObservation:
    board: np.ndarray
    policy: np.ndarray
    score: float
    value: float
    reach_weight: float
    pair_support: int


def factor_joint_placement(
    game,
    boundary_board,
    pairs,
    scores,
    policy_temperature,
    value_temperature=261.8,
    symmetrize_scores=True,
):
    """Factor a complete unordered-pair distribution into two V4 decisions."""
    boundary_board = np.asarray(boundary_board).astype(int, copy=True)
    worker_count = int(np.count_nonzero(boundary_board[0]))
    if worker_count not in (0, 2) or np.any(boundary_board[0] > 0):
        raise ValueError("Joint boundary must have zero current-player workers.")
    pairs = _normalized_pairs(pairs)
    expected_pairs = set(legal_unordered_pairs(game, boundary_board))
    if set(pairs) != expected_pairs:
        raise ValueError(
            "Joint teacher returned {} of {} legal unordered pairs.".format(
                len(pairs), len(expected_pairs)
            )
        )
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) != len(pairs):
        raise ValueError("Every joint placement pair requires one score.")
    if symmetrize_scores:
        scores, _ = symmetrize_joint_pair_scores(
            game, boundary_board, pairs, scores
        )
    pair_probabilities = pair_softmax(scores, policy_temperature)
    pair_values = np.asarray([
        score_to_value(score, value_temperature) for score in scores
    ], dtype=np.float64)

    first_policy = np.zeros(game.getActionSize(), dtype=np.float64)
    first_score_numerator = {}
    first_value_numerator = {}
    first_pair_mass = {}
    pair_by_first = {}
    for pair, score, value, probability in zip(
        pairs, scores, pair_values, pair_probabilities
    ):
        for first in pair:
            action = game.getPlacementAction(first)
            first_policy[action] += 0.5 * probability
            first_pair_mass[first] = first_pair_mass.get(first, 0.0) + probability
            first_score_numerator[first] = (
                first_score_numerator.get(first, 0.0) + probability * score
            )
            first_value_numerator[first] = (
                first_value_numerator.get(first, 0.0) + probability * value
            )
            pair_by_first.setdefault(first, []).append((pair, probability))

    observations = [PlacementTeacherObservation(
        board=boundary_board,
        policy=first_policy,
        score=float(np.dot(pair_probabilities, scores)),
        value=float(np.dot(pair_probabilities, pair_values)),
        reach_weight=1.0,
        pair_support=len(pairs),
    )]
    for first in sorted(pair_by_first):
        mass = first_pair_mass[first]
        conditional_policy = np.zeros(game.getActionSize(), dtype=np.float64)
        for pair, probability in pair_by_first[first]:
            second = pair[1] if pair[0] == first else pair[0]
            conditional_policy[game.getPlacementAction(second)] += probability / mass
        partial_board, next_player = game.getNextState(
            boundary_board, 1, game.getPlacementAction(first)
        )
        if next_player != 1:
            raise AssertionError("The first worker placement must retain the same player.")
        observations.append(PlacementTeacherObservation(
            board=partial_board,
            policy=conditional_policy,
            score=float(first_score_numerator[first] / mass),
            value=float(first_value_numerator[first] / mass),
            reach_weight=float(0.5 * mass),
            pair_support=len(pair_by_first[first]),
        ))
    if not np.isclose(first_policy.sum(), 1.0, atol=1e-9):
        raise AssertionError("Factored first-placement policy does not sum to one.")
    return observations


def aggregate_teacher_observations(game, observations):
    """D4-project and reach-weight observations into unique V4 positions."""
    aggregates = {}
    for observation in observations:
        board, policy, key = canonicalize_board_policy(
            game, observation.board, observation.policy
        )
        weight = float(observation.reach_weight)
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("Placement observation reach weights must be positive.")
        record = aggregates.get(key)
        if record is None:
            record = {
                "board": board.astype(np.int8),
                "policy_sum": np.zeros(game.getActionSize(), dtype=np.float64),
                "score_sum": 0.0,
                "value_sum": 0.0,
                "reach_weight": 0.0,
                "observation_count": 0,
                "pair_support_sum": 0.0,
            }
            aggregates[key] = record
        record["policy_sum"] += weight * policy
        record["score_sum"] += weight * float(observation.score)
        record["value_sum"] += weight * float(observation.value)
        record["reach_weight"] += weight
        record["observation_count"] += 1
        record["pair_support_sum"] += weight * int(observation.pair_support)
    return aggregates
