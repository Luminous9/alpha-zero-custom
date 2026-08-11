"""Placement-only teacher matchups with a shared standard-play adjudicator."""

from dataclasses import dataclass
import hashlib

import numpy as np

from .D4Canonical import canonicalize_board, restore_canonical_policy
from .OracleResearch import file_sha256
from .SantoriniOracle import external_actions_to_v3_actions
from .V4BootstrapCorpus import decode_sparse_policy
from .V4Placement import EXPECTED_ORBITS_BY_WORKER_COUNT


def _load_npz(path):
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


@dataclass
class PlacementChoice:
    action: int
    probability: float
    position_hash: str


class PlacementPolicyTeacher:
    """Lookup a D4-canonical sparse placement component in any board frame."""

    def __init__(self, game, name, path, require_complete=True):
        self.game = game
        self.name = str(name)
        self.path = str(path)
        self.sha256 = file_sha256(path)
        self.payload = _load_npz(path)
        hashes = list(map(str, self.payload["position_hashes"]))
        if len(hashes) != len(set(hashes)):
            raise ValueError("Placement teacher has duplicate position hashes.")
        self.indices = {key: index for index, key in enumerate(hashes)}
        worker_counts = np.asarray(self.payload["worker_counts"], dtype=np.int64)
        coverage = tuple(
            int(np.sum(worker_counts == count)) for count in range(4)
        )
        if require_complete and coverage != EXPECTED_ORBITS_BY_WORKER_COUNT:
            raise ValueError(
                "Placement teacher coverage is {} instead of {}.".format(
                    coverage, EXPECTED_ORBITS_BY_WORKER_COUNT
                )
            )
        self.coverage = coverage

    def distribution(self, canonical_board):
        representative, matching_transforms, key = canonicalize_board(canonical_board)
        position_hash = hashlib.sha256(key).hexdigest()
        if position_hash not in self.indices:
            raise KeyError(
                "{} has no placement target for {}.".format(
                    self.name, position_hash
                )
            )
        index = self.indices[position_hash]
        if not np.array_equal(representative, self.payload["boards"][index]):
            raise ValueError("Placement hash matched but canonical boards differ.")
        canonical_policy = decode_sparse_policy(self.payload, index).astype(np.float64)
        policy = restore_canonical_policy(
            self.game, canonical_policy, matching_transforms
        ).astype(np.float64)
        valids = self.game.getValidMoves(canonical_board, 1).astype(bool)
        if np.any(policy[~valids] > 1e-7):
            raise ValueError("Placement teacher assigns mass to an illegal action.")
        policy[~valids] = 0.0
        total = float(policy.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Placement teacher has no finite legal policy mass.")
        return policy / total, position_hash

    def choose(self, canonical_board, mode, rng=None):
        policy, position_hash = self.distribution(canonical_board)
        if mode == "greedy":
            best = np.flatnonzero(np.isclose(policy, np.max(policy), atol=1e-12))
            action = int(best[0])
        elif mode == "sampled":
            if rng is None:
                raise ValueError("Sampled placement requires an RNG.")
            action = int(rng.choice(len(policy), p=policy))
        else:
            raise ValueError("Unknown placement selection mode: {}".format(mode))
        return PlacementChoice(action, float(policy[action]), position_hash)


def build_completed_opening(game, player_one_teacher, player_two_teacher, mode, seed):
    """Let two component teachers place their workers and return the opening."""
    board = game.getInitBoard()
    current_player = 1
    rng = np.random.RandomState(int(seed))
    trace = []
    while game.isPlacementPhase(board):
        teacher = player_one_teacher if current_player == 1 else player_two_teacher
        canonical = game.getCanonicalForm(board, current_player)
        choice = teacher.choose(canonical, mode, rng)
        valids = game.getValidMoves(canonical, 1)
        if not valids[choice.action]:
            raise AssertionError("Placement teacher selected an illegal action.")
        trace.append({
            "player": int(current_player),
            "teacher": teacher.name,
            "action": int(choice.action),
            "probability": float(choice.probability),
            "position_hash": choice.position_hash,
        })
        board, current_player = game.getNextState(
            board, current_player, choice.action
        )
    if current_player != 1 or int(np.count_nonzero(board[0])) != 4:
        raise AssertionError("Placement did not terminate at the standard P1 boundary.")
    return board, trace


def play_oracle_continuation(game, opening_board, oracle, nodes, max_plies=256):
    """Play both sides with one deterministic oracle after resetting its game TT."""
    oracle.reset()
    board = np.asarray(opening_board).astype(int, copy=True)
    current_player = 1
    total_nodes = 0
    for ply in range(int(max_plies) + 1):
        ended = game.getGameEnded(board, current_player)
        if ended != 0:
            return {
                "result": int(current_player * ended),
                "plies": int(ply),
                "nodes_visited": int(total_nodes),
            }
        if ply == int(max_plies):
            raise RuntimeError("Oracle continuation exceeded the maximum ply count.")
        canonical = game.getCanonicalForm(board, current_player)
        response = oracle.analyze(canonical, nodes=int(nodes))
        total_nodes += int(response["nodes_visited"])
        equivalents = external_actions_to_v3_actions(
            game, canonical, response["best_move"]["actions"]
        )
        board, current_player = game.getNextState(
            board, current_player, min(equivalents)
        )
    raise AssertionError("Unreachable continuation loop exit.")


def deterministic_block_seed(seed, teacher_a, teacher_b, mode, block_id, assignment):
    payload = "{}|{}|{}|{}|{}|{}".format(
        int(seed), teacher_a, teacher_b, mode, int(block_id), assignment
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def paired_placement_block(
    game,
    teacher_a,
    teacher_b,
    mode,
    block_id,
    seed,
    oracle,
    oracle_nodes,
):
    """Swap placement seats, then adjudicate both openings identically."""
    seed_ab = deterministic_block_seed(
        seed, teacher_a.name, teacher_b.name, mode, block_id, "a_as_p1"
    )
    seed_ba = deterministic_block_seed(
        seed, teacher_a.name, teacher_b.name, mode, block_id, "b_as_p1"
    )
    opening_ab, trace_ab = build_completed_opening(
        game, teacher_a, teacher_b, mode, seed_ab
    )
    opening_ba, trace_ba = build_completed_opening(
        game, teacher_b, teacher_a, mode, seed_ba
    )
    result_ab = play_oracle_continuation(
        game, opening_ab, oracle, oracle_nodes
    )
    result_ba = play_oracle_continuation(
        game, opening_ba, oracle, oracle_nodes
    )
    _, _, key_ab = canonicalize_board(opening_ab)
    _, _, key_ba = canonicalize_board(opening_ba)
    return {
        "teacher_a": teacher_a.name,
        "teacher_b": teacher_b.name,
        "mode": mode,
        "block_id": int(block_id),
        "seed_a_as_p1": int(seed_ab),
        "seed_b_as_p1": int(seed_ba),
        "a_as_p1": {
            "opening_hash": hashlib.sha256(key_ab).hexdigest(),
            "trace": trace_ab,
            **result_ab,
        },
        "b_as_p1": {
            "opening_hash": hashlib.sha256(key_ba).hexdigest(),
            "trace": trace_ba,
            **result_ba,
        },
    }


def summarize_paired_records(records, bootstrap_samples=10_000, seed=0):
    if not records:
        raise ValueError("Cannot summarize an empty placement matchup.")
    block_points = []
    a_results = []
    for record in records:
        result_ab = int(record["a_as_p1"]["result"])
        result_ba = int(record["b_as_p1"]["result"])
        if result_ab not in (-1, 0, 1) or result_ba not in (-1, 0, 1):
            raise ValueError("Placement adjudication result must be -1, 0, or 1.")
        # A is P1 in the first game and P2 in the second game.
        a_results.extend((result_ab, -result_ba))
        block_points.append(0.5 + 0.25 * (result_ab - result_ba))
    block_points = np.asarray(block_points, dtype=np.float64)
    a_results = np.asarray(a_results, dtype=np.int8)
    interval = None
    if len(block_points) > 1 and int(bootstrap_samples) > 0:
        rng = np.random.RandomState(int(seed))
        indices = rng.randint(
            0, len(block_points), size=(int(bootstrap_samples), len(block_points))
        )
        samples = block_points[indices].mean(axis=1)
        interval = [
            float(np.percentile(samples, 2.5)),
            float(np.percentile(samples, 97.5)),
        ]
    return {
        "blocks": len(records),
        "games": 2 * len(records),
        "teacher_a_score": float(np.mean(block_points)),
        "teacher_a_paired_bootstrap_95_interval": interval,
        "teacher_a_wins": int(np.sum(a_results == 1)),
        "teacher_b_wins": int(np.sum(a_results == -1)),
        "draws": int(np.sum(a_results == 0)),
        "a_sweeps": int(np.sum(block_points == 1.0)),
        "b_sweeps": int(np.sum(block_points == 0.0)),
        "split_blocks": int(np.sum(block_points == 0.5)),
        "mean_plies": float(np.mean([
            game["plies"]
            for record in records
            for game in (record["a_as_p1"], record["b_as_p1"])
        ])),
        "total_nodes_visited": int(sum(
            game["nodes_visited"]
            for record in records
            for game in (record["a_as_p1"], record["b_as_p1"])
        )),
        "distinct_d4_openings": len({
            game["opening_hash"]
            for record in records
            for game in (record["a_as_p1"], record["b_as_p1"])
        }),
    }
