"""Shared, strategy-neutral utilities for Santorini oracle research tools."""

import threading

import numpy as np

from .SantoriniOracle import (
    SantoriniOracleProcess,
    canonical_board_to_fen,
    external_actions_to_v3_actions,
)


STAGES = ("early", "middle", "late")


def stage_for_builds(build_count):
    build_count = int(build_count)
    if build_count <= 5:
        return "early"
    if build_count <= 15:
        return "middle"
    return "late"


def canonical_d4_fen(board):
    variants = []
    for rotations in range(4):
        rotated = np.asarray([
            np.rot90(board[0], rotations),
            np.rot90(board[1], rotations),
        ])
        variants.append(canonical_board_to_fen(rotated))
        variants.append(canonical_board_to_fen(np.asarray([
            np.fliplr(rotated[0]),
            np.fliplr(rotated[1]),
        ])))
    return min(variants)


def collect_unique_replay_positions(replay_path):
    """Index the first replay orientation in each D4-equivalence class."""
    by_stage = {stage: {} for stage in STAGES}
    with np.load(replay_path, allow_pickle=False) as payload:
        for replay_index, board in enumerate(payload["boards"]):
            board = board.astype(int)
            if int(np.count_nonzero(board[0])) != 4:
                continue
            build_count = int(np.sum(board[1]))
            stage = stage_for_builds(build_count)
            d4_fen = canonical_d4_fen(board)
            existing = by_stage[stage].get(d4_fen)
            if existing is None:
                by_stage[stage][d4_fen] = {
                    "fen": d4_fen,
                    "d4_fen": d4_fen,
                    "stage": stage,
                    "build_count": build_count,
                    "replay_index": int(replay_index),
                    "replay_observations": 1,
                }
            else:
                existing["replay_observations"] += 1
    return {
        stage: sorted(by_stage[stage].values(), key=lambda record: record["d4_fen"])
        for stage in STAGES
    }


def decode_policy(payload, replay_index):
    action_size = int(payload["action_size"][0])
    offsets = payload["policy_offsets"]
    start = int(offsets[replay_index])
    end = int(offsets[replay_index + 1])
    policy = np.zeros(action_size, dtype=np.float32)
    indices = payload["policy_indices"][start:end].astype(np.int64)
    policy[indices] = payload["policy_values"][start:end]
    total = float(policy.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Replay policy {} has invalid mass {}.".format(replay_index, total))
    policy /= total
    return policy


def score_softmax(moves, temperature=100.0):
    if not moves:
        raise ValueError("Cannot create a soft target without ranked moves.")
    if float(temperature) <= 0:
        raise ValueError("Score temperature must be positive.")
    scores = np.asarray([float(move["score"]) for move in moves], dtype=np.float64)
    logits = np.clip((scores - scores.max()) / float(temperature), -50.0, 0.0)
    weights = np.exp(logits)
    return weights / weights.sum()


def top_overlap(first, second, count=3):
    first_set = {move["next_fen"] for move in first[:count]}
    second_set = {move["next_fen"] for move in second[:count]}
    union = first_set | second_set
    return float(len(first_set & second_set) / len(union)) if union else 1.0


def normalized_entropy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if len(probabilities) <= 1:
        return 0.0
    entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-30))))
    return entropy / float(np.log(len(probabilities)))


def confidence_metrics(shallow, deep, score_temperature):
    shallow_moves = shallow["moves"]
    deep_moves = deep["moves"]
    overlap = top_overlap(shallow_moves, deep_moves, count=3)
    top1_agreement = shallow_moves[0]["next_fen"] == deep_moves[0]["next_fen"]
    probabilities = score_softmax(deep_moves, score_temperature)
    scores = [int(move["score"]) for move in deep_moves]
    score_margin = scores[0] - scores[1] if len(scores) > 1 else None
    return {
        "top1_agreement": bool(top1_agreement),
        "top3_jaccard": float(overlap),
        "confident": bool(top1_agreement and overlap >= 0.5),
        "deep_score_margin": score_margin,
        "deep_soft_target_entropy": normalized_entropy(probabilities),
        "deep_soft_target_probabilities": probabilities.tolist(),
    }


def blended_teacher_policy(source_policy, oracle_actions, oracle_weight):
    source_policy = np.asarray(source_policy, dtype=np.float32)
    oracle_actions = sorted(set(int(action) for action in oracle_actions))
    if not oracle_actions:
        raise ValueError("The oracle returned no V3-equivalent actions.")
    if not 0 <= float(oracle_weight) <= 1:
        raise ValueError("Oracle weight must be between zero and one.")
    teacher = np.zeros_like(source_policy)
    teacher[oracle_actions] = 1.0 / len(oracle_actions)
    blended = (1.0 - float(oracle_weight)) * source_policy + float(oracle_weight) * teacher
    blended /= blended.sum()
    return blended.astype(np.float32)


def ranked_moves_to_v3_policy(game, board, moves, score_temperature):
    probabilities = score_softmax(moves, score_temperature)
    policy = np.zeros(game.getActionSize(), dtype=np.float32)
    for move, probability in zip(moves, probabilities):
        aliases = external_actions_to_v3_actions(game, board, move["actions"])
        alias_probability = float(probability) / len(aliases)
        for action in aliases:
            policy[int(action)] += alias_probability
    policy /= policy.sum()
    return policy


def blend_policies(source, oracle, oracle_weight):
    if not 0 <= float(oracle_weight) <= 1:
        raise ValueError("Oracle weight must be between zero and one.")
    blended = (
        (1.0 - float(oracle_weight)) * np.asarray(source, dtype=np.float32)
        + float(oracle_weight) * np.asarray(oracle, dtype=np.float32)
    )
    blended /= blended.sum()
    return blended.astype(np.float32)


class ParallelOraclePool:
    """Give every executor thread its own long-lived oracle subprocess."""

    def __init__(self, binary):
        self.binary = binary
        self.local = threading.local()
        self.oracles = []
        self.lock = threading.Lock()

    def oracle(self):
        oracle = getattr(self.local, "oracle", None)
        if oracle is None:
            oracle = SantoriniOracleProcess(self.binary)
            self.local.oracle = oracle
            with self.lock:
                self.oracles.append(oracle)
        return oracle

    def analyze(self, analyzer, *args, **kwargs):
        return analyzer(self.oracle(), *args, **kwargs)

    def close(self):
        for oracle in self.oracles:
            oracle.close()
