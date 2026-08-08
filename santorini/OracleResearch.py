"""Shared, strategy-neutral utilities for Santorini oracle research tools."""

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
import time

import numpy as np

from .SantoriniOracle import (
    DEFAULT_ORACLE_BINARY,
    SantoriniOracleProcess,
    canonical_board_to_fen,
    external_actions_to_v3_actions,
)


STAGES = ("early", "middle", "late")
ORACLE_LABEL_CACHE_SCHEMA_VERSION = 1
MATE_SCORE_THRESHOLD = 9_000


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


def file_sha256(path):
    """Return a content identity suitable for cache keys and manifests."""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class OracleLabelCache:
    """Persistent, process-safe cache for independently searched value labels."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.connection = sqlite3.connect(
            str(self.path), timeout=30.0, check_same_thread=False
        )
        self.connection.execute("PRAGMA busy_timeout = 30000")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS oracle_labels (
                d4_fen TEXT NOT NULL,
                node_budget INTEGER NOT NULL,
                engine_digest TEXT NOT NULL,
                calibration_version TEXT NOT NULL,
                response_json TEXT NOT NULL,
                score INTEGER NOT NULL,
                mate_band INTEGER NOT NULL,
                completed_depth INTEGER NOT NULL,
                actual_nodes INTEGER NOT NULL,
                mapped_value REAL NOT NULL,
                created_at_unix REAL NOT NULL,
                PRIMARY KEY (
                    d4_fen, node_budget, engine_digest, calibration_version
                )
            )
            """
        )
        stored_schema = self.connection.execute(
            "SELECT value FROM cache_metadata WHERE key = 'schema_version'"
        ).fetchone()
        if stored_schema is None:
            self.connection.execute(
                "INSERT INTO cache_metadata(key, value) VALUES('schema_version', ?)",
                (str(ORACLE_LABEL_CACHE_SCHEMA_VERSION),),
            )
        elif int(stored_schema[0]) != ORACLE_LABEL_CACHE_SCHEMA_VERSION:
            raise ValueError(
                "Oracle label cache schema {} is incompatible with expected schema {}.".format(
                    stored_schema[0], ORACLE_LABEL_CACHE_SCHEMA_VERSION
                )
            )
        self.connection.commit()

    @staticmethod
    def _key(d4_fen, node_budget, engine_digest, calibration_version):
        node_budget = int(node_budget)
        if not str(d4_fen).strip():
            raise ValueError("Oracle label D4 FEN must not be empty.")
        if node_budget < 1:
            raise ValueError("Oracle label node budget must be positive.")
        if not engine_digest:
            raise ValueError("Oracle label engine digest must not be empty.")
        if not calibration_version:
            raise ValueError("Oracle label calibration version must not be empty.")
        return (
            str(d4_fen),
            node_budget,
            str(engine_digest),
            str(calibration_version),
        )

    @staticmethod
    def _row_to_record(row):
        if row is None:
            return None
        return {
            "d4_fen": row[0],
            "requested_nodes": int(row[1]),
            "engine_digest": row[2],
            "calibration_version": row[3],
            "response": json.loads(row[4]),
            "score": int(row[5]),
            "mate_band": bool(row[6]),
            "completed_depth": int(row[7]),
            "actual_nodes": int(row[8]),
            "mapped_value": float(row[9]),
            "created_at_unix": float(row[10]),
        }

    def get(self, d4_fen, node_budget, engine_digest, calibration_version):
        key = self._key(d4_fen, node_budget, engine_digest, calibration_version)
        with self.lock:
            row = self.connection.execute(
                """
                SELECT d4_fen, node_budget, engine_digest, calibration_version,
                       response_json, score, mate_band, completed_depth,
                       actual_nodes, mapped_value, created_at_unix
                FROM oracle_labels
                WHERE d4_fen = ? AND node_budget = ? AND engine_digest = ?
                      AND calibration_version = ?
                """,
                key,
            ).fetchone()
        return self._row_to_record(row)

    def put(self, record):
        key = self._key(
            record["d4_fen"],
            record["requested_nodes"],
            record["engine_digest"],
            record["calibration_version"],
        )
        mapped_value = float(record["mapped_value"])
        if not np.isfinite(mapped_value) or not -1.0 <= mapped_value <= 1.0:
            raise ValueError("Oracle mapped value must be finite and between -1 and 1.")
        values = key + (
            json.dumps(record["response"], sort_keys=True, separators=(",", ":")),
            int(record["score"]),
            int(bool(record["mate_band"])),
            int(record["completed_depth"]),
            int(record["actual_nodes"]),
            mapped_value,
            float(record.get("created_at_unix", time.time())),
        )
        with self.lock:
            self.connection.execute(
                """
                INSERT OR IGNORE INTO oracle_labels(
                    d4_fen, node_budget, engine_digest, calibration_version,
                    response_json, score, mate_band, completed_depth,
                    actual_nodes, mapped_value, created_at_unix
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            self.connection.commit()
        return self.get(*key)

    def close(self):
        connection = getattr(self, "connection", None)
        if connection is not None:
            self.connection = None
            connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class ParallelOraclePool:
    """Give every executor thread an oracle whose searches start from a cold TT."""

    def __init__(self, binary, cache_path=None, oracle_factory=SantoriniOracleProcess):
        self.binary = binary
        self.oracle_factory = oracle_factory
        self.local = threading.local()
        self.oracles = []
        self.lock = threading.Lock()
        self.cache = OracleLabelCache(cache_path) if cache_path is not None else None
        digest_path = (
            binary
            or os.environ.get("SANTORINI_ORACLE_BINARY")
            or DEFAULT_ORACLE_BINARY
        )
        self.engine_digest = file_sha256(digest_path) if cache_path is not None else None

    def oracle(self):
        oracle = getattr(self.local, "oracle", None)
        if oracle is None:
            oracle = self.oracle_factory(self.binary)
            self.local.oracle = oracle
            with self.lock:
                self.oracles.append(oracle)
        return oracle

    def analyze(self, analyzer, *args, **kwargs):
        oracle = self.oracle()
        oracle.reset()
        return analyzer(oracle, *args, **kwargs)

    def label_fen(self, d4_fen, nodes, calibration_version, value_mapper):
        """Return a cached scalar label, searching from a cold TT on a miss.

        ``d4_fen`` must already be in the caller's chosen D4-canonical
        orientation. Keeping that requirement explicit prevents returning a best
        move expressed in a different orientation than the requested position.
        """
        if self.cache is None:
            raise ValueError("ParallelOraclePool requires cache_path for label_fen().")
        cached = self.cache.get(
            d4_fen, nodes, self.engine_digest, calibration_version
        )
        if cached is not None:
            cached["cache_hit"] = True
            return cached

        response = self.analyze(
            lambda oracle: oracle.analyze_fen(d4_fen, nodes=int(nodes))
        )
        if int(response["requested_nodes"]) != int(nodes):
            raise ValueError("Oracle response requested_nodes does not match the query.")
        best = response["best_move"]
        score = int(best["score"])
        record = {
            "d4_fen": str(d4_fen),
            "requested_nodes": int(nodes),
            "engine_digest": self.engine_digest,
            "calibration_version": str(calibration_version),
            "response": response,
            "score": score,
            "mate_band": abs(score) >= MATE_SCORE_THRESHOLD,
            "completed_depth": int(response["completed_depth"]),
            "actual_nodes": int(response["nodes_visited"]),
            "mapped_value": float(value_mapper(score)),
            "created_at_unix": time.time(),
        }
        stored = self.cache.put(record)
        stored["cache_hit"] = False
        return stored

    def close(self):
        for oracle in self.oracles:
            oracle.close()
        if self.cache is not None:
            self.cache.close()
