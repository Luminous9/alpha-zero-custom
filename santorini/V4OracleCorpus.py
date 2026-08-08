"""Reader and differential validator for santorini-ai V4 datagen shards."""

import json

import numpy as np

from .SantoriniOracle import (
    anonymous_board_key,
    external_actions_to_v3_actions,
    fen_to_canonical_board,
)


V4_CORPUS_SCHEMA_VERSION = 1
V4_TT_POLICY = "reset_per_independent_game"
V4_RECORD_SOURCES = {"main_line", "randomized_subgame"}

V4_MANIFEST_FIELDS = {
    "type",
    "schema_version",
    "engine_digest",
    "shard_id",
    "gods",
    "tt_policy",
    "generation",
}

V4_RECORD_FIELDS = {
    "type",
    "schema_version",
    "engine_digest",
    "shard_id",
    "game_id",
    "record_id",
    "fen",
    "side_to_move",
    "best_actions",
    "best_action_string",
    "best_successor_fen",
    "winner",
    "score",
    "mate_band",
    "completed_depth",
    "requested_nodes",
    "actual_nodes",
    "ply",
    "build_count",
    "random_prefix_plies",
    "source",
    "tt_policy",
}


def _require_fields(payload, required, description):
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError("{} is missing required fields: {}".format(description, missing))


def validate_v4_manifest(manifest):
    _require_fields(manifest, V4_MANIFEST_FIELDS, "V4 corpus manifest")
    if manifest["type"] != "manifest":
        raise ValueError("The first V4 corpus line must be a manifest.")
    if int(manifest["schema_version"]) != V4_CORPUS_SCHEMA_VERSION:
        raise ValueError("Unsupported V4 corpus schema version.")
    if manifest["gods"] != ["mortal", "mortal"]:
        raise ValueError("V4 corpus shards must be explicitly Mortal-vs-Mortal.")
    if manifest["tt_policy"] != V4_TT_POLICY:
        raise ValueError("V4 corpus shards must declare reset-per-independent-game TT use.")
    if not str(manifest["engine_digest"]).strip():
        raise ValueError("V4 corpus manifest engine_digest must not be empty.")
    if not str(manifest["shard_id"]).strip():
        raise ValueError("V4 corpus manifest shard_id must not be empty.")
    generation = manifest["generation"]
    for key in (
        "random_moves_min",
        "random_moves_max",
        "requested_node_limit",
        "min_depth_node_limit",
        "max_completed_depth",
        "subgame_initial_chance",
        "seed",
        "worker_index",
        "target_records",
    ):
        if key not in generation:
            raise ValueError("V4 corpus generation config is missing {}.".format(key))
    if int(generation["random_moves_min"]) > int(generation["random_moves_max"]):
        raise ValueError("V4 corpus random-move bounds are inverted.")
    if int(generation["target_records"]) < 1:
        raise ValueError("V4 corpus target_records must be positive.")
    return manifest


def _fen_player(fen):
    sections = str(fen).split("/")
    if len(sections) != 4 or sections[1] not in ("1", "2"):
        raise ValueError("V4 record contains an invalid FEN side-to-move marker.")
    return int(sections[1])


def validate_v4_record_metadata(manifest, record):
    """Validate record metadata and return its decoded position pair."""
    _require_fields(record, V4_RECORD_FIELDS, "V4 corpus record")
    if record["type"] != "position":
        raise ValueError("V4 corpus data lines must have type=position.")
    for key in ("schema_version", "engine_digest", "shard_id", "tt_policy"):
        expected = manifest[key]
        if record[key] != expected:
            raise ValueError("V4 record {} does not match its manifest.".format(key))
    if record["source"] not in V4_RECORD_SOURCES:
        raise ValueError("V4 record has an unknown source classification.")
    if not record["best_actions"] or not str(record["best_action_string"]).strip():
        raise ValueError("V4 record is missing its searched best action.")
    if any(action.get("type") == "no_moves" for action in record["best_actions"]):
        raise ValueError("V4 policy corpus cannot contain terminal no-moves records.")
    if int(record["side_to_move"]) != _fen_player(record["fen"]):
        raise ValueError("V4 record side_to_move disagrees with its FEN.")
    if int(record["winner"]) not in (1, 2):
        raise ValueError("V4 record winner must be player 1 or player 2.")
    if int(record["requested_nodes"]) < 1 or int(record["actual_nodes"]) < 1:
        raise ValueError("V4 record node counts are invalid.")
    if int(record["completed_depth"]) < 0 or int(record["ply"]) < 0:
        raise ValueError("V4 record depth/ply fields are invalid.")
    if bool(record["mate_band"]) != (abs(int(record["score"])) >= 9_000):
        raise ValueError("V4 record mate_band disagrees with its score.")
    generation = manifest["generation"]
    if int(record["requested_nodes"]) != int(generation["requested_node_limit"]):
        raise ValueError("V4 record requested_nodes disagrees with its manifest.")
    random_prefix = int(record["random_prefix_plies"])
    if not (
        int(generation["random_moves_min"])
        <= random_prefix
        <= int(generation["random_moves_max"])
    ):
        raise ValueError("V4 record random prefix is outside its manifest bounds.")
    if int(record["ply"]) < random_prefix:
        raise ValueError("V4 record ply precedes its random prefix.")
    if not str(record["game_id"]).startswith(str(manifest["shard_id"]) + ":"):
        raise ValueError("V4 record game_id does not belong to its shard.")
    if not str(record["record_id"]).startswith(str(record["game_id"]) + ":"):
        raise ValueError("V4 record_id does not belong to its game.")

    board = fen_to_canonical_board(record["fen"])
    successor = fen_to_canonical_board(record["best_successor_fen"])
    if int(record["build_count"]) != int(np.sum(board[1])):
        raise ValueError("V4 record build_count disagrees with its FEN.")
    return board, successor


def validate_v4_record(game, manifest, record):
    """Validate one record against both the shard contract and V3 rules."""
    board, successor = validate_v4_record_metadata(manifest, record)

    aliases = external_actions_to_v3_actions(game, board, record["best_actions"])
    expected_key = anonymous_board_key(successor)
    matching_aliases = []
    for action in aliases:
        next_board, next_player = game.getNextState(board, 1, int(action))
        next_canonical = game.getCanonicalForm(next_board, next_player)
        if anonymous_board_key(next_canonical) == expected_key:
            matching_aliases.append(int(action))
    if len(matching_aliases) != len(aliases):
        raise ValueError("V4 searched action path does not reproduce its successor FEN.")
    return matching_aliases


def load_v4_shard(path, game=None):
    """Load a JSONL shard, rejecting mixed metadata and duplicate record ids."""
    with open(path) as source:
        lines = [json.loads(line) for line in source if line.strip()]
    if not lines:
        raise ValueError("V4 corpus shard is empty: {}".format(path))
    manifest = validate_v4_manifest(lines[0])
    records = lines[1:]
    record_ids = [str(record.get("record_id")) for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("V4 corpus shard contains duplicate record ids.")
    winners_by_game = {}
    for record in records:
        if game is not None:
            validate_v4_record(game, manifest, record)
        else:
            validate_v4_record_metadata(manifest, record)
        game_id = str(record["game_id"])
        winner = int(record["winner"])
        if game_id in winners_by_game and winners_by_game[game_id] != winner:
            raise ValueError("V4 records from one game disagree about the winner.")
        winners_by_game[game_id] = winner
    return manifest, records
