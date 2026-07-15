from collections import deque
import os

import numpy as np


FORMAT_VERSION = 1


def _validate_compact_payload(payload):
    version = int(payload["format_version"][0])
    if version != FORMAT_VERSION:
        raise ValueError("Unsupported compact replay format version {}.".format(version))
    lengths = payload["history_lengths"].astype(np.int64)
    offsets = payload["policy_offsets"]
    example_count = int(lengths.sum())
    if len(payload["boards"]) != example_count or len(payload["values"]) != example_count:
        raise ValueError("Compact replay history lengths do not match board/value counts.")
    if len(offsets) != example_count + 1 or int(offsets[0]) != 0:
        raise ValueError("Compact replay policy offsets are inconsistent with example count.")
    if int(offsets[-1]) != len(payload["policy_indices"]):
        raise ValueError("Compact replay policy offsets do not match policy indices.")
    if len(payload["policy_indices"]) != len(payload["policy_values"]):
        raise ValueError("Compact replay policy indices and values have different lengths.")
    return lengths


def _write_compact_payload_atomic(destination_path, retained_payload, expected_lengths):
    temp_path = destination_path + ".trim.tmp"
    try:
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        with open(temp_path, "wb") as replay_file:
            np.savez_compressed(replay_file, **retained_payload)

        with np.load(temp_path, allow_pickle=False) as written:
            validated_lengths = _validate_compact_payload(written)
            if not np.array_equal(validated_lengths, expected_lengths):
                raise ValueError("Rewritten compact replay history windows failed validation.")
        os.replace(temp_path, destination_path)
    finally:
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass


def trim_compact_replay(path, keep_last_windows, output_path=None):
    """Atomically retain only the newest history windows in a compact replay."""
    keep_last_windows = int(keep_last_windows)
    if keep_last_windows < 1:
        raise ValueError("keep_last_windows must be at least 1.")

    source_path = os.path.abspath(os.fspath(path))
    destination_path = os.path.abspath(os.fspath(output_path or path))
    with np.load(source_path, allow_pickle=False) as payload:
        lengths = _validate_compact_payload(payload)
        before_windows = len(lengths)
        before_examples = int(lengths.sum())
        if keep_last_windows >= before_windows:
            if destination_path != source_path:
                raise ValueError("A separate output path requires trimming at least one history window.")
            return {
                "before_windows": before_windows,
                "after_windows": before_windows,
                "before_examples": before_examples,
                "after_examples": before_examples,
                "trimmed": False,
            }

        retained_lengths = lengths[-keep_last_windows:]
        first_example = int(lengths[:-keep_last_windows].sum())
        first_policy = int(payload["policy_offsets"][first_example])
        retained_offsets = (
            payload["policy_offsets"][first_example:].astype(np.int64) - first_policy
        )
        retained_payload = {
            "format_version": payload["format_version"],
            "action_size": payload["action_size"],
            "history_lengths": retained_lengths,
            "boards": payload["boards"][first_example:],
            "values": payload["values"][first_example:],
            "policy_offsets": retained_offsets,
            "policy_indices": payload["policy_indices"][first_policy:],
            "policy_values": payload["policy_values"][first_policy:],
        }

    _write_compact_payload_atomic(destination_path, retained_payload, retained_lengths)

    return {
        "before_windows": before_windows,
        "after_windows": len(retained_lengths),
        "before_examples": before_examples,
        "after_examples": int(retained_lengths.sum()),
        "trimmed": True,
    }


def collapse_compact_replay_symmetries(path, group_size=8):
    """Keep one representative from each legacy consecutive symmetry group."""
    group_size = int(group_size)
    if group_size < 2:
        raise ValueError("group_size must be at least 2.")

    destination_path = os.path.abspath(os.fspath(path))
    with np.load(destination_path, allow_pickle=False) as payload:
        lengths = _validate_compact_payload(payload)
        if np.any(lengths % group_size):
            raise ValueError(
                "Every history window length must be divisible by symmetry group size {}.".format(
                    group_size
                )
            )

        window_starts = np.concatenate(([0], np.cumsum(lengths[:-1]))).astype(np.int64)
        selected_examples = np.concatenate([
            start + np.arange(0, int(length), group_size, dtype=np.int64)
            for start, length in zip(window_starts, lengths)
        ])
        values = payload["values"]
        for example_index in selected_examples:
            group_values = values[example_index:example_index + group_size]
            if not np.all(group_values == group_values[0]):
                raise ValueError(
                    "Replay examples do not have constant values within symmetry groups."
                )

        offsets = payload["policy_offsets"]
        policy_lengths = offsets[selected_examples + 1] - offsets[selected_examples]
        retained_offsets = np.zeros(len(selected_examples) + 1, dtype=np.int64)
        retained_offsets[1:] = np.cumsum(policy_lengths, dtype=np.int64)
        policy_indices = payload["policy_indices"]
        policy_values = payload["policy_values"]
        retained_policy_indices = np.concatenate([
            policy_indices[int(offsets[index]):int(offsets[index + 1])]
            for index in selected_examples
        ])
        retained_policy_values = np.concatenate([
            policy_values[int(offsets[index]):int(offsets[index + 1])]
            for index in selected_examples
        ])
        retained_lengths = (lengths // group_size).astype(np.int64)
        retained_payload = {
            "format_version": payload["format_version"],
            "action_size": payload["action_size"],
            "history_lengths": retained_lengths,
            "boards": payload["boards"][selected_examples],
            "values": values[selected_examples],
            "policy_offsets": retained_offsets,
            "policy_indices": retained_policy_indices,
            "policy_values": retained_policy_values,
        }
        before_examples = int(lengths.sum())

    _write_compact_payload_atomic(destination_path, retained_payload, retained_lengths)
    return {
        "windows": len(retained_lengths),
        "before_examples": before_examples,
        "after_examples": int(retained_lengths.sum()),
        "symmetry_group_size": group_size,
        "collapsed": True,
    }


def save_compact_replay(path, history):
    lengths = np.asarray([len(window) for window in history], dtype=np.int64)
    examples = [example for window in history for example in window]
    if not examples:
        raise ValueError("Cannot save an empty replay buffer.")

    boards = np.asarray([example[0] for example in examples], dtype=np.int8)
    values = np.asarray([example[2] for example in examples], dtype=np.float32)
    offsets = np.zeros(len(examples) + 1, dtype=np.int64)
    policy_indices = []
    policy_values = []
    action_size = len(examples[0][1])

    for index, example in enumerate(examples):
        policy = np.asarray(example[1], dtype=np.float32)
        nonzero = np.flatnonzero(policy)
        policy_indices.append(nonzero.astype(np.uint16))
        policy_values.append(policy[nonzero])
        offsets[index + 1] = offsets[index] + len(nonzero)

    with open(path, "wb") as replay_file:
        np.savez_compressed(
            replay_file,
            format_version=np.asarray([FORMAT_VERSION], dtype=np.int16),
            action_size=np.asarray([action_size], dtype=np.int32),
            history_lengths=lengths,
            boards=boards,
            values=values,
            policy_offsets=offsets,
            policy_indices=np.concatenate(policy_indices) if policy_indices else np.array([], dtype=np.uint16),
            policy_values=np.concatenate(policy_values) if policy_values else np.array([], dtype=np.float32),
        )


def load_compact_replay(path):
    with np.load(path, allow_pickle=False) as payload:
        version = int(payload["format_version"][0])
        if version != FORMAT_VERSION:
            raise ValueError("Unsupported compact replay format version {}.".format(version))
        action_size = int(payload["action_size"][0])
        lengths = payload["history_lengths"].astype(np.int64)
        boards = payload["boards"]
        values = payload["values"]
        offsets = payload["policy_offsets"]
        indices = payload["policy_indices"]
        probabilities = payload["policy_values"]

        examples = []
        for example_index in range(len(boards)):
            start = int(offsets[example_index])
            end = int(offsets[example_index + 1])
            policy = np.zeros(action_size, dtype=np.float32)
            policy[indices[start:end].astype(np.int64)] = probabilities[start:end]
            examples.append((boards[example_index].astype(int), policy, float(values[example_index])))

    history = []
    start = 0
    for length in lengths:
        end = start + int(length)
        history.append(deque(examples[start:end]))
        start = end
    return history
