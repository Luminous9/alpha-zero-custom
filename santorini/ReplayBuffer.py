from collections import deque

import numpy as np


FORMAT_VERSION = 1


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
