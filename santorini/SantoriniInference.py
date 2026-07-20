"""Exact-input neural inference deduplication for batched Santorini search."""

import numpy as np


def board_cache_key(board):
    board = np.ascontiguousarray(board)
    return board.dtype.str, board.shape, board.tobytes()


def predict_batch_deduplicated(nnet, boards, cache=None, max_cache_entries=0):
    """Predict each byte-identical board once and expand results to input order."""
    boards = list(boards)
    if not boards:
        return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=np.float32), {
            'requested': 0,
            'executed': 0,
            'reused': 0,
        }

    persistent_cache = cache if cache is not None else {}
    max_cache_entries = max(0, int(max_cache_entries))
    resolved = {}
    missing_keys = []
    missing_boards = []
    input_keys = []
    for board in boards:
        key = board_cache_key(board)
        input_keys.append(key)
        if key in persistent_cache:
            resolved[key] = persistent_cache[key]
        elif key not in resolved:
            # None marks a unique input that still needs network evaluation.
            resolved[key] = None
            missing_keys.append(key)
            missing_boards.append(board)

    if missing_boards:
        if hasattr(nnet, 'predict_batch'):
            missing_policies, missing_values = nnet.predict_batch(missing_boards)
        else:
            predictions = [nnet.predict(board) for board in missing_boards]
            missing_policies, missing_values = zip(*predictions)
        missing_policies = np.asarray(missing_policies)
        missing_values = np.asarray(missing_values).reshape(-1)
        for key, policy, value in zip(missing_keys, missing_policies, missing_values):
            prediction = (policy, float(value))
            resolved[key] = prediction
            if (
                cache is not None
                and max_cache_entries > 0
                and len(persistent_cache) < max_cache_entries
            ):
                persistent_cache[key] = prediction

        # The common late-game case has no duplicates. Avoid rebuilding and
        # copying a large dense policy batch when input and network order match.
        if len(missing_boards) == len(boards):
            return missing_policies, missing_values.astype(np.float32, copy=False), {
                'requested': len(boards),
                'executed': len(boards),
                'reused': 0,
            }

    policies = np.asarray([resolved[key][0] for key in input_keys])
    values = np.asarray([resolved[key][1] for key in input_keys], dtype=np.float32)
    executed = len(missing_boards)
    return policies, values, {
        'requested': len(boards),
        'executed': executed,
        'reused': len(boards) - executed,
    }
