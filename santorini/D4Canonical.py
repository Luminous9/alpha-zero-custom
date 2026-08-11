"""Shared exact D4 canonicalization utilities for V4 data and inference."""

import numpy as np

from .SantoriniOracle import anonymous_board_key


D4_TRANSFORMS = tuple(
    (rotations, flip)
    for rotations in range(4)
    for flip in (False, True)
)


def transform_board(board, rotations, flip):
    """Apply the spatial transform used by ``SantoriniGame.getSymmetries``."""
    transformed = np.rot90(np.asarray(board), int(rotations), axes=(-2, -1))
    if flip:
        transformed = np.flip(transformed, axis=-1)
    return np.ascontiguousarray(transformed)


def normalize_worker_labels(board):
    """Give indistinguishable workers deterministic labels after a transform."""
    board = np.asarray(board).copy()
    for sign in (1, -1):
        for label, location in enumerate(
            sorted(map(tuple, np.argwhere(board[0] * sign > 0))), start=1
        ):
            board[0][location] = sign * label
    return board


def _normalize_worker_labels_batch(boards):
    boards = np.asarray(boards).copy()
    pieces = boards[:, 0].reshape(len(boards), -1)
    positive = pieces > 0
    negative = pieces < 0
    positive_labels = np.cumsum(positive, axis=1) * positive
    negative_labels = np.cumsum(negative, axis=1) * negative
    pieces[:] = positive_labels - negative_labels
    return boards


def canonicalize_boards(boards):
    """Canonicalize a batch and return representative masks and anonymous keys.

    ``matching_masks[i, j]`` is true when transform ``j`` maps input ``i`` to
    its minimal anonymous representative. Keeping the full mask preserves exact
    stabilizer projection while avoiding per-board Python work in inference.
    """
    boards = np.asarray(boards)
    if boards.ndim != 4 or boards.shape[1:] != (2, 5, 5):
        raise ValueError("D4 canonicalization expects boards shaped (N, 2, 5, 5).")
    if not len(boards):
        return (
            np.empty((0, 2, 5, 5), dtype=boards.dtype),
            np.empty((0, len(D4_TRANSFORMS)), dtype=bool),
            [],
        )
    # At one or two positions, setting up the vectorized eight-transform tensor
    # costs more than the scalar path. MCTS commonly emits a few such tail
    # batches, so retain the cheaper exact implementation there.
    if len(boards) <= 2:
        items = [canonicalize_board(board) for board in boards]
        matching_masks = np.asarray([
            [transform in item[1] for transform in D4_TRANSFORMS]
            for item in items
        ], dtype=bool)
        return (
            np.asarray([item[0] for item in items]),
            matching_masks,
            [item[2] for item in items],
        )

    transformed = np.stack([
        transform_board(boards, rotations, flip)
        for rotations, flip in D4_TRANSFORMS
    ], axis=1)
    anonymous = np.empty(transformed.shape, dtype=np.int8)
    anonymous[:, :, 0] = np.sign(transformed[:, :, 0]).astype(np.int8)
    anonymous[:, :, 1] = transformed[:, :, 1].astype(np.int8)
    flat_keys = np.ascontiguousarray(anonymous).reshape(
        len(boards), len(D4_TRANSFORMS), -1
    ).view(np.uint8)

    # Python bytes compare lexicographically as unsigned octets. Iteratively
    # retain transforms matching the smallest byte at each offset to reproduce
    # ``min(anonymous_board_key(...))`` exactly without Python byte objects.
    matching_masks = np.ones(
        (len(boards), len(D4_TRANSFORMS)), dtype=bool
    )
    for offset in range(flat_keys.shape[2]):
        values = flat_keys[:, :, offset].astype(np.int16)
        minimum = np.min(
            np.where(matching_masks, values, 256), axis=1
        )
        matching_masks &= values == minimum[:, None]

    first_indices = np.argmax(matching_masks, axis=1)
    canonical = transformed[np.arange(len(boards)), first_indices]
    canonical = _normalize_worker_labels_batch(canonical)
    canonical_keys = [
        anonymous[index, transform].tobytes()
        for index, transform in enumerate(first_indices)
    ]
    return canonical, matching_masks, canonical_keys


def canonicalize_board(board):
    """Return the minimal anonymous D4 representative and all maps reaching it.

    Multiple transforms reach the representative when the position has a
    non-trivial stabilizer. Keeping every such transform is required to project
    a directional policy onto that stabilizer before mapping it back.
    """
    transformed = [
        transform_board(board, rotations, flip)
        for rotations, flip in D4_TRANSFORMS
    ]
    keys = [anonymous_board_key(item) for item in transformed]
    canonical_key = min(keys)
    matching_transforms = tuple(
        transform
        for transform, key in zip(D4_TRANSFORMS, keys)
        if key == canonical_key
    )
    first_index = keys.index(canonical_key)
    canonical_board = normalize_worker_labels(transformed[first_index])
    return canonical_board, matching_transforms, canonical_key


def canonicalize_board_policy(game, board, policy):
    """Choose the minimal D4 board and average policy over its stabilizer."""
    canonical_board, matching_transforms, canonical_key = canonicalize_board(board)
    transformed_policies = [
        game._transform_policy_array(np.asarray(policy), rotations, flip)
        for rotations, flip in matching_transforms
    ]
    canonical_policy = np.mean(
        np.asarray(transformed_policies, dtype=np.float64), axis=0
    )
    return canonical_board, canonical_policy, canonical_key


def restore_canonical_policy(game, canonical_policy, matching_transforms):
    """Project a canonical-frame policy and map it to the original frame."""
    canonical_policy = np.asarray(canonical_policy)
    restored = np.zeros_like(canonical_policy)
    for rotations, flip in matching_transforms:
        old_indices, new_indices = game.getPolicySymmetryPermutation(rotations, flip)
        restored[old_indices] += canonical_policy[new_indices]
    restored /= len(matching_transforms)
    return restored


def restore_canonical_policies(game, canonical_policies, matching_masks):
    """Batch-project canonical policies and restore their input frames."""
    canonical_policies = np.asarray(canonical_policies)
    matching_masks = np.asarray(matching_masks, dtype=bool)
    if canonical_policies.ndim != 2:
        raise ValueError("Canonical policies must have shape (N, action_size).")
    if matching_masks.shape != (len(canonical_policies), len(D4_TRANSFORMS)):
        raise ValueError("Canonical policy transform masks have the wrong shape.")
    if canonical_policies.shape[1] != game.getActionSize():
        raise ValueError("Canonical policies have the wrong action size.")
    counts = matching_masks.sum(axis=1)
    if np.any(counts == 0):
        raise ValueError("Every canonical policy requires a matching transform.")
    if len(canonical_policies) <= 2:
        return np.asarray([
            restore_canonical_policy(
                game,
                policy,
                tuple(
                    transform
                    for transform, matches in zip(D4_TRANSFORMS, mask)
                    if matches
                ),
            )
            for policy, mask in zip(canonical_policies, matching_masks)
        ])

    permutations = [
        game.getPolicySymmetryPermutation(rotations, flip)[1]
        for rotations, flip in D4_TRANSFORMS
    ]
    restored = np.zeros_like(canonical_policies)
    single = counts == 1
    for transform_index, permutation in enumerate(permutations):
        rows = np.flatnonzero(single & matching_masks[:, transform_index])
        if len(rows):
            restored[rows] = canonical_policies[rows][:, permutation]

    symmetric_rows = np.flatnonzero(~single)
    if len(symmetric_rows):
        for transform_index, permutation in enumerate(permutations):
            rows = symmetric_rows[matching_masks[symmetric_rows, transform_index]]
            if len(rows):
                restored[rows] += canonical_policies[rows][:, permutation]
        restored[symmetric_rows] /= counts[symmetric_rows, None]
    return restored
