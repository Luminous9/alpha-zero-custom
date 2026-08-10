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
