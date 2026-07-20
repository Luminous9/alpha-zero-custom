"""Deterministic D4 diagnostics for Santorini policy and value predictions."""

import hashlib

import numpy as np


SYMMETRY_BUCKETS = ('placement', 'early', 'middle', 'late')


def transform_board(board, rotations, flip):
    transformed = np.rot90(np.asarray(board), rotations, axes=(-2, -1))
    if flip:
        transformed = np.flip(transformed, axis=-1)
    return np.ascontiguousarray(transformed)


def transformations(board):
    return [
        (rotations, flip, transform_board(board, rotations, flip))
        for rotations in range(4)
        for flip in (False, True)
    ]


def symmetry_bucket(game, board):
    if hasattr(game, 'isPlacementPhase') and game.isPlacementPhase(board):
        return 'placement'
    build_count = int(np.asarray(board)[1].sum())
    if build_count <= 5:
        return 'early'
    if build_count <= 15:
        return 'middle'
    return 'late'


def _network_visible_board(board):
    board = np.asarray(board)
    return np.stack((np.sign(board[0]), np.clip(board[1], 0, 4))).astype(np.int8)


def canonical_symmetry_key(board):
    visible = _network_visible_board(board)
    encodings = [candidate.tobytes() for _, _, candidate in transformations(visible)]
    return min(encodings)


def _canonical_representative(board):
    candidates = transformations(board)
    return min(
        candidates,
        key=lambda item: (
            _network_visible_board(item[2]).tobytes(),
            np.asarray(item[2]).tobytes(),
        ),
    )[2]


def build_diagnostic_suite(game, examples, sample_size):
    """Build a deterministic, D4-deduplicated, stage-stratified replay suite."""
    sample_size = max(0, int(sample_size))
    if sample_size == 0:
        return [], [], []

    grouped = {bucket: {} for bucket in SYMMETRY_BUCKETS}
    for example in examples:
        board = np.asarray(example[0])
        target = float(example[2])
        bucket = symmetry_bucket(game, board)
        key = canonical_symmetry_key(board)
        entry = grouped[bucket].setdefault(key, {'targets': []})
        entry['targets'].append(target)
        if 'board' not in entry:
            entry['board'] = _canonical_representative(board)

    quota, remainder = divmod(sample_size, len(SYMMETRY_BUCKETS))
    requested = {
        bucket: quota + int(index < remainder)
        for index, bucket in enumerate(SYMMETRY_BUCKETS)
    }
    selected = []
    leftovers = []
    for bucket in SYMMETRY_BUCKETS:
        entries = []
        for key, entry in grouped[bucket].items():
            rank = hashlib.blake2b(
                key,
                digest_size=16,
                person=b'SantoriniSym',
            ).digest()
            candidate = (
                rank,
                bucket,
                entry['board'],
                float(np.mean(entry['targets'])),
            )
            entries.append(candidate)
        entries.sort(key=lambda item: item[0])
        selected.extend(entries[:requested[bucket]])
        leftovers.extend(entries[requested[bucket]:])

    if len(selected) < sample_size:
        leftovers.sort(key=lambda item: item[0])
        selected.extend(leftovers[:sample_size - len(selected)])
    selected.sort(key=lambda item: (SYMMETRY_BUCKETS.index(item[1]), item[0]))
    return (
        [item[2] for item in selected],
        [item[3] for item in selected],
        [item[1] for item in selected],
    )


def suite_fingerprint(boards, targets, buckets):
    digest = hashlib.blake2b(digest_size=12, person=b'SantSymSuite')
    for board, target, bucket in zip(boards, targets, buckets):
        digest.update(canonical_symmetry_key(board))
        digest.update(np.float32(target).tobytes())
        digest.update(str(bucket).encode('ascii'))
    return digest.hexdigest()


def _restore_policy(game, policy, rotations, flip):
    old_indices, new_indices = game.getPolicySymmetryPermutation(rotations, flip)
    restored = np.zeros_like(policy)
    restored[old_indices] = policy[new_indices]
    return restored


def _normalized_legal_policy(game, board, policy):
    valids = game.getValidMoves(board, 1).astype(bool)
    legal = np.where(valids, np.asarray(policy, dtype=np.float64), 0.0)
    mass = float(legal.sum())
    if mass:
        legal /= mass
    return legal, mass


def position_metrics(game, variants, policies, values, target_value=None):
    restored_policies = []
    legal_masses = []
    for (rotations, flip, board), policy in zip(variants, policies):
        legal, mass = _normalized_legal_policy(game, board, policy)
        restored_policies.append(_restore_policy(game, legal, rotations, flip))
        legal_masses.append(mass)
    restored_policies = np.asarray(restored_policies, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    mean_policy = restored_policies.mean(axis=0)
    total_variations = 0.5 * np.abs(restored_policies - mean_policy).sum(axis=1)
    top_actions = np.argmax(restored_policies, axis=1)
    _, top_counts = np.unique(top_actions, return_counts=True)
    result = {
        'policy_mean_orbit_total_variation': float(total_variations.mean()),
        'policy_max_orbit_total_variation': float(total_variations.max()),
        'policy_top_action_consistency': float(top_counts.max() / len(top_actions)),
        'policy_any_top_action_change': bool(len(top_counts) > 1),
        'legal_policy_mass_mean': float(np.mean(legal_masses)),
        'legal_policy_mass_min': float(np.min(legal_masses)),
        'value_orbit_mean': float(values.mean()),
        'value_orbit_std': float(values.std()),
        'value_orbit_range': float(values.max() - values.min()),
        'value_orbit_mean_absolute_deviation': float(
            np.mean(np.abs(values - values.mean()))
        ),
        'value_sign_disagreement': bool(values.min() < 0 <= values.max()),
        'orientation_values': [float(value) for value in values],
    }
    if target_value is not None:
        target_value = float(target_value)
        orientation_mse = float(np.mean((values - target_value) ** 2))
        ensemble_mse = float((values.mean() - target_value) ** 2)
        result.update({
            'target_value': target_value,
            'value_orientation_mse': orientation_mse,
            'value_ensemble_mse': ensemble_mse,
            'value_symmetry_excess_mse': max(0.0, orientation_mse - ensemble_mse),
        })
    return result


def evaluate_suite(game, nnet, boards, targets=None, buckets=None):
    boards = list(boards)
    if not boards:
        return {'positions': [], 'aggregate': {}}
    targets = [None] * len(boards) if targets is None else list(targets)
    buckets = (
        [symmetry_bucket(game, board) for board in boards]
        if buckets is None else list(buckets)
    )
    all_variants = [transformations(board) for board in boards]
    flat_boards = [item[2] for variants in all_variants for item in variants]
    policies, values = nnet.predict_batch(flat_boards)
    positions = []
    for index, (board, target, bucket, variants) in enumerate(
        zip(boards, targets, buckets, all_variants)
    ):
        offset = index * 8
        metrics = position_metrics(
            game,
            variants,
            policies[offset:offset + 8],
            values[offset:offset + 8],
            target_value=target,
        )
        metrics['bucket'] = str(bucket)
        metrics['board'] = np.asarray(board)
        positions.append(metrics)
    aggregate = {'all': aggregate_positions(positions)}
    for bucket in SYMMETRY_BUCKETS:
        aggregate[bucket] = aggregate_positions([
            position for position in positions if position['bucket'] == bucket
        ])
    return {'positions': positions, 'aggregate': aggregate}


def aggregate_positions(positions):
    if not positions:
        return {'positions': 0}

    def values(key):
        return np.asarray([position[key] for position in positions], dtype=np.float64)

    value_stds = values('value_orbit_std')
    value_ranges = values('value_orbit_range')
    metrics = {
        'positions': len(positions),
        'policy_mean_orbit_total_variation': float(
            values('policy_mean_orbit_total_variation').mean()
        ),
        'policy_max_orbit_total_variation': float(
            values('policy_max_orbit_total_variation').max()
        ),
        'policy_top_action_consistency_rate': float(
            values('policy_top_action_consistency').mean()
        ),
        'policy_positions_with_top_action_change_rate': float(
            values('policy_any_top_action_change').mean()
        ),
        'legal_policy_mass_mean': float(values('legal_policy_mass_mean').mean()),
        'value_mean_orbit_std': float(value_stds.mean()),
        'value_p95_orbit_std': float(np.percentile(value_stds, 95)),
        'value_mean_orbit_range': float(value_ranges.mean()),
        'value_p95_orbit_range': float(np.percentile(value_ranges, 95)),
        'value_max_orbit_range': float(value_ranges.max()),
        'value_mean_orbit_absolute_deviation': float(
            values('value_orbit_mean_absolute_deviation').mean()
        ),
        'value_sign_disagreement_rate': float(values('value_sign_disagreement').mean()),
    }
    if all('value_orientation_mse' in position for position in positions):
        metrics.update({
            'value_orientation_mse': float(values('value_orientation_mse').mean()),
            'value_ensemble_mse': float(values('value_ensemble_mse').mean()),
            'value_symmetry_excess_mse': float(values('value_symmetry_excess_mse').mean()),
        })
    return metrics


def flatten_aggregate(aggregate, prefix='symmetry'):
    flattened = {}
    for bucket, metrics in aggregate.items():
        for key, value in metrics.items():
            flattened['{}_{}_{}'.format(prefix, bucket, key)] = value
    return flattened
