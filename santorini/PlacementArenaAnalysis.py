"""Duplicate-aware placement-inclusive neural arena evaluation."""

from collections import Counter, defaultdict
import hashlib
import time

import numpy as np

from BatchedArena import BatchedMCTSArena


def _sha256_parts(*parts):
    digest = hashlib.sha256()
    for part in parts:
        if isinstance(part, str):
            part = part.encode('utf-8')
        elif not isinstance(part, bytes):
            part = np.asarray(part).tobytes()
        digest.update(len(part).to_bytes(8, 'little'))
        digest.update(part)
    return digest.hexdigest()


def _side_signature(side_to_player):
    return 'p1={}|p2={}'.format(
        int(side_to_player[1]), int(side_to_player[-1])
    )


def _current_score(winner):
    return 1.0 if int(winner) == -1 else 0.5 if int(winner) == 0 else 0.0


def _bootstrap_interval(values, seed, samples=10_000, weights=None):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None, None
    weights = (
        np.ones(len(values), dtype=np.float64)
        if weights is None else np.asarray(weights, dtype=np.float64)
    )
    if weights.shape != values.shape or np.any(weights < 0) or not weights.sum():
        raise ValueError('Invalid bootstrap weights.')
    rng = np.random.RandomState(int(seed))
    means = []
    remaining = int(samples)
    while remaining:
        count = min(500, remaining)
        indices = rng.randint(len(values), size=(count, len(values)))
        sampled_values = values[indices]
        sampled_weights = weights[indices]
        means.append(
            np.sum(sampled_values * sampled_weights, axis=1)
            / np.sum(sampled_weights, axis=1)
        )
        remaining -= count
    low, high = np.quantile(np.concatenate(means), (0.025, 0.975))
    return float(low), float(high)


def _score_summary(values, seed, samples=10_000, weights=None, scale=1.0):
    values = np.asarray(values, dtype=np.float64) / float(scale)
    weights = (
        np.ones(len(values), dtype=np.float64)
        if weights is None else np.asarray(weights, dtype=np.float64)
    )
    low, high = _bootstrap_interval(values, seed, samples, weights)
    return {
        'score': float(np.average(values, weights=weights)),
        'bootstrap_95_low': low,
        'bootstrap_95_high': high,
        'units': int(len(values)),
        'weight_sum': float(weights.sum()),
        'weight_ess': float(weights.sum() ** 2 / np.sum(weights ** 2)),
    }


def _grouped_score(records, key, seed, samples, scale=1.0):
    groups = defaultdict(list)
    for record in records:
        groups[record[key]].append(float(record['score']))
    values = [float(np.mean(group)) for group in groups.values()]
    summary = _score_summary(values, seed, samples, scale=scale)
    summary['maximum_multiplicity'] = max(map(len, groups.values()))
    summary['frequency_histogram'] = {
        str(frequency): int(count)
        for frequency, count in sorted(Counter(map(len, groups.values())).items())
    }
    return summary


def _capped_group_weights(keys, cap_fraction):
    """Return occurrence weights whose normalized group share is capped.

    When there are fewer than ``ceil(1 / cap_fraction)`` groups, the requested
    cap is mathematically impossible. In that case equal-group weighting is the
    most diverse achievable view and ``cap_achievable`` is false.
    """
    frequencies = Counter(keys)
    group_count = len(frequencies)
    if not group_count:
        raise ValueError('Cannot cap an empty collection of groups.')
    minimum_share = 1.0 / group_count
    cap_achievable = minimum_share <= float(cap_fraction) + 1e-12
    if not cap_achievable:
        group_mass_cap = 0.0
    elif max(frequencies.values()) / len(keys) <= float(cap_fraction):
        return np.ones(len(keys), dtype=np.float64), {
            'cap_achievable': True,
            'group_mass_cap': float(max(frequencies.values())),
            'maximum_weighted_group_share': float(
                max(frequencies.values()) / len(keys)
            ),
        }
    else:
        low = 0.0
        high = float(max(frequencies.values()))
        for _ in range(80):
            middle = (low + high) / 2.0
            total = sum(min(float(frequency), middle) for frequency in frequencies.values())
            share = middle / total if total else minimum_share
            if share <= float(cap_fraction):
                low = middle
            else:
                high = middle
        group_mass_cap = low

    if group_mass_cap == 0.0:
        group_weights = {
            key: 1.0 / frequency for key, frequency in frequencies.items()
        }
    else:
        group_weights = {
            key: min(1.0, group_mass_cap / frequency)
            for key, frequency in frequencies.items()
        }
    weights = np.asarray([group_weights[key] for key in keys], dtype=np.float64)
    weighted_masses = defaultdict(float)
    for key, weight in zip(keys, weights):
        weighted_masses[key] += float(weight)
    maximum_share = max(weighted_masses.values()) / float(weights.sum())
    return weights, {
        'cap_achievable': bool(cap_achievable),
        'group_mass_cap': float(group_mass_cap),
        'maximum_weighted_group_share': float(maximum_share),
    }


def _paired_weighted_score(pair_records, game_weights, seed, samples):
    """Weighted score with a cluster bootstrap over seat-swapped pairs."""
    game_weights = np.asarray(game_weights, dtype=np.float64)
    if len(game_weights) != 2 * len(pair_records):
        raise ValueError('Expected two game weights per pair.')
    numerators = []
    denominators = []
    offset = 0
    for pair in pair_records:
        weights = game_weights[offset:offset + 2]
        scores = np.asarray(
            [float(game['score']) for game in pair['games']], dtype=np.float64
        )
        numerators.append(float(np.dot(weights, scores)))
        denominators.append(float(weights.sum()))
        offset += 2
    numerators = np.asarray(numerators, dtype=np.float64)
    denominators = np.asarray(denominators, dtype=np.float64)
    rng = np.random.RandomState(int(seed))
    means = []
    remaining = int(samples)
    while remaining:
        count = min(500, remaining)
        indices = rng.randint(len(pair_records), size=(count, len(pair_records)))
        means.append(
            numerators[indices].sum(axis=1) / denominators[indices].sum(axis=1)
        )
        remaining -= count
    low, high = np.quantile(np.concatenate(means), (0.025, 0.975))
    return {
        'score': float(numerators.sum() / denominators.sum()),
        'bootstrap_95_low': float(low),
        'bootstrap_95_high': float(high),
        'units': int(len(pair_records)),
        'weight_sum': float(game_weights.sum()),
        'weight_ess': float(
            game_weights.sum() ** 2 / np.sum(game_weights ** 2)
        ),
    }


def summarize_duplicate_aware_records(
    game_records,
    seed,
    bootstrap_samples=10_000,
    d4_cap_fraction=0.05,
    minimum_d4_ess_ratio=0.75,
):
    """Summarize already-adjudicated paired placement records."""
    grouped = defaultdict(list)
    for record in game_records:
        grouped[int(record['pair_index'])].append(record)
    if not grouped or any(len(group) != 2 for group in grouped.values()):
        raise ValueError('Duplicate-aware arena requires two games per pair.')

    pair_records = []
    for pair_index, games in sorted(grouped.items()):
        games = sorted(games, key=lambda item: item['seat_order'])
        if {game['seat_order'] for game in games} != {
            'contestant1_first', 'contestant2_first'
        }:
            raise ValueError('Duplicate-aware pair is not seat-swapped.')
        pair_records.append({
            'pair_index': pair_index,
            'score': float(sum(game['score'] for game in games)),
            'exact_block_key': _sha256_parts(*[
                game['exact_continuation_key'] for game in games
            ]),
            'd4_block_key': _sha256_parts(*[
                game['d4_opening_key'] for game in games
            ]),
            'games': games,
        })

    raw_values = [record['score'] for record in pair_records]
    raw = _score_summary(
        raw_values, int(seed) ^ 0x5101, bootstrap_samples, scale=2.0
    )
    exact_blocks = _grouped_score(
        pair_records, 'exact_block_key', int(seed) ^ 0x5102,
        bootstrap_samples, scale=2.0,
    )
    d4_blocks = _grouped_score(
        pair_records, 'd4_block_key', int(seed) ^ 0x5103,
        bootstrap_samples, scale=2.0,
    )

    d4_frequencies = Counter(
        game['d4_opening_key'] for record in pair_records for game in record['games']
    )
    exact_frequencies = Counter(
        game['exact_opening_key'] for record in pair_records for game in record['games']
    )
    game_count = 2 * len(pair_records)
    game_level = [game for record in pair_records for game in record['games']]
    full_trajectory_frequencies = Counter(
        game['full_trajectory_hash'] for game in game_level
    )
    capped_weights, cap_details = _capped_group_weights(
        [game['d4_opening_key'] for game in game_level], d4_cap_fraction
    )
    capped = _paired_weighted_score(
        pair_records, capped_weights, int(seed) ^ 0x5104, bootstrap_samples
    )
    capped['d4_cap_fraction'] = float(d4_cap_fraction)
    capped.update(cap_details)

    unique_continuations = _grouped_score(
        game_level, 'exact_continuation_key', int(seed) ^ 0x5105,
        bootstrap_samples,
    )
    equal_d4_openings = _grouped_score(
        game_level, 'd4_opening_key', int(seed) ^ 0x5106,
        bootstrap_samples,
    )

    d4_ess = float(game_count ** 2 / sum(
        frequency ** 2 for frequency in d4_frequencies.values()
    ))
    exact_ess = float(game_count ** 2 / sum(
        frequency ** 2 for frequency in exact_frequencies.values()
    ))
    required_d4_ess = float(minimum_d4_ess_ratio * len(pair_records))
    max_d4_frequency = max(d4_frequencies.values())
    max_exact_frequency = max(exact_frequencies.values())

    def point_direction(summary):
        return 1 if summary['score'] > 0.5 else -1 if summary['score'] < 0.5 else 0

    def resolved_direction(summary):
        if summary['bootstrap_95_low'] > 0.5:
            return 1
        if summary['bootstrap_95_high'] < 0.5:
            return -1
        return 0

    direction_agreement = point_direction(raw) == point_direction(capped)
    resolved_agreement = (
        resolved_direction(raw) != 0
        and resolved_direction(raw) == resolved_direction(capped)
    )
    natural_diversity_pass = (
        d4_ess >= required_d4_ess
        and max_d4_frequency / game_count <= float(d4_cap_fraction)
    )

    return {
        'schema_version': 1,
        'contract': 'duplicate_aware_placement_inclusive_summary',
        'games': int(game_count),
        'pairs': int(len(pair_records)),
        'pair_outcomes': {
            'candidate_2_0': int(sum(record['score'] == 2.0 for record in pair_records)),
            'split_1_1': int(sum(record['score'] == 1.0 for record in pair_records)),
            'candidate_0_2': int(sum(record['score'] == 0.0 for record in pair_records)),
            'other_with_draws': int(sum(
                record['score'] not in (0.0, 1.0, 2.0) for record in pair_records
            )),
        },
        'raw_policy_weighted': raw,
        'capped_d4': capped,
        'unique_exact_blocks': exact_blocks,
        'unique_d4_blocks': d4_blocks,
        'unique_exact_continuations': unique_continuations,
        'equal_d4_openings': equal_d4_openings,
        'diversity': {
            'distinct_exact_openings': int(len(exact_frequencies)),
            'distinct_d4_openings': int(len(d4_frequencies)),
            'exact_opening_ess': exact_ess,
            'd4_opening_ess': d4_ess,
            'required_d4_opening_ess': required_d4_ess,
            'maximum_exact_opening_multiplicity': int(max_exact_frequency),
            'maximum_d4_opening_multiplicity': int(max_d4_frequency),
            'maximum_exact_opening_share': float(max_exact_frequency / game_count),
            'maximum_d4_opening_share': float(max_d4_frequency / game_count),
            'exact_opening_frequency_histogram': {
                str(frequency): int(count)
                for frequency, count in sorted(Counter(exact_frequencies.values()).items())
            },
            'd4_opening_frequency_histogram': {
                str(frequency): int(count)
                for frequency, count in sorted(Counter(d4_frequencies.values()).items())
            },
            'distinct_full_trajectories': int(len({
                game['full_trajectory_hash'] for game in game_level
            })),
            'literal_duplicate_full_trajectories': int(
                game_count - len(full_trajectory_frequencies)
            ),
            'maximum_full_trajectory_multiplicity': int(
                max(full_trajectory_frequencies.values())
            ),
            'full_trajectory_frequency_histogram': {
                str(frequency): int(count)
                for frequency, count in sorted(Counter(
                    full_trajectory_frequencies.values()
                ).items())
            },
            'distinct_continuation_trajectories': int(len({
                game['continuation_trajectory_hash'] for game in game_level
            })),
            'literal_duplicate_continuations': int(
                game_count - len({game['exact_continuation_key'] for game in game_level})
            ),
        },
        'decision_checks': {
            'raw_and_capped_point_direction_agree': bool(direction_agreement),
            'raw_and_capped_intervals_resolve_same_direction': bool(resolved_agreement),
            'natural_d4_diversity_pass': bool(natural_diversity_pass),
            'promotion_evidence_pass': bool(
                direction_agreement
                and resolved_agreement
                and natural_diversity_pass
                and cap_details['cap_achievable']
            ),
        },
        'pair_records': pair_records,
    }


def collect_d4_unique_placements(
    game,
    controller,
    search_args,
    target_openings,
    batch_size,
    seed,
    max_occurrences=1200,
    sample_batch_size=120,
    quiet=False,
):
    """Collect first-discovered D4 families from one learned placement policy."""
    target_openings = int(target_openings)
    max_occurrences = int(max_occurrences)
    sample_batch_size = int(sample_batch_size)
    if target_openings < 1:
        raise ValueError('The unique-opening target must be positive.')
    if max_occurrences < target_openings:
        raise ValueError('The occurrence budget cannot be below the target.')
    if sample_batch_size < 1:
        raise ValueError('The placement sample batch must be positive.')

    rng = np.random.RandomState(int(seed))
    selected = []
    selected_d4 = set()
    exact_frequencies = Counter()
    d4_frequencies = Counter()
    used_seeds = set()
    occurrences = 0
    inference = {'batches': 0, 'requested': 0, 'executed': 0, 'reused': 0}
    started = time.perf_counter()

    while len(selected) < target_openings and occurrences < max_occurrences:
        count = min(sample_batch_size, max_occurrences - occurrences)
        game_seeds = []
        while len(game_seeds) < count:
            candidate = int(rng.randint(0, 2 ** 31 - 1))
            if candidate not in used_seeds:
                used_seeds.add(candidate)
                game_seeds.append(candidate)
        arena = BatchedMCTSArena(
            game,
            controller,
            controller,
            search_args,
            batch_size=batch_size,
            quiet=quiet,
            placement_temperature=1.0,
            game_seeds=game_seeds,
        )
        records = arena.generatePlacements(
            count,
            desc='Collecting D4-unique learned placements',
            seat_order='source_self_play',
        )
        for key, value in arena.inferenceDiagnostics().items():
            if key in inference:
                inference[key] += int(value)
        for offset, record in enumerate(records):
            occurrence_index = occurrences + offset
            board = np.ascontiguousarray(record['opening_board'], dtype=np.int8)
            exact_key = _sha256_parts(board.tobytes())
            d4_key = _sha256_parts(record['symmetry_opening'])
            exact_frequencies[exact_key] += 1
            d4_frequencies[d4_key] += 1
            if d4_key in selected_d4 or len(selected) >= target_openings:
                continue
            selected_d4.add(d4_key)
            selected.append({
                'suite_index': int(len(selected)),
                'discovery_occurrence': int(occurrence_index),
                'game_seed': int(record['game_seed']),
                'opening_board': board.copy(),
                'opening': record['opening'],
                'exact_opening_key': exact_key,
                'd4_opening_key': d4_key,
                'placement_actions': tuple(record['placement_actions']),
            })
        occurrences += count

    if len(selected) < target_openings:
        raise RuntimeError(
            'Found only {} D4-unique placements in {} occurrences; requested {}.'.format(
                len(selected), occurrences, target_openings
            )
        )

    target_reached_at = int(selected[-1]['discovery_occurrence'] + 1)
    return selected, {
        'target_d4_unique_openings': int(target_openings),
        'sampled_occurrences': int(occurrences),
        'acceptance_rate': float(target_openings / occurrences),
        'target_reached_at_occurrence': target_reached_at,
        'selection_acceptance_rate': float(target_openings / target_reached_at),
        'unused_batch_tail_occurrences': int(occurrences - target_reached_at),
        'distinct_exact_openings_observed': int(len(exact_frequencies)),
        'distinct_d4_openings_observed': int(len(d4_frequencies)),
        'maximum_exact_opening_multiplicity': int(max(exact_frequencies.values())),
        'maximum_d4_opening_multiplicity': int(max(d4_frequencies.values())),
        'maximum_d4_opening_share': float(
            max(d4_frequencies.values()) / occurrences
        ),
        'd4_opening_ess': float(occurrences ** 2 / sum(
            frequency ** 2 for frequency in d4_frequencies.values()
        )),
        'elapsed_seconds': float(time.perf_counter() - started),
        'inference': inference,
    }


def run_duplicate_aware_matchup(
    game,
    contestant1,
    contestant2,
    search_args,
    games,
    batch_size,
    game_seeds,
    seed,
    quiet=False,
    bootstrap_samples=10_000,
    d4_cap_fraction=0.05,
    minimum_d4_ess_ratio=0.75,
):
    """Generate placements, adjudicate unique continuations, and score both views."""
    games = int(games)
    if games < 2 or games % 2:
        raise ValueError('Duplicate-aware matchup games must be positive and even.')
    if len(game_seeds) != games // 2:
        raise ValueError('Duplicate-aware matchup needs one seed per pair.')
    search_mode = str(search_args.get('searchMode', 'puct')).lower()
    if search_mode == 'gumbel' and float(search_args.get('gumbelScale', 1.0)) != 0.0:
        raise ValueError(
            'Exact-continuation reuse requires deterministic standard play '
            '(gumbelScale=0).'
        )

    placement_arena = BatchedMCTSArena(
        game,
        contestant1,
        contestant2,
        search_args,
        batch_size=batch_size,
        quiet=quiet,
        placement_temperature=1.0,
        game_seeds=game_seeds,
    )
    started = time.perf_counter()
    placements = placement_arena.generatePlacementGames(games)
    placement_elapsed = time.perf_counter() - started
    placement_inference = placement_arena.inferenceDiagnostics()

    unique_specs = []
    representative_by_key = {}
    for placement in placements:
        side = _side_signature(placement['side_to_player'])
        exact_opening_key = _sha256_parts(
            np.ascontiguousarray(placement['opening_board'], dtype=np.int8).tobytes()
        )
        continuation_key = _sha256_parts(exact_opening_key, side)
        placement['exact_opening_key'] = exact_opening_key
        placement['exact_continuation_key'] = continuation_key
        placement['d4_opening_key'] = _sha256_parts(placement['symmetry_opening'])
        if continuation_key in representative_by_key:
            continue
        specification_id = len(unique_specs)
        representative_by_key[continuation_key] = specification_id
        unique_specs.append({
            'specification_id': specification_id,
            'opening_board': placement['opening_board'],
            'side_to_player': placement['side_to_player'],
            'game_seed': placement['game_seed'],
            'game_index': placement['pair_index'],
            'seat_order': placement['seat_order'],
        })

    continuation_arena = BatchedMCTSArena(
        game,
        contestant1,
        contestant2,
        search_args,
        batch_size=batch_size,
        quiet=quiet,
        record_placement_diagnostics=True,
    )
    started = time.perf_counter()
    continuation_arena.playGameSpecifications(
        unique_specs, desc='BatchedArena unique continuations'
    )
    continuation_elapsed = time.perf_counter() - started
    results_by_id = {
        int(record['specification_id']): record
        for record in continuation_arena.game_records
    }
    if len(results_by_id) != len(unique_specs):
        raise ValueError('Unique continuation adjudication lost a specification.')

    serializable_records = []
    seen_continuations = set()
    for placement in placements:
        key = placement['exact_continuation_key']
        specification_id = representative_by_key[key]
        result = results_by_id[specification_id]
        standard_actions = tuple(result['standard_trajectory'])
        placement_actions = tuple(placement['placement_actions'])
        side = _side_signature(placement['side_to_player'])
        continuation_trajectory_hash = _sha256_parts(
            placement['exact_opening_key'], side, standard_actions
        )
        full_trajectory_hash = _sha256_parts(
            side, placement_actions, standard_actions
        )
        serializable_records.append({
            'pair_index': int(placement['pair_index']),
            'seat_order': placement['seat_order'],
            'game_seed': placement['game_seed'],
            'contestant1_side': next(
                int(physical_side)
                for physical_side, contestant in placement['side_to_player'].items()
                if contestant == 1
            ),
            'winner': int(result['winner']),
            'winner_side': int(result['winner_side']),
            'score': _current_score(result['winner']),
            'opening': placement['opening'],
            'exact_opening_key': placement['exact_opening_key'],
            'd4_opening_key': placement['d4_opening_key'],
            'exact_continuation_key': key,
            'continuation_cache_hit': key in seen_continuations,
            'placement_actions': list(placement_actions),
            'standard_trajectory': list(standard_actions),
            'continuation_trajectory_hash': continuation_trajectory_hash,
            'full_trajectory_hash': full_trajectory_hash,
        })
        seen_continuations.add(key)

    summary = summarize_duplicate_aware_records(
        serializable_records,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
        d4_cap_fraction=d4_cap_fraction,
        minimum_d4_ess_ratio=minimum_d4_ess_ratio,
    )
    summary['execution'] = {
        'placement_elapsed_seconds': float(placement_elapsed),
        'continuation_elapsed_seconds': float(continuation_elapsed),
        'total_elapsed_seconds': float(placement_elapsed + continuation_elapsed),
        'sampled_continuation_occurrences': int(games),
        'unique_continuations_executed': int(len(unique_specs)),
        'continuations_reused': int(games - len(unique_specs)),
        'placement_inference': placement_inference,
        'continuation_inference': continuation_arena.inferenceDiagnostics(),
    }
    return summary
