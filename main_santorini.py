import argparse
import logging
import os
import random

import numpy as np
import torch

try:
    import coloredlogs
except ImportError:
    coloredlogs = None

from Coach import Coach, preserve_rng_state
from santorini.pytorch.NNet import build_nnet
from santorini.pytorch.NNet import args as nnet_args
from santorini.pytorch.LegacyNNet import LegacyNNetWrapper
from santorini.SantoriniGame import SantoriniGame as Game
from santorini.SantoriniOpeningBook import (SantoriniOpeningSampler,
                                            SantoriniMixedOpeningSampler,
                                            SantoriniRandomOpeningSampler,
                                            SantoriniOpeningSuite,
                                            find_opening_book)
from utils import dotdict

log = logging.getLogger(__name__)

if coloredlogs is not None:
    coloredlogs.install(level='INFO')
else:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s[%(process)d] %(levelname)s %(message)s',
    )

DEFAULT_ARENA_OPENING_SUITE = './santorini/opening_suites/bootstrap_arena_suite.json'
DEFAULT_OPENING_BOOK = './santorini/opening_books/bootstrap_result/opening_book.json'
V3_DEFAULT_REPLAY_REUSE = 16.0
V3_DEFAULT_VALIDATION_FRACTION = 0.05
V3_DEFAULT_LEARNING_RATE = 3e-4
V3_DEFAULT_WEIGHT_DECAY = 1e-4
V3_DEFAULT_LR_SCHEDULE = [(200, 1e-4), (400, 3e-5)]
V3_DEFAULT_SYMMETRY_CONSISTENCY_FRACTION = 0.25
V3_DEFAULT_SYMMETRY_CONSISTENCY_POLICY_WEIGHT = 0.05
V3_DEFAULT_SYMMETRY_CONSISTENCY_VALUE_WEIGHT = 0.05

PRESETS = {
    'full': {
        'numIters': 1000,
        'numEps': 100,
        'tempThreshold': 15,
        'updateThreshold': 0.52,
        'maxlenOfQueue': 200000,
        'numMCTSSims': 50,
        'arenaCompare': 40,
        'checkpoint': './temp/santorini/',
        'numItersForTrainExamplesHistory': 20,
        'checkpointExamplesToKeep': 1,
        'deleteLoadedExamplesAfterFirstIteration': True,
        'atomicExamplesSave': False,
        'saveBestTrainExamples': False,
        'epochs': 10,
        'batch_size': 64,
    },
    'local': {
        'numIters': 10,
        'numEps': 10,
        'tempThreshold': 10,
        'updateThreshold': 0.52,
        'maxlenOfQueue': 50000,
        'numMCTSSims': 16,
        'arenaCompare': 10,
        'checkpoint': './temp/santorini_local/',
        'numItersForTrainExamplesHistory': 5,
        'checkpointExamplesToKeep': 1,
        'deleteLoadedExamplesAfterFirstIteration': True,
        'atomicExamplesSave': False,
        'saveBestTrainExamples': False,
        'epochs': 2,
        'batch_size': 64,
    },
}


def parse_lr_schedule(value):
    if value is None:
        return None
    value = value.strip()
    if not value or value.lower() in ('none', 'off'):
        return []
    schedule = []
    try:
        for item in value.split(','):
            iteration, learning_rate = item.split(':', 1)
            schedule.append((int(iteration), float(learning_rate)))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            'Expected ITERATION:RATE pairs, for example 200:0.0001,400:0.00003.'
        ) from exc
    if any(iteration < 1 or learning_rate <= 0 for iteration, learning_rate in schedule):
        raise argparse.ArgumentTypeError('Schedule iterations and learning rates must be positive.')
    if [iteration for iteration, _ in schedule] != sorted({iteration for iteration, _ in schedule}):
        raise argparse.ArgumentTypeError('Schedule iterations must be unique and increasing.')
    return schedule


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Santorini AlphaZero model.')
    parser.add_argument('--preset', choices=sorted(PRESETS.keys()), default='full')
    parser.add_argument('--architecture', choices=['v2', 'v3', 'v4'], default='v2')
    parser.add_argument('--training-mode', choices=['arena', 'latest'])
    parser.add_argument('--num-iters', type=int)
    parser.add_argument('--num-eps', type=int)
    parser.add_argument('--temp-threshold', type=int)
    parser.add_argument('--update-threshold', type=float)
    parser.add_argument('--maxlen-of-queue', type=int)
    parser.add_argument('--num-mcts-sims', type=int)
    parser.add_argument(
        '--search-mode',
        choices=['puct', 'gumbel'],
        default='puct',
        help='MCTS root/interior policy. Gumbel is experimental; PUCT remains the default.',
    )
    parser.add_argument('--gumbel-max-considered-actions', type=int, default=16)
    parser.add_argument('--gumbel-scale', type=float, default=1.0)
    parser.add_argument(
        '--gumbel-placement-scale',
        type=float,
        help='Gumbel scale for placement roots; defaults to --gumbel-scale.',
    )
    parser.add_argument(
        '--placement-scale-exploration-probability',
        type=float,
        default=0.0,
        help=(
            'Per-self-play-game probability of replacing --gumbel-placement-scale '
            'with --placement-exploration-gumbel-scale. Evaluation is unaffected.'
        ),
    )
    parser.add_argument(
        '--placement-exploration-gumbel-scale',
        type=float,
        help='Exploratory self-play placement scale; defaults to the base placement scale.',
    )
    parser.add_argument(
        '--evaluation-gumbel-scale',
        type=float,
        help='Standard-play Gumbel scale for milestone and anchor evaluation; defaults to self-play scale.',
    )
    parser.add_argument(
        '--evaluation-gumbel-placement-scale',
        type=float,
        help='Placement Gumbel scale for milestone evaluation; defaults to self-play placement scale.',
    )
    parser.add_argument(
        '--no-tactical-shortcuts',
        action='store_true',
        help='Disable exact immediate-win and one-ply forced-defense MCTS shortcuts.',
    )
    parser.add_argument(
        '--playout-cap-randomization',
        action='store_true',
        help=(
            'Randomly use full or fast MCTS searches during self-play. Only full-search '
            'positions are stored for training; fast turns disable root noise and action sampling.'
        ),
    )
    parser.add_argument('--playout-cap-full-probability', type=float, default=0.25)
    parser.add_argument(
        '--playout-cap-fast-sims',
        type=int,
        help='MCTS simulations on fast turns; defaults to one third of --num-mcts-sims.',
    )
    parser.add_argument(
        '--playout-cap-randomize-placement',
        action='store_true',
        help='Also randomize the four placement turns; by default they always receive full search.',
    )
    parser.add_argument('--arena-compare', type=int)
    parser.add_argument('--cpuct', type=float, default=1.0)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--load-folder', type=str)
    parser.add_argument('--load-file', type=str)
    parser.add_argument('--load-model', action='store_true')
    parser.add_argument('--load-examples', action='store_true')
    parser.add_argument(
        '--start-iteration',
        type=int,
        help='Override the last completed iteration stored in a resumed checkpoint.',
    )
    parser.add_argument('--examples-file', type=str)
    parser.add_argument('--skip-first-self-play', action='store_true')
    parser.add_argument('--history-iters', type=int)
    parser.add_argument(
        '--checkpoint-examples-to-keep',
        type=int,
        help='Keep only this many checkpoint_*.pth.tar.examples snapshots. latest.examples and best.pth.tar.examples are always kept.',
    )
    parser.add_argument(
        '--keep-loaded-examples',
        action='store_true',
        help='Do not delete the examples file loaded at startup after the first completed iteration.',
    )
    parser.add_argument(
        '--atomic-examples-save',
        action='store_true',
        help='Write replay examples through a .tmp file before replacing the target. Safer on crash, but briefly needs one extra full replay file of storage.',
    )
    parser.add_argument(
        '--save-best-examples',
        action='store_true',
        help='Also save best.pth.tar.examples. By default Santorini keeps latest.examples plus the retained checkpoint examples to save disk.',
    )
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument(
        '--max-train-steps',
        type=int,
        help=(
            'Safety cap on optimizer steps per iteration. V3 applies it after the '
            'fresh-data replay-reuse calculation; V2 applies it after its epoch calculation.'
        ),
    )
    parser.add_argument(
        '--replay-reuse',
        type=float,
        help=(
            'Requested training draws per newly generated training position. '
            'V3 defaults to 16; V2 retains epoch-based scheduling.'
        ),
    )
    parser.add_argument(
        '--replay-reuse-warmup-iters',
        type=int,
        default=0,
        help=(
            'Linearly ramp fresh-data replay reuse from 1/N of --replay-reuse '
            'at absolute iteration 1 to the full value at iteration N.'
        ),
    )
    parser.add_argument(
        '--validation-fraction',
        type=float,
        help='Deterministic held-out fraction of replay positions. V3 defaults to 0.05.',
    )
    parser.add_argument(
        '--placement-replay-fraction',
        type=float,
        help=(
            'V4 training-draw fraction reserved for placement positions after '
            'D4 aggregation; defaults to 0.15 for V4 and 0 otherwise.'
        ),
    )
    parser.add_argument(
        '--placement-replay-frequency-exponent',
        type=float,
        default=0.5,
        help=(
            'Exponent applied to an aggregated placement state occurrence count. '
            '0.5 is square-root weighting; 1.0 reproduces raw duplicate weight.'
        ),
    )
    parser.add_argument(
        '--unique-neural-start-fraction',
        type=float,
        help=(
            'Fraction of V4 ordinary games started from a within-iteration '
            'D4-unique opening generated by the current neural search; defaults to 0.10.'
        ),
    )
    parser.add_argument(
        '--unique-neural-start-max-attempt-factor',
        type=int,
        default=32,
        help='Maximum generated placement candidates per requested unique neural start.',
    )
    parser.add_argument('--optimizer', choices=['adam', 'adamw'])
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument(
        '--policy-head-learning-rate',
        type=float,
        help='V4 policy-head LR override; the shared trunk retains --learning-rate.',
    )
    parser.add_argument(
        '--value-head-learning-rate',
        type=float,
        help='V4 value-head LR override; the shared trunk retains --learning-rate.',
    )
    parser.add_argument('--weight-decay', type=float)
    parser.add_argument(
        '--lr-schedule',
        type=parse_lr_schedule,
        help='Absolute-iteration learning-rate changes as ITERATION:RATE pairs; use none to disable.',
    )
    parser.add_argument(
        '--value-target-anchor-checkpoint',
        type=str,
        help='Frozen V4 checkpoint used to bridge value targets toward outcome z.',
    )
    parser.add_argument('--value-target-beta-start', type=float, default=1.0)
    parser.add_argument('--value-target-beta-end', type=float, default=1.0)
    parser.add_argument('--value-target-beta-start-iteration', type=int, default=1)
    parser.add_argument('--value-target-beta-end-iteration', type=int, default=1)
    parser.add_argument('--value-target-anchor-batch-size', type=int, default=512)
    parser.add_argument(
        '--symmetry-augmentation',
        choices=['expanded', 'on-the-fly', 'none'],
        help=(
            'Store all eight symmetries, sample one per training draw, or store '
            'one position with no augmentation. V3 defaults to on-the-fly; V4 '
            'defaults to none because its trainable wrapper canonicalizes exactly.'
        ),
    )
    parser.add_argument(
        '--symmetry-consistency-fraction',
        type=float,
        help=(
            'Fraction of each optimizer batch paired with a second D4 orientation. '
            'V3 defaults to 0.25; use 0 to disable.'
        ),
    )
    parser.add_argument(
        '--symmetry-consistency-policy-weight',
        type=float,
        help='Weight for mapped-back policy Jensen-Shannon consistency; V3 defaults to 0.05.',
    )
    parser.add_argument(
        '--symmetry-consistency-value-weight',
        type=float,
        help='Weight for D4 value-consistency MSE; V3 defaults to 0.05.',
    )
    parser.add_argument(
        '--no-search-symmetry-evaluation',
        action='store_true',
        help=(
            'Disable random D4 neural evaluation inside MCTS. V3 enables it by default; '
            'this does not affect replay augmentation.'
        ),
    )
    parser.add_argument(
        '--root-symmetry-samples',
        type=int,
        help='Distinct D4 orientations averaged at standard self-play roots; V3 defaults to 2.',
    )
    parser.add_argument(
        '--placement-root-symmetry-samples',
        type=int,
        help='Distinct D4 orientations averaged at placement self-play roots; V3 defaults to 8.',
    )
    parser.add_argument(
        '--evaluation-root-symmetry-samples',
        type=int,
        help='Distinct D4 orientations averaged at standard evaluation roots; V3 defaults to 8.',
    )
    parser.add_argument(
        '--evaluation-placement-root-symmetry-samples',
        type=int,
        help='Distinct D4 orientations averaged at placement evaluation roots; V3 defaults to 8.',
    )
    parser.add_argument(
        '--no-inference-deduplication',
        action='store_true',
        help='Disable exact-board neural inference reuse during V3 batched search.',
    )
    parser.add_argument(
        '--inference-cache-size',
        type=int,
        help='Maximum exact-board predictions cached during one move; V3 defaults to 4096.',
    )
    parser.add_argument('--self-play-batch-size', type=int, default=1)
    parser.add_argument('--arena-batch-size', type=int)
    parser.add_argument(
        '--oracle-sparring-probability',
        type=float,
        default=0.0,
        help='Fraction of P2 games replaced by paired completed-opening oracle sparring.',
    )
    parser.add_argument('--oracle-sparring-nodes', type=int, default=5000)
    parser.add_argument('--oracle-sparring-workers', type=int, default=4)
    parser.add_argument('--oracle-sparring-opening-seed', type=int, default=20260921)
    parser.add_argument('--oracle-sparring-ladder-version', type=int, default=2)
    parser.add_argument(
        '--oracle-sparring-ratchet-games',
        type=int,
        default=80,
        help='Rolling paired-game window used to detect a stale oracle rung.',
    )
    parser.add_argument(
        '--oracle-sparring-ratchet-score',
        type=float,
        default=0.55,
        help='Pause after a checkpoint when V4 reaches this rolling sparring score.',
    )
    parser.add_argument(
        '--no-oracle-sparring-ratchet',
        action='store_true',
        help='Disable the rolling oracle-rung recalibration gate.',
    )
    parser.add_argument('--oracle-binary', type=str)
    parser.add_argument(
        '--opening-source',
        choices=['mixed', 'unique', 'book', 'game'],
        default='mixed',
        help='mixed samples old filtered book starts plus unique random placements; unique samples all symmetry-unique starting placements; book uses the old opening book sampler; game uses SantoriniGame.getInitBoard().',
    )
    parser.add_argument('--opening-book', type=str)
    parser.add_argument('--arena-opening-suite', type=str)
    parser.add_argument('--no-opening-book', action='store_true', help='Deprecated alias for --opening-source game.')
    parser.add_argument('--self-play-opening-max-abs-value', type=float, default=0.30)
    parser.add_argument('--self-play-old-filter-probability', type=float, default=1.00)
    parser.add_argument('--self-play-value-probability', type=float, default=0.00)
    parser.add_argument('--self-play-opening-tail-probability', type=float, default=0.00)
    parser.add_argument('--opening-mix-unique-probability', type=float, default=0.20)
    parser.add_argument('--arena-opening-top-fraction', type=float, default=0.50)
    parser.add_argument('--arena-opening-max-abs-value', type=float, default=0.14)
    parser.add_argument('--no-opening-random-orientation', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--placement-temperature', type=float, default=1.0)
    parser.add_argument(
        '--policy-target-temperature',
        type=float,
        help=(
            'Temperature applied to MCTS visits stored in replay, independently of action selection. '
            'V3 defaults to 1.0; V2 defaults to its action temperature for backward compatibility.'
        ),
    )
    parser.add_argument('--dirichlet-alpha', type=float, default=0.30)
    parser.add_argument('--dirichlet-epsilon', type=float, default=0.25)
    parser.add_argument('--no-dirichlet-noise', action='store_true')
    parser.add_argument('--compact-replay', action='store_true')
    parser.add_argument('--telemetry-dir', type=str)
    parser.add_argument('--milestone-interval', type=int)
    parser.add_argument('--reference-suite', type=str)
    parser.add_argument('--telemetry-match-games', type=int, default=40)
    parser.add_argument('--telemetry-match-batch-size', type=int)
    parser.add_argument(
        '--telemetry-placement-games',
        type=int,
        help='Placement-inclusive milestone games; defaults to --telemetry-match-games.',
    )
    parser.add_argument('--telemetry-placement-temperature', type=float, default=1.0)
    parser.add_argument('--telemetry-opening-seed', type=int, default=20260715)
    parser.add_argument('--no-telemetry-matches', action='store_true')
    parser.add_argument('--telemetry-sample-size', type=int, default=256)
    parser.add_argument(
        '--prior-target-kl-warning-threshold',
        type=float,
        default=0.15,
        help=(
            'Non-gating search-signal watch threshold for the per-iteration '
            'standard-play mean KL(search target || generating raw prior).'
        ),
    )
    parser.add_argument(
        '--prior-target-kl-warning-min-positions',
        type=int,
        default=256,
        help='Minimum measured standard positions before the KL watch is eligible.',
    )
    parser.add_argument(
        '--prior-target-kl-warning-iterations',
        type=int,
        default=3,
        help='Consecutive low-signal iterations needed to emit the KL warning.',
    )
    parser.add_argument(
        '--symmetry-telemetry-sample-size',
        type=int,
        help=(
            'Fixed held-out positions evaluated in all eight D4 orientations each iteration. '
            'V3 defaults to 64; use 0 to disable.'
        ),
    )
    parser.add_argument(
        '--v4-seam-telemetry-suite',
        type=str,
        help=(
            'Frozen P1c seam-suite .npz. When set, report exposure-quartile '
            'losses and their high-minus-low regression during P2.'
        ),
    )
    parser.add_argument('--v4-seam-telemetry-interval', type=int, default=1)
    parser.add_argument('--v4-seam-telemetry-batch-size', type=int, default=256)
    parser.add_argument('--v4-seam-telemetry-bootstrap-samples', type=int, default=2000)
    parser.add_argument('--v4-seam-telemetry-seed', type=int, default=20260818)
    parser.add_argument(
        '--v4-seam-telemetry-alert-delta',
        type=float,
        default=0.02,
        help=(
            'Warn when the Q4-minus-Q1 objective contrast has increased this '
            'much from the frozen P1c baseline.'
        ),
    )
    parser.add_argument(
        '--v4-teacher-objective-step-threshold',
        type=float,
        default=0.05,
        help=(
            'Pause after a resumable checkpoint when the frozen teacher '
            'objective rises by more than this amount versus the preceding '
            'iteration (the P1c baseline for iteration one).'
        ),
    )
    parser.add_argument(
        '--v4-teacher-objective-cumulative-threshold',
        type=float,
        help=(
            'Pause for review when the frozen teacher objective rises this much '
            'from the branch-start reference. Disabled when omitted.'
        ),
    )
    parser.add_argument(
        '--no-v4-teacher-objective-gate',
        action='store_true',
        help='Report but do not pause on frozen teacher-objective step increases.',
    )
    parser.add_argument(
        '--v4-deep-value-telemetry-suite',
        help=(
            'Frozen deep-oracle value suite with iteration-11 reference '
            'predictions. Evaluated after every V4 latest-mode iteration.'
        ),
    )
    parser.add_argument('--v4-deep-value-telemetry-batch-size', type=int, default=256)
    parser.add_argument(
        '--v4-deep-value-telemetry-bootstrap-samples', type=int, default=2000
    )
    parser.add_argument('--v4-deep-value-telemetry-seed', type=int, default=20260814)
    parser.add_argument('--v4-deep-value-overall-pearson-drop', type=float, default=0.02)
    parser.add_argument('--v4-deep-value-overall-mse-rise', type=float, default=0.015)
    parser.add_argument('--v4-deep-value-recent-pearson-drop', type=float, default=0.04)
    parser.add_argument('--v4-deep-value-recent-mse-rise', type=float, default=0.02)
    parser.add_argument('--v4-deep-value-warning-iterations', type=int, default=2)
    parser.add_argument(
        '--no-v4-deep-value-gate', action='store_true',
        help='Report frozen deep-value drift without pausing after confirmation.',
    )
    parser.add_argument(
        '--anchor-checkpoint',
        type=str,
        help='Exact checkpoint file, or a directory containing one, for fixed-opponent telemetry.',
    )
    parser.add_argument('--anchor-architecture', choices=['v1', 'v2', 'v3', 'v4'], default='v1')
    parser.add_argument('--anchor-interval', type=int, default=10)
    parser.add_argument('--anchor-games', type=int, default=40)
    parser.add_argument('--anchor-placement-games', type=int, default=0)
    parser.add_argument('--anchor-mcts-sims', type=int)
    args = parser.parse_args()
    if args.opening_mix_unique_probability < 0.0 or args.opening_mix_unique_probability > 1.0:
        parser.error('--opening-mix-unique-probability must be between 0 and 1.')
    placement_replay_fraction = (
        args.placement_replay_fraction
        if args.placement_replay_fraction is not None
        else (0.15 if args.architecture == 'v4' else 0.0)
    )
    if not 0.0 <= placement_replay_fraction < 1.0:
        parser.error('--placement-replay-fraction must be in [0, 1).')
    if not 0.0 <= args.placement_replay_frequency_exponent <= 1.0:
        parser.error('--placement-replay-frequency-exponent must be in [0, 1].')
    unique_neural_start_fraction = (
        args.unique_neural_start_fraction
        if args.unique_neural_start_fraction is not None
        else (0.10 if args.architecture == 'v4' else 0.0)
    )
    if not 0.0 <= unique_neural_start_fraction <= 1.0:
        parser.error('--unique-neural-start-fraction must be between 0 and 1.')
    if args.unique_neural_start_max_attempt_factor < 1:
        parser.error('--unique-neural-start-max-attempt-factor must be at least 1.')
    if unique_neural_start_fraction > 0 and args.architecture != 'v4':
        parser.error('--unique-neural-start-fraction is currently supported only for V4.')
    if not 0.0 <= args.oracle_sparring_probability <= 1.0:
        parser.error('--oracle-sparring-probability must be between 0 and 1.')
    for option, value in (
        ('--oracle-sparring-nodes', args.oracle_sparring_nodes),
        ('--oracle-sparring-workers', args.oracle_sparring_workers),
        ('--oracle-sparring-ladder-version', args.oracle_sparring_ladder_version),
    ):
        if value < 1:
            parser.error('{} must be at least 1.'.format(option))
    if args.oracle_sparring_ratchet_games < 2 or args.oracle_sparring_ratchet_games % 2:
        parser.error('--oracle-sparring-ratchet-games must be a positive even number.')
    if not 0.0 < args.oracle_sparring_ratchet_score <= 1.0:
        parser.error('--oracle-sparring-ratchet-score must be in (0, 1].')
    if args.oracle_sparring_probability > 0 and not args.oracle_binary:
        parser.error('--oracle-binary is required when oracle sparring is enabled.')
    if args.oracle_sparring_probability > 0 and args.architecture != 'v4':
        parser.error('--oracle-sparring-probability is currently supported only for V4.')
    if args.start_iteration is not None and args.start_iteration < 0:
        parser.error('--start-iteration cannot be negative.')
    if args.start_iteration is not None and not args.load_model:
        parser.error('--start-iteration requires --load-model.')
    for option, value in (
        ('--telemetry-match-games', args.telemetry_match_games),
        ('--telemetry-placement-games', args.telemetry_placement_games),
        ('--anchor-games', args.anchor_games if args.anchor_checkpoint else None),
        (
            '--anchor-placement-games',
            args.anchor_placement_games if args.anchor_checkpoint else None,
        ),
    ):
        if value is not None and (value < 0 or value % 2):
            parser.error('{} must be a non-negative even number.'.format(option))
    if args.telemetry_placement_temperature < 0:
        parser.error('--telemetry-placement-temperature cannot be negative.')
    if args.prior_target_kl_warning_threshold < 0:
        parser.error('--prior-target-kl-warning-threshold cannot be negative.')
    if args.prior_target_kl_warning_min_positions < 1:
        parser.error('--prior-target-kl-warning-min-positions must be positive.')
    if args.prior_target_kl_warning_iterations < 1:
        parser.error('--prior-target-kl-warning-iterations must be positive.')
    if (
        args.symmetry_telemetry_sample_size is not None
        and args.symmetry_telemetry_sample_size < 0
    ):
        parser.error('--symmetry-telemetry-sample-size cannot be negative.')
    for option, value in (
        ('--v4-seam-telemetry-interval', args.v4_seam_telemetry_interval),
        ('--v4-seam-telemetry-batch-size', args.v4_seam_telemetry_batch_size),
        (
            '--v4-seam-telemetry-bootstrap-samples',
            args.v4_seam_telemetry_bootstrap_samples,
        ),
    ):
        if value < 1:
            parser.error('{} must be at least 1.'.format(option))
    if args.v4_seam_telemetry_alert_delta < 0:
        parser.error('--v4-seam-telemetry-alert-delta cannot be negative.')
    if args.v4_teacher_objective_step_threshold < 0:
        parser.error('--v4-teacher-objective-step-threshold cannot be negative.')
    if (
        args.v4_teacher_objective_cumulative_threshold is not None
        and args.v4_teacher_objective_cumulative_threshold < 0
    ):
        parser.error('--v4-teacher-objective-cumulative-threshold cannot be negative.')
    if (
        args.v4_seam_telemetry_suite
        and not args.no_v4_teacher_objective_gate
        and args.v4_seam_telemetry_interval != 1
    ):
        parser.error(
            '--v4-teacher-objective gate requires '
            '--v4-seam-telemetry-interval 1.'
        )
    for option, value in (
        ('--v4-deep-value-telemetry-batch-size', args.v4_deep_value_telemetry_batch_size),
        (
            '--v4-deep-value-telemetry-bootstrap-samples',
            args.v4_deep_value_telemetry_bootstrap_samples,
        ),
        ('--v4-deep-value-warning-iterations', args.v4_deep_value_warning_iterations),
    ):
        if value < 1:
            parser.error('{} must be positive.'.format(option))
    for option, value in (
        ('--v4-deep-value-overall-pearson-drop', args.v4_deep_value_overall_pearson_drop),
        ('--v4-deep-value-overall-mse-rise', args.v4_deep_value_overall_mse_rise),
        ('--v4-deep-value-recent-pearson-drop', args.v4_deep_value_recent_pearson_drop),
        ('--v4-deep-value-recent-mse-rise', args.v4_deep_value_recent_mse_rise),
    ):
        if value < 0:
            parser.error('{} cannot be negative.'.format(option))
    if args.replay_reuse_warmup_iters < 0:
        parser.error('--replay-reuse-warmup-iters cannot be negative.')
    for option, value in (
        ('--learning-rate', args.learning_rate),
        ('--policy-head-learning-rate', args.policy_head_learning_rate),
        ('--value-head-learning-rate', args.value_head_learning_rate),
    ):
        if value is not None and value <= 0:
            parser.error('{} must be positive.'.format(option))
    if (
        args.policy_head_learning_rate is not None
        or args.value_head_learning_rate is not None
    ) and args.architecture != 'v4':
        parser.error('Differential head learning rates are supported only for V4.')
    if args.value_target_anchor_checkpoint and args.architecture != 'v4':
        parser.error('--value-target-anchor-checkpoint is supported only for V4.')
    for option, value in (
        ('--value-target-beta-start', args.value_target_beta_start),
        ('--value-target-beta-end', args.value_target_beta_end),
    ):
        if not 0.0 <= value <= 1.0:
            parser.error('{} must be between zero and one.'.format(option))
    if args.value_target_beta_start_iteration < 1:
        parser.error('--value-target-beta-start-iteration must be positive.')
    if args.value_target_beta_end_iteration < args.value_target_beta_start_iteration:
        parser.error('--value-target-beta-end-iteration cannot precede its start.')
    if args.value_target_anchor_batch_size < 1:
        parser.error('--value-target-anchor-batch-size must be positive.')
    if not args.value_target_anchor_checkpoint and (
        args.value_target_beta_start != 1.0 or args.value_target_beta_end != 1.0
    ):
        parser.error('A beta bridge requires --value-target-anchor-checkpoint.')
    if args.inference_cache_size is not None and args.inference_cache_size < 0:
        parser.error('--inference-cache-size cannot be negative.')
    if args.policy_target_temperature is not None and args.policy_target_temperature < 0:
        parser.error('--policy-target-temperature cannot be negative.')
    if args.anchor_interval < 1:
        parser.error('--anchor-interval must be at least 1.')
    if args.milestone_interval is not None and args.milestone_interval < 1:
        parser.error('--milestone-interval must be at least 1.')
    if args.anchor_mcts_sims is not None and args.anchor_mcts_sims < 1:
        parser.error('--anchor-mcts-sims must be at least 1.')
    if args.gumbel_max_considered_actions < 1:
        parser.error('--gumbel-max-considered-actions must be at least 1.')
    if args.gumbel_scale < 0:
        parser.error('--gumbel-scale cannot be negative.')
    if args.gumbel_placement_scale is not None and args.gumbel_placement_scale < 0:
        parser.error('--gumbel-placement-scale cannot be negative.')
    if not 0 <= args.placement_scale_exploration_probability <= 1:
        parser.error('--placement-scale-exploration-probability must be between 0 and 1.')
    if (
        args.placement_exploration_gumbel_scale is not None
        and args.placement_exploration_gumbel_scale < 0
    ):
        parser.error('--placement-exploration-gumbel-scale cannot be negative.')
    if args.evaluation_gumbel_scale is not None and args.evaluation_gumbel_scale < 0:
        parser.error('--evaluation-gumbel-scale cannot be negative.')
    if (
        args.evaluation_gumbel_placement_scale is not None
        and args.evaluation_gumbel_placement_scale < 0
    ):
        parser.error('--evaluation-gumbel-placement-scale cannot be negative.')
    if args.max_train_steps is not None and args.max_train_steps < 1:
        parser.error('--max-train-steps must be at least 1.')
    if args.replay_reuse is not None and args.replay_reuse <= 0:
        parser.error('--replay-reuse must be positive.')
    if args.validation_fraction is not None and not 0 <= args.validation_fraction < 0.5:
        parser.error('--validation-fraction must be at least 0 and less than 0.5.')
    if args.learning_rate is not None and args.learning_rate <= 0:
        parser.error('--learning-rate must be positive.')
    if args.weight_decay is not None and args.weight_decay < 0:
        parser.error('--weight-decay cannot be negative.')
    if (
        args.symmetry_consistency_fraction is not None
        and not 0 <= args.symmetry_consistency_fraction <= 1
    ):
        parser.error('--symmetry-consistency-fraction must be between 0 and 1.')
    for option, value in (
        (
            '--symmetry-consistency-policy-weight',
            args.symmetry_consistency_policy_weight,
        ),
        (
            '--symmetry-consistency-value-weight',
            args.symmetry_consistency_value_weight,
        ),
    ):
        if value is not None and value < 0:
            parser.error('{} cannot be negative.'.format(option))
    for option, value in (
        ('--root-symmetry-samples', args.root_symmetry_samples),
        ('--placement-root-symmetry-samples', args.placement_root_symmetry_samples),
        ('--evaluation-root-symmetry-samples', args.evaluation_root_symmetry_samples),
        (
            '--evaluation-placement-root-symmetry-samples',
            args.evaluation_placement_root_symmetry_samples,
        ),
    ):
        if value is not None and not 1 <= value <= 8:
            parser.error('{} must be between 1 and 8.'.format(option))
    if not 0 < args.playout_cap_full_probability <= 1:
        parser.error('--playout-cap-full-probability must be greater than 0 and at most 1.')
    if args.playout_cap_randomization:
        full_sims = args.num_mcts_sims or PRESETS[args.preset]['numMCTSSims']
        fast_sims = args.playout_cap_fast_sims or max(1, full_sims // 3)
        if fast_sims >= full_sims:
            parser.error('--playout-cap-fast-sims must be less than the full MCTS simulation count.')
    return args


def build_coach_args(parsed_args):
    preset = PRESETS[parsed_args.preset]
    checkpoint = parsed_args.checkpoint or preset['checkpoint']
    load_folder = parsed_args.load_folder or checkpoint
    arena_batch_size = parsed_args.arena_batch_size or parsed_args.self_play_batch_size

    modern_architecture = parsed_args.architecture in ('v3', 'v4')
    training_mode = getattr(parsed_args, 'training_mode', None) or (
        'latest' if modern_architecture else 'arena'
    )
    symmetry_augmentation = getattr(parsed_args, 'symmetry_augmentation', None) or (
        'none' if parsed_args.architecture == 'v4'
        else 'on-the-fly' if parsed_args.architecture == 'v3'
        else 'expanded'
    )
    policy_target_temperature = getattr(parsed_args, 'policy_target_temperature', None)
    if policy_target_temperature is None and modern_architecture:
        policy_target_temperature = 1.0
    full_mcts_sims = parsed_args.num_mcts_sims or preset['numMCTSSims']
    playout_cap_fast_sims = (
        parsed_args.playout_cap_fast_sims
        if parsed_args.playout_cap_fast_sims is not None
        else max(1, full_mcts_sims // 3)
    )
    load_file = getattr(parsed_args, 'load_file', None) or (
        'latest-training.pth.tar' if training_mode == 'latest' else 'best.pth.tar'
    )
    return dotdict({
        'numIters': parsed_args.num_iters or preset['numIters'],
        'numEps': parsed_args.num_eps or preset['numEps'],
        'tempThreshold': parsed_args.temp_threshold or preset['tempThreshold'],
        'updateThreshold': parsed_args.update_threshold or preset['updateThreshold'],
        'maxlenOfQueue': parsed_args.maxlen_of_queue or preset['maxlenOfQueue'],
        'numMCTSSims': parsed_args.num_mcts_sims or preset['numMCTSSims'],
        'searchMode': getattr(parsed_args, 'search_mode', 'puct'),
        'gumbelMaxConsideredActions': getattr(
            parsed_args, 'gumbel_max_considered_actions', 16
        ),
        'gumbelScale': getattr(parsed_args, 'gumbel_scale', 1.0),
        'gumbelPlacementScale': (
            parsed_args.gumbel_placement_scale
            if getattr(parsed_args, 'gumbel_placement_scale', None) is not None
            else getattr(parsed_args, 'gumbel_scale', 1.0)
        ),
        'placementScaleExplorationProbability': getattr(
            parsed_args, 'placement_scale_exploration_probability', 0.0
        ),
        'placementExplorationGumbelScale': (
            parsed_args.placement_exploration_gumbel_scale
            if getattr(parsed_args, 'placement_exploration_gumbel_scale', None) is not None
            else (
                parsed_args.gumbel_placement_scale
                if getattr(parsed_args, 'gumbel_placement_scale', None) is not None
                else getattr(parsed_args, 'gumbel_scale', 1.0)
            )
        ),
        'evaluationGumbelScale': (
            parsed_args.evaluation_gumbel_scale
            if getattr(parsed_args, 'evaluation_gumbel_scale', None) is not None
            else getattr(parsed_args, 'gumbel_scale', 1.0)
        ),
        'evaluationGumbelPlacementScale': (
            parsed_args.evaluation_gumbel_placement_scale
            if getattr(parsed_args, 'evaluation_gumbel_placement_scale', None) is not None
            else (
                parsed_args.gumbel_placement_scale
                if getattr(parsed_args, 'gumbel_placement_scale', None) is not None
                else getattr(parsed_args, 'gumbel_scale', 1.0)
            )
        ),
        'arenaCompare': parsed_args.arena_compare or preset['arenaCompare'],
        'cpuct': parsed_args.cpuct,
        'checkpoint': checkpoint,
        'load_model': parsed_args.load_model,
        'load_folder_file': (load_folder, load_file),
        'numItersForTrainExamplesHistory': parsed_args.history_iters or preset['numItersForTrainExamplesHistory'],
        'checkpointExamplesToKeep': (
            parsed_args.checkpoint_examples_to_keep
            if parsed_args.checkpoint_examples_to_keep is not None
            else (0 if training_mode == 'latest' else preset['checkpointExamplesToKeep'])
        ),
        'deleteLoadedExamplesAfterFirstIteration': (
            preset['deleteLoadedExamplesAfterFirstIteration']
            and not parsed_args.keep_loaded_examples
        ),
        'atomicExamplesSave': parsed_args.atomic_examples_save or preset['atomicExamplesSave'],
        'saveBestTrainExamples': parsed_args.save_best_examples or preset['saveBestTrainExamples'],
        'selfPlayBatchSize': parsed_args.self_play_batch_size,
        'arenaBatchSize': arena_batch_size,
        'oracleSparringProbability': getattr(
            parsed_args, 'oracle_sparring_probability', 0.0
        ),
        'oracleSparringNodes': getattr(parsed_args, 'oracle_sparring_nodes', 5_000),
        'oracleSparringWorkers': getattr(parsed_args, 'oracle_sparring_workers', 4),
        'oracleSparringOpeningSeed': getattr(
            parsed_args, 'oracle_sparring_opening_seed', 20260921
        ),
        'oracleSparringLadderVersion': getattr(
            parsed_args, 'oracle_sparring_ladder_version', 2
        ),
        'oracleSparringRatchetGames': getattr(
            parsed_args, 'oracle_sparring_ratchet_games', 80
        ),
        'oracleSparringRatchetScore': getattr(
            parsed_args, 'oracle_sparring_ratchet_score', 0.55
        ),
        'oracleSparringRatchetEnabled': not getattr(
            parsed_args, 'no_oracle_sparring_ratchet', False
        ),
        'oracleBinary': getattr(parsed_args, 'oracle_binary', None),
        'quiet': parsed_args.quiet,
        'trainingMode': training_mode,
        'placementTemperature': getattr(parsed_args, 'placement_temperature', 1.0),
        'policyTargetTemperature': policy_target_temperature,
        'playoutCapRandomization': getattr(parsed_args, 'playout_cap_randomization', False),
        'playoutCapFullProbability': getattr(parsed_args, 'playout_cap_full_probability', 0.25),
        'playoutCapFastSims': playout_cap_fast_sims,
        'playoutCapFullPlacement': not getattr(
            parsed_args,
            'playout_cap_randomize_placement',
            False,
        ),
        'addDirichletNoise': (
            training_mode == 'latest' and not getattr(parsed_args, 'no_dirichlet_noise', False)
        ),
        'dirichletAlpha': getattr(parsed_args, 'dirichlet_alpha', 0.30),
        'dirichletEpsilon': getattr(parsed_args, 'dirichlet_epsilon', 0.25),
        'tacticalShortcuts': not getattr(parsed_args, 'no_tactical_shortcuts', False),
        'compactReplay': getattr(parsed_args, 'compact_replay', False) or training_mode == 'latest',
        'symmetryAugmentation': symmetry_augmentation,
        'searchSymmetryEvaluation': (
            parsed_args.architecture == 'v3'
            and not getattr(parsed_args, 'no_search_symmetry_evaluation', False)
        ),
        'inferenceDeduplication': (
            modern_architecture
            and not getattr(parsed_args, 'no_inference_deduplication', False)
        ),
        'inferenceCacheSize': (
            parsed_args.inference_cache_size
            if parsed_args.inference_cache_size is not None
            else (4096 if modern_architecture else 0)
        ),
        'rootSymmetrySamples': (
            parsed_args.root_symmetry_samples
            if parsed_args.root_symmetry_samples is not None
            else (2 if parsed_args.architecture == 'v3' else 1)
        ),
        'placementRootSymmetrySamples': (
            parsed_args.placement_root_symmetry_samples
            if parsed_args.placement_root_symmetry_samples is not None
            else (8 if parsed_args.architecture == 'v3' else 1)
        ),
        'evaluationRootSymmetrySamples': (
            parsed_args.evaluation_root_symmetry_samples
            if parsed_args.evaluation_root_symmetry_samples is not None
            else (8 if parsed_args.architecture == 'v3' else 1)
        ),
        'evaluationPlacementRootSymmetrySamples': (
            parsed_args.evaluation_placement_root_symmetry_samples
            if parsed_args.evaluation_placement_root_symmetry_samples is not None
            else (8 if parsed_args.architecture == 'v3' else 1)
        ),
        'replayReuse': (
            parsed_args.replay_reuse
            if parsed_args.replay_reuse is not None
            else (V3_DEFAULT_REPLAY_REUSE if modern_architecture else None)
        ),
        'validationFraction': (
            parsed_args.validation_fraction
            if parsed_args.validation_fraction is not None
            else (V3_DEFAULT_VALIDATION_FRACTION if modern_architecture else 0.0)
        ),
        'placementReplayFraction': (
            parsed_args.placement_replay_fraction
            if parsed_args.placement_replay_fraction is not None
            else (0.15 if parsed_args.architecture == 'v4' else 0.0)
        ),
        'placementReplayFrequencyExponent': (
            parsed_args.placement_replay_frequency_exponent
        ),
        'uniqueNeuralStartFraction': (
            parsed_args.unique_neural_start_fraction
            if parsed_args.unique_neural_start_fraction is not None
            else (0.10 if parsed_args.architecture == 'v4' else 0.0)
        ),
        'uniqueNeuralStartMaxAttemptFactor': (
            parsed_args.unique_neural_start_max_attempt_factor
        ),
        'telemetryDir': getattr(parsed_args, 'telemetry_dir', None) or os.path.join(checkpoint, 'telemetry'),
        'milestoneInterval': (
            getattr(parsed_args, 'milestone_interval', None)
            or (20 if modern_architecture else 10)
        ),
        'referenceSuite': getattr(parsed_args, 'reference_suite', None),
        'telemetryMatchGames': (
            0 if getattr(parsed_args, 'no_telemetry_matches', False)
            else getattr(parsed_args, 'telemetry_match_games', 40)
        ),
        'telemetryMatchBatchSize': (
            getattr(parsed_args, 'telemetry_match_batch_size', None)
            or parsed_args.self_play_batch_size
        ),
        'telemetryPlacementGames': (
            0 if getattr(parsed_args, 'no_telemetry_matches', False)
            else (
                getattr(parsed_args, 'telemetry_placement_games', None)
                if getattr(parsed_args, 'telemetry_placement_games', None) is not None
                else getattr(parsed_args, 'telemetry_match_games', 40)
            )
        ),
        'telemetryPlacementTemperature': getattr(parsed_args, 'telemetry_placement_temperature', 1.0),
        'telemetryOpeningSeed': getattr(parsed_args, 'telemetry_opening_seed', 20260715),
        'anchorInterval': getattr(parsed_args, 'anchor_interval', 10),
        'anchorGames': getattr(parsed_args, 'anchor_games', 40),
        'anchorPlacementGames': getattr(parsed_args, 'anchor_placement_games', 0),
        'anchorMCTSSims': (
            getattr(parsed_args, 'anchor_mcts_sims', None)
            or parsed_args.num_mcts_sims
            or preset['numMCTSSims']
        ),
        'anchorArchitecture': getattr(parsed_args, 'anchor_architecture', 'v1'),
        'startIteration': 0,
        'telemetrySampleSize': getattr(parsed_args, 'telemetry_sample_size', 256),
        'priorTargetKLWarningThreshold': getattr(
            parsed_args, 'prior_target_kl_warning_threshold', 0.15
        ),
        'priorTargetKLWarningMinPositions': getattr(
            parsed_args, 'prior_target_kl_warning_min_positions', 256
        ),
        'priorTargetKLWarningIterations': getattr(
            parsed_args, 'prior_target_kl_warning_iterations', 3
        ),
        'symmetryTelemetrySampleSize': (
            parsed_args.symmetry_telemetry_sample_size
            if parsed_args.symmetry_telemetry_sample_size is not None
            else (64 if parsed_args.architecture == 'v3' else 0)
        ),
        'v4SeamTelemetrySuite': getattr(parsed_args, 'v4_seam_telemetry_suite', None),
        'v4SeamTelemetryInterval': getattr(
            parsed_args, 'v4_seam_telemetry_interval', 1
        ),
        'v4SeamTelemetryBatchSize': getattr(
            parsed_args, 'v4_seam_telemetry_batch_size', 256
        ),
        'v4SeamTelemetryBootstrapSamples': getattr(
            parsed_args, 'v4_seam_telemetry_bootstrap_samples', 2000
        ),
        'v4SeamTelemetrySeed': getattr(
            parsed_args, 'v4_seam_telemetry_seed', 20260818
        ),
        'v4SeamTelemetryAlertDelta': getattr(
            parsed_args, 'v4_seam_telemetry_alert_delta', 0.02
        ),
        'v4TeacherObjectiveStepThreshold': getattr(
            parsed_args, 'v4_teacher_objective_step_threshold', 0.05
        ),
        'v4TeacherObjectiveCumulativeThreshold': getattr(
            parsed_args, 'v4_teacher_objective_cumulative_threshold', None
        ),
        'v4TeacherObjectiveGateEnabled': not getattr(
            parsed_args, 'no_v4_teacher_objective_gate', False
        ),
        'v4DeepValueTelemetrySuite': getattr(
            parsed_args, 'v4_deep_value_telemetry_suite', None
        ),
        'v4DeepValueTelemetryBatchSize': getattr(
            parsed_args, 'v4_deep_value_telemetry_batch_size', 256
        ),
        'v4DeepValueTelemetryBootstrapSamples': getattr(
            parsed_args, 'v4_deep_value_telemetry_bootstrap_samples', 2000
        ),
        'v4DeepValueTelemetrySeed': getattr(
            parsed_args, 'v4_deep_value_telemetry_seed', 20260814
        ),
        'v4DeepValueOverallPearsonDrop': getattr(
            parsed_args, 'v4_deep_value_overall_pearson_drop', 0.02
        ),
        'v4DeepValueOverallMseRise': getattr(
            parsed_args, 'v4_deep_value_overall_mse_rise', 0.015
        ),
        'v4DeepValueRecentPearsonDrop': getattr(
            parsed_args, 'v4_deep_value_recent_pearson_drop', 0.04
        ),
        'v4DeepValueRecentMseRise': getattr(
            parsed_args, 'v4_deep_value_recent_mse_rise', 0.02
        ),
        'v4DeepValueWarningIterations': getattr(
            parsed_args, 'v4_deep_value_warning_iterations', 2
        ),
        'v4DeepValueGateEnabled': not getattr(
            parsed_args, 'no_v4_deep_value_gate', False
        ),
    })


def resolve_anchor_checkpoint_path(path):
    path = os.path.abspath(os.path.expanduser(os.fspath(path)))
    if os.path.isfile(path):
        return path
    if not os.path.exists(path):
        raise FileNotFoundError('Anchor checkpoint does not exist: {}'.format(path))
    if not os.path.isdir(path):
        raise ValueError('Anchor checkpoint path is neither a file nor a directory: {}'.format(path))

    candidates = []
    for root, _, filenames in os.walk(path):
        candidates.extend(
            os.path.join(root, filename)
            for filename in filenames
            if filename.endswith('.pth.tar')
        )
    candidates.sort()
    preferred = [candidate for candidate in candidates if os.path.basename(candidate) == 'best.pth.tar']
    if len(preferred) == 1:
        return preferred[0]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError('No .pth.tar checkpoint found under anchor directory: {}'.format(path))
    raise ValueError(
        'Multiple anchor checkpoints found; pass the exact file path: {}'.format(', '.join(candidates))
    )


def build_anchor_nnet(game, architecture, checkpoint_path):
    anchor = LegacyNNetWrapper(game) if architecture == 'v1' else build_nnet(game, architecture)
    anchor.load_checkpoint(os.path.dirname(checkpoint_path), os.path.basename(checkpoint_path))
    return anchor


def opening_book_candidates(parsed_args, coach_args):
    checkpoint = coach_args.checkpoint
    load_folder = coach_args.load_folder_file[0]
    checkpoint_name = os.path.basename(os.path.normpath(checkpoint))
    load_folder_name = os.path.basename(os.path.normpath(load_folder))
    return [
        parsed_args.opening_book,
        DEFAULT_OPENING_BOOK,
        os.path.join(checkpoint, 'opening_book.json'),
        os.path.join(checkpoint, 'opening_books', 'opening_book.json'),
        os.path.join(os.path.dirname(checkpoint), 'opening_books', checkpoint_name, 'opening_book.json'),
        os.path.join('temp', 'santorini_opening_books', checkpoint_name, 'opening_book.json'),
        os.path.join('temp', 'santorini_opening_books', load_folder_name, 'opening_book.json'),
    ]


def build_book_opening_sampler(parsed_args, coach_args):
    opening_book = find_opening_book(opening_book_candidates(parsed_args, coach_args))
    if opening_book is None:
        if parsed_args.opening_book:
            raise FileNotFoundError("No opening book found at {}".format(parsed_args.opening_book))
        log.warning('No opening book found for book-filtered starts.')
        return None

    arena_suite = None
    if parsed_args.arena_opening_suite:
        if not os.path.isfile(parsed_args.arena_opening_suite):
            raise FileNotFoundError("No arena opening suite found at {}".format(parsed_args.arena_opening_suite))
        arena_suite = SantoriniOpeningSuite.load(parsed_args.arena_opening_suite)

    sampler = SantoriniOpeningSampler.load(
        opening_book,
        arena_suite=arena_suite,
        self_play_max_abs_value=parsed_args.self_play_opening_max_abs_value,
        self_play_old_filter_probability=parsed_args.self_play_old_filter_probability,
        self_play_value_probability=parsed_args.self_play_value_probability,
        self_play_tail_probability=parsed_args.self_play_opening_tail_probability,
        arena_top_fraction=parsed_args.arena_opening_top_fraction,
        arena_max_abs_value=parsed_args.arena_opening_max_abs_value,
        random_orientation=not parsed_args.no_opening_random_orientation,
    )
    log.info(
        'Loaded opening book "%s" (%s positions, %s arena candidates)',
        opening_book,
        len(sampler.book.positions),
        len(sampler._arena_candidates()),
    )
    if arena_suite is not None:
        log.info('Loaded arena opening suite "%s" (%s positions)', parsed_args.arena_opening_suite, len(arena_suite.positions))
    return sampler


def build_unique_opening_sampler(parsed_args):
    sampler = SantoriniRandomOpeningSampler(
        board_size=5,
        random_orientation=not parsed_args.no_opening_random_orientation,
    )
    log.info(
        'Using %s symmetry-unique random opening placements.',
        len(sampler.positions),
    )
    return sampler


def build_opening_sampler(parsed_args, coach_args):
    if parsed_args.no_opening_book:
        parsed_args.opening_source = 'game'

    if (
        parsed_args.opening_source == 'unique'
        and (parsed_args.opening_book or parsed_args.arena_opening_suite)
    ):
        parsed_args.opening_source = 'book'
        log.info('Using opening book sampler because an opening book or arena suite was provided explicitly.')

    if parsed_args.opening_source == 'game':
        log.info('Using SantoriniGame.getInitBoard() for training starts.')
        return None

    if parsed_args.opening_source == 'unique':
        return build_unique_opening_sampler(parsed_args)

    if parsed_args.opening_source == 'book':
        sampler = build_book_opening_sampler(parsed_args, coach_args)
        if sampler is None:
            log.warning('No opening book found; using game random starts.')
        return sampler

    book_sampler = build_book_opening_sampler(parsed_args, coach_args)
    unique_sampler = build_unique_opening_sampler(parsed_args)
    if book_sampler is None:
        log.warning('No opening book found; using unique random opening placements only.')
        return unique_sampler

    mixed_sampler = SantoriniMixedOpeningSampler(
        book_sampler,
        unique_sampler,
        unique_probability=parsed_args.opening_mix_unique_probability,
    )
    log.info(
        'Using mixed opening sampler: %.0f%% book-filtered starts, %.0f%% unique random starts.',
        100.0 * (1.0 - mixed_sampler.unique_probability),
        100.0 * mixed_sampler.unique_probability,
    )
    return mixed_sampler


def main():
    parsed_args = parse_args()
    random.seed(parsed_args.seed)
    np.random.seed(parsed_args.seed)
    torch.manual_seed(parsed_args.seed)

    preset = PRESETS[parsed_args.preset]
    nnet_args.epochs = parsed_args.epochs or preset['epochs']
    nnet_args.batch_size = parsed_args.batch_size or preset['batch_size']
    nnet_args.max_train_steps = parsed_args.max_train_steps
    modern_architecture = parsed_args.architecture in ('v3', 'v4')
    nnet_args.replay_reuse = (
        parsed_args.replay_reuse
        if parsed_args.replay_reuse is not None
        else (V3_DEFAULT_REPLAY_REUSE if modern_architecture else None)
    )
    nnet_args.replay_reuse_warmup_iters = parsed_args.replay_reuse_warmup_iters
    nnet_args.optimizer = parsed_args.optimizer or ('adamw' if modern_architecture else 'adam')
    nnet_args.lr = (
        parsed_args.learning_rate
        if parsed_args.learning_rate is not None
        else (V3_DEFAULT_LEARNING_RATE if modern_architecture else 0.001)
    )
    nnet_args.policy_head_lr = parsed_args.policy_head_learning_rate
    nnet_args.value_head_lr = parsed_args.value_head_learning_rate
    nnet_args.value_target_anchor_checkpoint = parsed_args.value_target_anchor_checkpoint
    nnet_args.value_target_beta_start = parsed_args.value_target_beta_start
    nnet_args.value_target_beta_end = parsed_args.value_target_beta_end
    nnet_args.value_target_beta_start_iteration = (
        parsed_args.value_target_beta_start_iteration
    )
    nnet_args.value_target_beta_end_iteration = parsed_args.value_target_beta_end_iteration
    nnet_args.value_target_anchor_batch_size = parsed_args.value_target_anchor_batch_size
    nnet_args.weight_decay = (
        parsed_args.weight_decay
        if parsed_args.weight_decay is not None
        else (V3_DEFAULT_WEIGHT_DECAY if modern_architecture else 0.0)
    )
    nnet_args.lr_schedule = (
        parsed_args.lr_schedule
        if parsed_args.lr_schedule is not None
        else (list(V3_DEFAULT_LR_SCHEDULE) if modern_architecture else [])
    )
    nnet_args.symmetry_consistency_fraction = (
        parsed_args.symmetry_consistency_fraction
        if parsed_args.symmetry_consistency_fraction is not None
        else (
            V3_DEFAULT_SYMMETRY_CONSISTENCY_FRACTION
            if parsed_args.architecture == 'v3' else 0.0
        )
    )
    nnet_args.symmetry_consistency_policy_weight = (
        parsed_args.symmetry_consistency_policy_weight
        if parsed_args.symmetry_consistency_policy_weight is not None
        else (
            V3_DEFAULT_SYMMETRY_CONSISTENCY_POLICY_WEIGHT
            if parsed_args.architecture == 'v3' else 0.0
        )
    )
    nnet_args.symmetry_consistency_value_weight = (
        parsed_args.symmetry_consistency_value_weight
        if parsed_args.symmetry_consistency_value_weight is not None
        else (
            V3_DEFAULT_SYMMETRY_CONSISTENCY_VALUE_WEIGHT
            if parsed_args.architecture == 'v3' else 0.0
        )
    )
    nnet_args.quiet = parsed_args.quiet
    coach_args = build_coach_args(parsed_args)
    nnet_args.on_the_fly_symmetry = coach_args.symmetryAugmentation == 'on-the-fly'
    if modern_architecture:
        opening_sampler = None
        log.info('%s learns placement from the empty board; opening samplers are disabled.', parsed_args.architecture.upper())
    else:
        opening_sampler = build_opening_sampler(parsed_args, coach_args)

    log.info('Loading %s...', Game.__name__)
    game = Game(
        5,
        true_random_placement=not modern_architecture,
        sequential_placement=modern_architecture,
    )

    log.info('Loading Santorini network architecture %s...', parsed_args.architecture)
    nnet = build_nnet(game, parsed_args.architecture)

    anchor_nnet = None
    if parsed_args.anchor_checkpoint:
        anchor_checkpoint = resolve_anchor_checkpoint_path(parsed_args.anchor_checkpoint)
        log.info(
            'Loading fixed %s anchor checkpoint "%s"...',
            parsed_args.anchor_architecture,
            anchor_checkpoint,
        )
        with preserve_rng_state():
            anchor_nnet = build_anchor_nnet(game, parsed_args.anchor_architecture, anchor_checkpoint)

    loaded_metadata = {}
    if coach_args.load_model:
        log.info('Loading checkpoint "%s/%s"...', coach_args.load_folder_file[0], coach_args.load_folder_file[1])
        loaded_metadata = nnet.load_checkpoint(
            coach_args.load_folder_file[0],
            coach_args.load_folder_file[1],
            load_optimizer=coach_args.trainingMode == 'latest',
        )
        if coach_args.trainingMode == 'latest':
            coach_args['startIteration'] = int(loaded_metadata.get('iteration', 0))
            if parsed_args.start_iteration is not None:
                log.warning(
                    'Overriding checkpoint iteration metadata %s with %s.',
                    coach_args.startIteration,
                    parsed_args.start_iteration,
                )
                coach_args['startIteration'] = parsed_args.start_iteration
            if parsed_args.start_iteration is not None:
                log.info('Resume iteration override applied; continuing after iteration %s.', coach_args.startIteration)
            elif 'iteration' in loaded_metadata:
                log.info('Resume metadata loaded; continuing after iteration %s.', coach_args.startIteration)
            else:
                raise ValueError(
                    'Latest-mode resume checkpoint has no iteration metadata. Refusing to restart numbering at 1; '
                    'use --start-iteration with the known last completed iteration.'
                )
            if (
                coach_args.startIteration > 0
                and coach_args.startIteration % max(1, int(coach_args.milestoneInterval)) == 0
            ):
                resume_anchor = 'checkpoint_{}.pth.tar'.format(coach_args.startIteration)
                resume_anchor_path = os.path.join(coach_args.checkpoint, resume_anchor)
                if not os.path.isfile(resume_anchor_path):
                    log.info(
                        'Creating missing milestone resume anchor "%s" from the loaded model.',
                        resume_anchor_path,
                    )
                    nnet.save_checkpoint(coach_args.checkpoint, resume_anchor)
        coach_args['resumeMetadata'] = dict(loaded_metadata)
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    coach = Coach(
        game,
        nnet,
        coach_args,
        opening_sampler=opening_sampler,
        anchor_nnet=anchor_nnet,
    )

    if coach_args.load_model and parsed_args.load_examples:
        log.info("Loading 'trainExamples' from file...")
        examples_file = parsed_args.examples_file
        if examples_file is None:
            examples_file = 'latest.examples.npz' if coach_args.compactReplay else 'latest.examples'
        coach.loadTrainExamples(
            examples_file,
            skipFirstSelfPlay=parsed_args.skip_first_self_play,
        )

    log.info(
        'Config: architecture=%s preset=%s iters=%s eps=%s sims=%s search=%s gumbel_actions=%s gumbel_scale=%.2f gumbel_placement_scale=%.2f placement_explore=%.1f%%@%.2f eval_gumbel_scale=%.2f eval_gumbel_placement_scale=%.2f tactical=%s playout_cap=%s full_prob=%.2f fast_sims=%s placement_full=%s self_play_batch=%s arena=%s arena_batch=%s epochs=%s max_train_steps=%s replay_reuse=%s validation=%.3f batch=%s symmetry=%s consistency=%.2f@%.3f/%.3f symmetry_telemetry=%s inference_dedup=%s/%s search_symmetry=%s root_symmetries=%s/%s eval_root_symmetries=%s/%s policy_target_temp=%s optimizer=%s lr=%g weight_decay=%g lr_schedule=%s checkpoint=%s',
        parsed_args.architecture,
        parsed_args.preset,
        coach_args.numIters,
        coach_args.numEps,
        coach_args.numMCTSSims,
        coach_args.searchMode,
        coach_args.gumbelMaxConsideredActions,
        coach_args.gumbelScale,
        coach_args.gumbelPlacementScale,
        100.0 * coach_args.placementScaleExplorationProbability,
        coach_args.placementExplorationGumbelScale,
        coach_args.evaluationGumbelScale,
        coach_args.evaluationGumbelPlacementScale,
        coach_args.tacticalShortcuts,
        coach_args.playoutCapRandomization,
        coach_args.playoutCapFullProbability,
        coach_args.playoutCapFastSims,
        coach_args.playoutCapFullPlacement,
        coach_args.selfPlayBatchSize,
        coach_args.arenaCompare,
        coach_args.arenaBatchSize,
        nnet_args.epochs,
        nnet_args.max_train_steps,
        coach_args.replayReuse,
        coach_args.validationFraction,
        nnet_args.batch_size,
        coach_args.symmetryAugmentation,
        nnet_args.symmetry_consistency_fraction,
        nnet_args.symmetry_consistency_policy_weight,
        nnet_args.symmetry_consistency_value_weight,
        coach_args.symmetryTelemetrySampleSize,
        coach_args.inferenceDeduplication,
        coach_args.inferenceCacheSize,
        coach_args.searchSymmetryEvaluation,
        coach_args.rootSymmetrySamples,
        coach_args.placementRootSymmetrySamples,
        coach_args.evaluationRootSymmetrySamples,
        coach_args.evaluationPlacementRootSymmetrySamples,
        coach_args.policyTargetTemperature,
        nnet_args.optimizer,
        nnet_args.lr,
        nnet_args.weight_decay,
        nnet_args.lr_schedule,
        coach_args.checkpoint,
    )
    log.info(
        'Replay/opening design: placement_draws=%.1f%% frequency_exponent=%.2f; '
        'D4-unique neural starts=%.1f%% max_attempt_factor=%s.',
        100.0 * coach_args.placementReplayFraction,
        coach_args.placementReplayFrequencyExponent,
        100.0 * coach_args.uniqueNeuralStartFraction,
        coach_args.uniqueNeuralStartMaxAttemptFactor,
    )
    if anchor_nnet is not None:
        log.info(
            'Fixed anchor: architecture=%s interval=%s games=%s sims=%s',
            coach_args.anchorArchitecture,
            coach_args.anchorInterval,
            coach_args.anchorGames,
            coach_args.anchorMCTSSims,
        )
    log.info('Starting the Santorini learning process')
    try:
        coach.learn()
    finally:
        coach.close()


if __name__ == "__main__":
    main()
