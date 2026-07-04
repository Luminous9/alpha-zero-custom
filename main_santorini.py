import argparse
import logging
import os
import random

import coloredlogs
import numpy as np
import torch

from Coach import Coach
from santorini.pytorch.NNet import build_nnet
from santorini.pytorch.NNet import args as nnet_args
from santorini.SantoriniGame import SantoriniGame as Game
from santorini.SantoriniOpeningBook import (SantoriniOpeningSampler,
                                            SantoriniRandomOpeningSampler,
                                            SantoriniOpeningSuite,
                                            find_opening_book)
from utils import dotdict

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')

DEFAULT_ARENA_OPENING_SUITE = './santorini/opening_suites/bootstrap_arena_suite.json'
DEFAULT_OPENING_BOOK = './santorini/opening_books/bootstrap_result/opening_book.json'

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


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Santorini AlphaZero model.')
    parser.add_argument('--preset', choices=sorted(PRESETS.keys()), default='full')
    parser.add_argument('--architecture', choices=['v2', 'v3'], default='v2')
    parser.add_argument('--num-iters', type=int)
    parser.add_argument('--num-eps', type=int)
    parser.add_argument('--temp-threshold', type=int)
    parser.add_argument('--update-threshold', type=float)
    parser.add_argument('--maxlen-of-queue', type=int)
    parser.add_argument('--num-mcts-sims', type=int)
    parser.add_argument('--arena-compare', type=int)
    parser.add_argument('--cpuct', type=float, default=1.0)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--load-folder', type=str)
    parser.add_argument('--load-file', type=str, default='best.pth.tar')
    parser.add_argument('--load-model', action='store_true')
    parser.add_argument('--load-examples', action='store_true')
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
    parser.add_argument('--self-play-batch-size', type=int, default=1)
    parser.add_argument('--arena-batch-size', type=int)
    parser.add_argument(
        '--opening-source',
        choices=['unique', 'book', 'game'],
        default='unique',
        help='unique samples all symmetry-unique starting placements; book uses the old opening book sampler; game uses SantoriniGame.getInitBoard().',
    )
    parser.add_argument('--opening-book', type=str)
    parser.add_argument('--arena-opening-suite', type=str)
    parser.add_argument('--no-opening-book', action='store_true', help='Deprecated alias for --opening-source game.')
    parser.add_argument('--self-play-opening-max-abs-value', type=float, default=0.30)
    parser.add_argument('--self-play-old-filter-probability', type=float, default=0.70)
    parser.add_argument('--self-play-value-probability', type=float, default=0.25)
    parser.add_argument('--self-play-opening-tail-probability', type=float, default=0.05)
    parser.add_argument('--arena-opening-top-fraction', type=float, default=0.50)
    parser.add_argument('--arena-opening-max-abs-value', type=float, default=0.14)
    parser.add_argument('--no-opening-random-orientation', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--quiet', action='store_true')
    return parser.parse_args()


def build_coach_args(parsed_args):
    preset = PRESETS[parsed_args.preset]
    checkpoint = parsed_args.checkpoint or preset['checkpoint']
    load_folder = parsed_args.load_folder or checkpoint
    arena_batch_size = parsed_args.arena_batch_size or parsed_args.self_play_batch_size

    return dotdict({
        'numIters': parsed_args.num_iters or preset['numIters'],
        'numEps': parsed_args.num_eps or preset['numEps'],
        'tempThreshold': parsed_args.temp_threshold or preset['tempThreshold'],
        'updateThreshold': parsed_args.update_threshold or preset['updateThreshold'],
        'maxlenOfQueue': parsed_args.maxlen_of_queue or preset['maxlenOfQueue'],
        'numMCTSSims': parsed_args.num_mcts_sims or preset['numMCTSSims'],
        'arenaCompare': parsed_args.arena_compare or preset['arenaCompare'],
        'cpuct': parsed_args.cpuct,
        'checkpoint': checkpoint,
        'load_model': parsed_args.load_model,
        'load_folder_file': (load_folder, parsed_args.load_file),
        'numItersForTrainExamplesHistory': parsed_args.history_iters or preset['numItersForTrainExamplesHistory'],
        'checkpointExamplesToKeep': (
            parsed_args.checkpoint_examples_to_keep
            if parsed_args.checkpoint_examples_to_keep is not None
            else preset['checkpointExamplesToKeep']
        ),
        'deleteLoadedExamplesAfterFirstIteration': (
            preset['deleteLoadedExamplesAfterFirstIteration']
            and not parsed_args.keep_loaded_examples
        ),
        'atomicExamplesSave': parsed_args.atomic_examples_save or preset['atomicExamplesSave'],
        'saveBestTrainExamples': parsed_args.save_best_examples or preset['saveBestTrainExamples'],
        'selfPlayBatchSize': parsed_args.self_play_batch_size,
        'arenaBatchSize': arena_batch_size,
        'quiet': parsed_args.quiet,
    })


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
        sampler = SantoriniRandomOpeningSampler(
            board_size=5,
            random_orientation=not parsed_args.no_opening_random_orientation,
        )
        log.info(
            'Using %s symmetry-unique random opening placements for self-play and paired arena starts.',
            len(sampler.positions),
        )
        return sampler

    opening_book = find_opening_book(opening_book_candidates(parsed_args, coach_args))
    if opening_book is None:
        if parsed_args.opening_book:
            raise FileNotFoundError("No opening book found at {}".format(parsed_args.opening_book))
        log.warning('No opening book found; using game random starts.')
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


def main():
    parsed_args = parse_args()
    random.seed(parsed_args.seed)
    np.random.seed(parsed_args.seed)
    torch.manual_seed(parsed_args.seed)

    preset = PRESETS[parsed_args.preset]
    nnet_args.epochs = parsed_args.epochs or preset['epochs']
    nnet_args.batch_size = parsed_args.batch_size or preset['batch_size']
    nnet_args.quiet = parsed_args.quiet
    coach_args = build_coach_args(parsed_args)
    opening_sampler = build_opening_sampler(parsed_args, coach_args)

    log.info('Loading %s...', Game.__name__)
    game = Game(5, true_random_placement=True)

    log.info('Loading Santorini network architecture %s...', parsed_args.architecture)
    nnet = build_nnet(game, parsed_args.architecture)

    if coach_args.load_model:
        log.info('Loading checkpoint "%s/%s"...', coach_args.load_folder_file[0], coach_args.load_folder_file[1])
        nnet.load_checkpoint(coach_args.load_folder_file[0], coach_args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    coach = Coach(game, nnet, coach_args, opening_sampler=opening_sampler)

    if coach_args.load_model and parsed_args.load_examples:
        log.info("Loading 'trainExamples' from file...")
        examples_file = parsed_args.examples_file
        if examples_file is None and parsed_args.load_file == 'best.pth.tar':
            examples_file = 'latest.examples'
        coach.loadTrainExamples(
            examples_file,
            skipFirstSelfPlay=parsed_args.skip_first_self_play,
        )

    log.info(
        'Config: architecture=%s preset=%s iters=%s eps=%s sims=%s self_play_batch=%s arena=%s arena_batch=%s epochs=%s batch=%s checkpoint=%s',
        parsed_args.architecture,
        parsed_args.preset,
        coach_args.numIters,
        coach_args.numEps,
        coach_args.numMCTSSims,
        coach_args.selfPlayBatchSize,
        coach_args.arenaCompare,
        coach_args.arenaBatchSize,
        nnet_args.epochs,
        nnet_args.batch_size,
        coach_args.checkpoint,
    )
    log.info('Starting the Santorini learning process')
    coach.learn()


if __name__ == "__main__":
    main()
