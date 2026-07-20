import argparse
import json
import os
import sys

import numpy as np

import Arena
from BatchedArena import BatchedMCTSArena
from MCTS import MCTS
from santorini.SantoriniGame import SantoriniGame
from santorini.SantoriniOpeningBook import (
    SantoriniOpeningBook,
    SantoriniRandomOpeningSampler,
    SantoriniOpeningSampler,
    SantoriniOpeningSuite,
    find_opening_book,
    random_board_orientation,
)
from santorini.SantoriniPlayers import GreedySantoriniPlayer, RandomPlayer
from santorini.pytorch.LegacyNNet import LegacyNNetWrapper
from santorini.pytorch.NNet import build_nnet as build_santorini_nnet
from tqdm import tqdm
from utils import dotdict


def build_nnet(game, architecture):
    if architecture == 'v2':
        return build_santorini_nnet(game, architecture)
    if architecture == 'v3':
        return build_santorini_nnet(game, architecture)
    if architecture == 'v1':
        return LegacyNNetWrapper(game)
    raise ValueError("Unknown architecture: {}".format(architecture))


def search_args(
    sims,
    search_mode='puct',
    gumbel_max_considered_actions=16,
    gumbel_scale=1.0,
    gumbel_placement_scale=None,
    search_symmetry_evaluation=False,
    root_symmetry_samples=8,
    placement_root_symmetry_samples=8,
    inference_deduplication=False,
    inference_cache_size=4096,
):
    if gumbel_placement_scale is None:
        gumbel_placement_scale = gumbel_scale
    return dotdict({
        'numMCTSSims': int(sims),
        'cpuct': 1.0,
        'searchMode': search_mode,
        'gumbelMaxConsideredActions': int(gumbel_max_considered_actions),
        'gumbelScale': float(gumbel_scale),
        'gumbelPlacementScale': float(gumbel_placement_scale),
        'tacticalShortcuts': True,
        'searchSymmetryEvaluation': bool(search_symmetry_evaluation),
        'rootSymmetrySamples': int(root_symmetry_samples),
        'placementRootSymmetrySamples': int(placement_root_symmetry_samples),
        'inferenceDeduplication': bool(inference_deduplication),
        'inferenceCacheSize': int(inference_cache_size),
    })


class NetworkMCTSPlayer:
    """MCTS player around an already-loaded network, reset before every arena game."""

    def __init__(
        self,
        game,
        nnet,
        sims,
        action_temp=0.0,
        search_mode='puct',
        gumbel_max_considered_actions=16,
        gumbel_scale=1.0,
        gumbel_placement_scale=None,
        search_symmetry_evaluation=False,
        root_symmetry_samples=8,
        placement_root_symmetry_samples=8,
        inference_deduplication=False,
        inference_cache_size=4096,
    ):
        self.game = game
        self.nnet = nnet
        self.mcts_args = search_args(
            sims,
            search_mode=search_mode,
            gumbel_max_considered_actions=gumbel_max_considered_actions,
            gumbel_scale=gumbel_scale,
            gumbel_placement_scale=gumbel_placement_scale,
            search_symmetry_evaluation=search_symmetry_evaluation,
            root_symmetry_samples=root_symmetry_samples,
            placement_root_symmetry_samples=placement_root_symmetry_samples,
            inference_deduplication=inference_deduplication,
            inference_cache_size=inference_cache_size,
        )
        self.action_temp = action_temp
        self.mcts = None

    def startGame(self):
        self.mcts = MCTS(self.game, self.nnet, self.mcts_args)

    def play(self, board):
        if self.mcts is None:
            self.startGame()
        probs = self.mcts.getActionProb(board, temp=self.action_temp)
        return select_legal_action(self.game, board, probs, sample=self.action_temp > 0)

    __call__ = play


class NeuralMCTSPlayer(NetworkMCTSPlayer):
    def __init__(
        self,
        game,
        checkpoint_folder,
        checkpoint_file,
        sims,
        architecture='v2',
        action_temp=0.0,
        search_mode='puct',
        gumbel_max_considered_actions=16,
        gumbel_scale=1.0,
        gumbel_placement_scale=None,
        search_symmetry_evaluation=False,
        root_symmetry_samples=8,
        placement_root_symmetry_samples=8,
        inference_deduplication=False,
        inference_cache_size=4096,
    ):
        nnet = build_nnet(game, architecture)
        nnet.load_checkpoint(checkpoint_folder, checkpoint_file)
        super().__init__(
            game,
            nnet,
            sims,
            action_temp=action_temp,
            search_mode=search_mode,
            gumbel_max_considered_actions=gumbel_max_considered_actions,
            gumbel_scale=gumbel_scale,
            gumbel_placement_scale=gumbel_placement_scale,
            search_symmetry_evaluation=search_symmetry_evaluation,
            root_symmetry_samples=root_symmetry_samples,
            placement_root_symmetry_samples=placement_root_symmetry_samples,
            inference_deduplication=inference_deduplication,
            inference_cache_size=inference_cache_size,
        )


def select_legal_action(game, board, probs, sample=False):
    probs = np.array(probs)
    valids = game.getValidMoves(board, 1)
    masked_probs = probs * valids
    if masked_probs.sum() > 0:
        if sample:
            masked_probs = masked_probs / masked_probs.sum()
            return int(np.random.choice(len(masked_probs), p=masked_probs))
        return int(np.argmax(masked_probs))
    return int(np.flatnonzero(valids)[0])


def build_baseline(game, name):
    if name == 'random':
        return RandomPlayer(game).play
    if name == 'greedy':
        return GreedySantoriniPlayer(game).play
    raise ValueError("Unknown baseline: {}".format(name))


def batched_arena_requested(args):
    return int(getattr(args, 'arena_batch_size', 1)) > 1


def validate_batched_arena_args(parser, args):
    values = args if isinstance(args, dict) else vars(args)
    placement_only = bool(values.get('placement_only_comparison', False))
    if placement_only:
        if int(values.get('arena_batch_size', 1)) <= 1:
            parser.error('--placement-only-comparison requires --arena-batch-size > 1.')
        if not values.get('opponent_checkpoint_folder'):
            parser.error('--placement-only-comparison requires --opponent-checkpoint-folder.')
        if not values.get('standard_controller_folder'):
            parser.error('--placement-only-comparison requires --standard-controller-folder.')
        if values.get('opening_source', 'book') != 'game':
            parser.error('--placement-only-comparison requires --opening-source game.')
        if values.get('architecture') != 'v3' or values.get('opponent_architecture') != 'v3':
            parser.error('--placement-only-comparison requires two V3 placement contestants.')

    if not batched_arena_requested(args):
        return
    if not args.opponent_checkpoint_folder:
        parser.error('--arena-batch-size > 1 currently requires --opponent-checkpoint-folder.')
    if args.action_temp != 0:
        parser.error('--arena-batch-size > 1 requires deterministic play with --action-temp 0.')


def load_opening_board(opening_book_path, opening_id):
    book = SantoriniOpeningBook.load(opening_book_path)
    for position in book.positions:
        if int(position["id"]) == int(opening_id):
            pieces = np.array(position["pieces"], dtype=int)
            heights = np.zeros_like(pieces, dtype=int)
            return np.array([pieces, heights], dtype=int), position

    raise ValueError(
        "Opening id {} was not found in {}".format(opening_id, opening_book_path)
    )


def opening_json_kind(path):
    with open(path) as opening_file:
        payload = json.load(opening_file)
    if "player1_choices" in payload:
        return "book"
    if "positions" in payload:
        return "suite"
    return "unknown"


def board_from_opening_position(position, random_orientation_enabled=True, rng=None):
    pieces = np.array(position["pieces"], dtype=int)
    heights = np.zeros_like(pieces, dtype=int)
    board = np.array([pieces, heights], dtype=int)
    if random_orientation_enabled:
        board = random_board_orientation(board, rng=rng)
    return board


def sample_opening_suite(opening_suite_path, count, random_orientation_enabled=True, rng=None):
    suite = SantoriniOpeningSuite.load(opening_suite_path)
    count = int(count)
    if count <= 0:
        return suite, []

    replace = count > len(suite.positions)
    rng = rng if rng is not None else np.random
    indices = rng.choice(len(suite.positions), size=count, replace=replace)
    boards = [
        board_from_opening_position(
            suite.positions[int(index)],
            random_orientation_enabled=random_orientation_enabled,
            rng=rng,
        )
        for index in indices
    ]
    return suite, boards


def opening_book_candidates(args):
    checkpoint = args.checkpoint_folder
    checkpoint_name = os.path.basename(os.path.normpath(checkpoint))
    candidates = [
        args.opening_book_path,
        os.path.join(checkpoint, 'opening_book.json'),
        os.path.join(checkpoint, 'opening_books', 'opening_book.json'),
        os.path.join(os.path.dirname(checkpoint), 'opening_books', checkpoint_name, 'opening_book.json'),
        os.path.join('temp', 'santorini_opening_books', checkpoint_name, 'opening_book.json'),
    ]

    if args.opponent_checkpoint_folder:
        opponent_checkpoint = args.opponent_checkpoint_folder
        opponent_checkpoint_name = os.path.basename(os.path.normpath(opponent_checkpoint))
        candidates.extend([
            os.path.join(opponent_checkpoint, 'opening_book.json'),
            os.path.join(opponent_checkpoint, 'opening_books', 'opening_book.json'),
            os.path.join(
                os.path.dirname(opponent_checkpoint),
                'opening_books',
                opponent_checkpoint_name,
                'opening_book.json',
            ),
            os.path.join('temp', 'santorini_opening_books', opponent_checkpoint_name, 'opening_book.json'),
        ])

    return candidates


def build_opening_suite(args):
    if args.no_opening_book:
        return None, None, None, 'random_start'

    opening_source = getattr(args, 'opening_source', 'book')
    rng = np.random.RandomState(int(getattr(args, 'opening_seed', 20260715)))
    if opening_source == 'game':
        print("Using game random starts.")
        return None, None, None, 'random_start'

    if args.opening_id is not None and getattr(args, 'arena_opening_suite', None):
        raise ValueError('--opening-id cannot be used with --arena-opening-suite.')
    if args.opening_id is not None and opening_source != 'book':
        raise ValueError('--opening-id can only be used with --opening-source book.')

    if opening_source == 'unique':
        sampler = SantoriniRandomOpeningSampler(
            board_size=5,
            random_orientation=not args.no_opening_random_orientation,
            rng=rng,
        )
        opening_boards = sampler.sample_distinct_arena_suite(int(args.games / 2))
        print(
            'Sampled {} fixed, distinct paired openings from {} symmetry-unique placements.'.format(
                len(opening_boards),
                len(sampler.positions),
            )
        )
        return None, opening_boards, None, 'sampled_unique'

    if getattr(args, 'arena_opening_suite', None):
        suite, opening_boards = sample_opening_suite(
            args.arena_opening_suite,
            int(args.games / 2),
            random_orientation_enabled=not args.no_opening_random_orientation,
            rng=rng,
        )
        print(
            'Loaded opening suite "{}" ({} positions).'.format(
                args.arena_opening_suite,
                len(suite.positions),
            )
        )
        return args.arena_opening_suite, opening_boards, None, 'sampled_suite'

    opening_book_path = find_opening_book(opening_book_candidates(args))
    if opening_book_path is None:
        if args.opening_book_path:
            raise FileNotFoundError("No opening book found at {}".format(args.opening_book_path))
        print("No opening book found; using game random starts.")
        return None, None, None, 'random_start'

    if args.opening_id is None and opening_json_kind(opening_book_path) == 'suite':
        suite, opening_boards = sample_opening_suite(
            opening_book_path,
            int(args.games / 2),
            random_orientation_enabled=not args.no_opening_random_orientation,
            rng=rng,
        )
        print(
            'Loaded opening suite "{}" ({} positions).'.format(
                opening_book_path,
                len(suite.positions),
            )
        )
        return opening_book_path, opening_boards, None, 'sampled_suite'

    if args.opening_id is not None:
        opening_board, opening_position = load_opening_board(
            opening_book_path,
            args.opening_id,
        )
        opening_boards = [opening_board for _ in range(max(1, int(args.games / 2)))]
        return opening_book_path, opening_boards, opening_position, 'fixed_id'

    sampler = SantoriniOpeningSampler.load(
        opening_book_path,
        arena_top_fraction=args.arena_opening_top_fraction,
        arena_max_abs_value=args.arena_opening_max_abs_value,
        random_orientation=not args.no_opening_random_orientation,
        rng=rng,
    )
    opening_boards = sampler.sample_arena_suite(int(args.games / 2))
    print(
        'Loaded opening book "{}" ({} positions, {} arena candidates).'.format(
            opening_book_path,
            len(sampler.book.positions),
            len(sampler._arena_candidates()),
        )
    )
    return opening_book_path, opening_boards, None, 'sampled_book'


def display_name_from_folder(folder):
    return os.path.basename(os.path.normpath(folder))


def new_seat_stats(name):
    return {
        "name": name,
        "first_player": {"wins": 0, "losses": 0, "draws": 0},
        "second_player": {"wins": 0, "losses": 0, "draws": 0},
    }


def record_seat_result(seat_stats, first_contestant, second_contestant, game_result):
    if game_result == 1:
        seat_stats[first_contestant]["first_player"]["wins"] += 1
        seat_stats[second_contestant]["second_player"]["losses"] += 1
    elif game_result == -1:
        seat_stats[first_contestant]["first_player"]["losses"] += 1
        seat_stats[second_contestant]["second_player"]["wins"] += 1
    else:
        seat_stats[first_contestant]["first_player"]["draws"] += 1
        seat_stats[second_contestant]["second_player"]["draws"] += 1


def format_seat_record(record, include_draws=True):
    if include_draws:
        return record
    return {
        "wins": record["wins"],
        "losses": record["losses"],
    }


def play_opening_games_by_seat(
    player1,
    player2,
    game,
    opening_board,
    games,
    contestant1_name,
    contestant2_name,
    show_progress=True,
):
    games_per_seat = int(games / 2)
    one_won = 0
    two_won = 0
    draws = 0
    seat_stats = {
        "contestant1": new_seat_stats(contestant1_name),
        "contestant2": new_seat_stats(contestant2_name),
    }

    arena = Arena.Arena(player1, player2, game, display=SantoriniGame.display)
    for _ in tqdm(
        range(games_per_seat),
        desc="Opening pit ({} first)".format(contestant1_name),
        disable=not show_progress,
        file=sys.stdout,
        dynamic_ncols=True,
    ):
        game_result = arena.playGame(verbose=False, opening_board=opening_board)
        record_seat_result(seat_stats, "contestant1", "contestant2", game_result)
        if game_result == 1:
            one_won += 1
        elif game_result == -1:
            two_won += 1
        else:
            draws += 1

    arena.player1, arena.player2 = arena.player2, arena.player1
    for _ in tqdm(
        range(games_per_seat),
        desc="Opening pit ({} first)".format(contestant2_name),
        disable=not show_progress,
        file=sys.stdout,
        dynamic_ncols=True,
    ):
        game_result = arena.playGame(verbose=False, opening_board=opening_board)
        record_seat_result(seat_stats, "contestant2", "contestant1", game_result)
        if game_result == -1:
            one_won += 1
        elif game_result == 1:
            two_won += 1
        else:
            draws += 1

    return one_won, two_won, draws, seat_stats


def main():
    parser = argparse.ArgumentParser(description='Pit a Santorini neural MCTS player against a baseline.')
    parser.add_argument('--baseline', choices=['random', 'greedy'], default='random')
    parser.add_argument('--games', type=int, default=4)
    parser.add_argument('--sims', type=int, default=25)
    parser.add_argument('--search-mode', choices=['puct', 'gumbel'], default='puct')
    parser.add_argument('--gumbel-max-considered-actions', type=int, default=16)
    parser.add_argument('--gumbel-scale', type=float, default=1.0)
    parser.add_argument(
        '--gumbel-placement-scale',
        type=float,
        help='Gumbel scale for placement roots; defaults to --gumbel-scale.',
    )
    parser.add_argument(
        '--no-search-symmetry-evaluation',
        action='store_true',
        help='Disable D4-randomized leaf evaluation for V3 contestants.',
    )
    parser.add_argument(
        '--root-symmetry-samples',
        type=int,
        default=8,
        help='Distinct D4 orientations averaged at standard V3 evaluation roots.',
    )
    parser.add_argument(
        '--placement-root-symmetry-samples',
        type=int,
        default=8,
        help='Distinct D4 orientations averaged at placement V3 evaluation roots.',
    )
    parser.add_argument('--no-inference-deduplication', action='store_true')
    parser.add_argument('--inference-cache-size', type=int, default=4096)
    parser.add_argument('--checkpoint-folder', default='./temp/santorini_quick/')
    parser.add_argument('--checkpoint-file', default='best.pth.tar')
    parser.add_argument('--architecture', choices=['v1', 'v2', 'v3'], default='v2')
    parser.add_argument('--opponent-checkpoint-folder')
    parser.add_argument('--opponent-checkpoint-file', default='best.pth.tar')
    parser.add_argument('--opponent-architecture', choices=['v1', 'v2', 'v3'], default='v2')
    parser.add_argument('--opponent-sims', type=int)
    parser.add_argument('--opponent-search-mode', choices=['puct', 'gumbel'], default='puct')
    parser.add_argument('--opponent-gumbel-max-considered-actions', type=int, default=16)
    parser.add_argument('--opponent-gumbel-scale', type=float, default=1.0)
    parser.add_argument(
        '--opponent-gumbel-placement-scale',
        type=float,
        help='Opponent Gumbel scale for placement roots; defaults to --opponent-gumbel-scale.',
    )
    parser.add_argument(
        '--placement-only-comparison',
        action='store_true',
        help=(
            'Let the two contestants choose placements, then use one fixed neural-MCTS '
            'controller for both sides during all standard play.'
        ),
    )
    parser.add_argument(
        '--standard-controller-folder',
        help='Checkpoint folder for the shared post-placement controller.',
    )
    parser.add_argument(
        '--standard-controller-file',
        default='latest.pth.tar',
        help='Checkpoint filename for the shared post-placement controller.',
    )
    parser.add_argument(
        '--standard-controller-architecture',
        choices=['v1', 'v2', 'v3'],
        default='v3',
    )
    parser.add_argument(
        '--standard-controller-sims',
        type=int,
        help='Post-placement simulations per move; defaults to --sims.',
    )
    parser.add_argument(
        '--standard-controller-search-mode',
        choices=['puct', 'gumbel'],
        default='gumbel',
    )
    parser.add_argument('--standard-controller-gumbel-max-considered-actions', type=int, default=16)
    parser.add_argument(
        '--standard-controller-gumbel-scale',
        type=float,
        default=0.0,
        help='Gumbel scale for shared standard play; 0 gives deterministic search.',
    )
    parser.add_argument(
        '--arena-batch-size',
        type=int,
        default=1,
        help='Batch this many simultaneous games for two-checkpoint deterministic pits.',
    )
    parser.add_argument(
        '--action-temp',
        type=float,
        default=0.0,
        help='Temperature for sampling neural MCTS actions. 0 keeps deterministic argmax play.',
    )
    parser.add_argument('--fresh', action='store_true', help='Use an untrained network even if a checkpoint exists.')
    parser.add_argument('--opening-book-path', help='Optional opening book JSON path.')
    parser.add_argument('--arena-opening-suite', help='Optional fixed opening suite JSON path to sample paired arena openings from.')
    parser.add_argument('--opening-id', type=int, help='Opening response id to use for every paired seat game.')
    parser.add_argument(
        '--opening-source',
        choices=['book', 'unique', 'game'],
        default='book',
        help='Opening source for paired arena starts. book preserves legacy book/suite behavior; unique samples the 9,664 symmetry-unique worker placements; game uses SantoriniGame random starts.',
    )
    parser.add_argument('--no-opening-book', action='store_true', help='Deprecated alias for --opening-source game.')
    parser.add_argument('--arena-opening-top-fraction', type=float, default=0.50)
    parser.add_argument('--arena-opening-max-abs-value', type=float, default=0.14)
    parser.add_argument('--no-opening-random-orientation', action='store_true')
    parser.add_argument(
        '--opening-seed',
        type=int,
        default=20260715,
        help='Fixed seed used to reconstruct paired book, suite, or unique openings.',
    )
    parser.add_argument('--json-out', help='Optional path to write evaluation results as JSON.')
    args = parser.parse_args()
    if args.no_opening_book:
        args.opening_source = 'game'

    if args.opening_id is not None and args.opening_source != 'book':
        parser.error('--opening-id can only be used with --opening-source book.')
    if args.opening_id is not None and args.arena_opening_suite:
        parser.error('--opening-id cannot be used with --arena-opening-suite.')
    if args.action_temp < 0:
        parser.error('--action-temp must be non-negative.')
    if args.arena_batch_size < 1:
        parser.error('--arena-batch-size must be at least 1.')
    if args.gumbel_max_considered_actions < 1 or args.opponent_gumbel_max_considered_actions < 1:
        parser.error('Gumbel max considered actions must be at least 1.')
    if args.standard_controller_gumbel_max_considered_actions < 1:
        parser.error('Standard-controller Gumbel max considered actions must be at least 1.')
    if args.gumbel_scale < 0 or args.opponent_gumbel_scale < 0:
        parser.error('Gumbel scale cannot be negative.')
    if args.standard_controller_gumbel_scale < 0:
        parser.error('Standard-controller Gumbel scale cannot be negative.')
    if args.standard_controller_sims is not None and args.standard_controller_sims < 1:
        parser.error('--standard-controller-sims must be at least 1.')
    if not 1 <= args.root_symmetry_samples <= 8:
        parser.error('--root-symmetry-samples must be between 1 and 8.')
    if not 1 <= args.placement_root_symmetry_samples <= 8:
        parser.error('--placement-root-symmetry-samples must be between 1 and 8.')
    if args.inference_cache_size < 0:
        parser.error('--inference-cache-size cannot be negative.')
    if args.gumbel_placement_scale is None:
        args.gumbel_placement_scale = args.gumbel_scale
    if args.opponent_gumbel_placement_scale is None:
        args.opponent_gumbel_placement_scale = args.opponent_gumbel_scale
    if args.gumbel_placement_scale < 0 or args.opponent_gumbel_placement_scale < 0:
        parser.error('Gumbel placement scale cannot be negative.')
    validate_batched_arena_args(parser, args)
    np.random.seed(args.opening_seed)

    uses_v3 = args.architecture == 'v3' or (
        args.opponent_checkpoint_folder and args.opponent_architecture == 'v3'
    )
    game = SantoriniGame(
        5,
        true_random_placement=not uses_v3,
        sequential_placement=uses_v3,
    )
    opening_book_path, opening_boards, opening_position, opening_mode = build_opening_suite(args)
    if opening_position is not None:
        opening_board = opening_boards[0]
        print(
            "Loaded opening id {}: P1 rank {} P1={} P2={} response rank {}".format(
                opening_position["id"],
                opening_position["player1_rank"],
                opening_position["player1"],
                opening_position["player2"],
                opening_position["player2_response_rank"],
            )
        )

    checkpoint_path = os.path.join(args.checkpoint_folder, args.checkpoint_file)
    contestant1_name = display_name_from_folder(args.checkpoint_folder)
    if args.opponent_checkpoint_folder:
        opponent_checkpoint_path = os.path.join(args.opponent_checkpoint_folder, args.opponent_checkpoint_file)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError("No model in path {}".format(checkpoint_path))
        if not os.path.exists(opponent_checkpoint_path):
            raise FileNotFoundError("No opponent model in path {}".format(opponent_checkpoint_path))

        print("Loaded checkpoint: {}".format(checkpoint_path))
        print("Loaded opponent checkpoint: {}".format(opponent_checkpoint_path))
        print("Architecture: {}; opponent architecture: {}".format(
            args.architecture,
            args.opponent_architecture,
        ))
        nnet_player_obj = NeuralMCTSPlayer(
            game,
            args.checkpoint_folder,
            args.checkpoint_file,
            args.sims,
            architecture=args.architecture,
            action_temp=args.action_temp,
            search_mode=args.search_mode,
            gumbel_max_considered_actions=args.gumbel_max_considered_actions,
            gumbel_scale=args.gumbel_scale,
            gumbel_placement_scale=args.gumbel_placement_scale,
            search_symmetry_evaluation=(
                args.architecture == 'v3' and not args.no_search_symmetry_evaluation
            ),
            root_symmetry_samples=args.root_symmetry_samples,
            placement_root_symmetry_samples=args.placement_root_symmetry_samples,
            inference_deduplication=(
                args.architecture == 'v3' and not args.no_inference_deduplication
            ),
            inference_cache_size=args.inference_cache_size,
        )
        opponent_player_obj = NeuralMCTSPlayer(
            game,
            args.opponent_checkpoint_folder,
            args.opponent_checkpoint_file,
            args.opponent_sims or args.sims,
            architecture=args.opponent_architecture,
            action_temp=args.action_temp,
            search_mode=args.opponent_search_mode,
            gumbel_max_considered_actions=args.opponent_gumbel_max_considered_actions,
            gumbel_scale=args.opponent_gumbel_scale,
            gumbel_placement_scale=args.opponent_gumbel_placement_scale,
            search_symmetry_evaluation=(
                args.opponent_architecture == 'v3'
                and not args.no_search_symmetry_evaluation
            ),
            root_symmetry_samples=args.root_symmetry_samples,
            placement_root_symmetry_samples=args.placement_root_symmetry_samples,
            inference_deduplication=(
                args.opponent_architecture == 'v3'
                and not args.no_inference_deduplication
            ),
            inference_cache_size=args.inference_cache_size,
        )
        player1 = nnet_player_obj
        player2 = opponent_player_obj
        contestant2_name = display_name_from_folder(args.opponent_checkpoint_folder)
        if contestant1_name == contestant2_name or args.search_mode != args.opponent_search_mode:
            contestant1_name = "{} [{}]".format(contestant1_name, args.search_mode)
            contestant2_name = "{} [{}]".format(contestant2_name, args.opponent_search_mode)
    else:
        nnet = build_nnet(game, args.architecture)
        if not args.fresh and os.path.exists(checkpoint_path):
            nnet.load_checkpoint(args.checkpoint_folder, args.checkpoint_file)
            print("Loaded checkpoint: {} ({})".format(checkpoint_path, args.architecture))
        else:
            print("Using fresh untrained network.")

        player1 = NetworkMCTSPlayer(
            game,
            nnet,
            args.sims,
            action_temp=args.action_temp,
            search_mode=args.search_mode,
            gumbel_max_considered_actions=args.gumbel_max_considered_actions,
            gumbel_scale=args.gumbel_scale,
            gumbel_placement_scale=args.gumbel_placement_scale,
            search_symmetry_evaluation=(
                args.architecture == 'v3' and not args.no_search_symmetry_evaluation
            ),
            root_symmetry_samples=args.root_symmetry_samples,
            placement_root_symmetry_samples=args.placement_root_symmetry_samples,
            inference_deduplication=(
                args.architecture == 'v3' and not args.no_inference_deduplication
            ),
            inference_cache_size=args.inference_cache_size,
        )
        player2 = build_baseline(game, args.baseline)
        contestant2_name = args.baseline

    if args.action_temp > 0:
        print("Sampling neural MCTS actions with temperature {}.".format(args.action_temp))
    print("Contestant 1 search: {} ({} sims; Gumbel standard/placement scales {}/{}).".format(
        args.search_mode,
        args.sims,
        args.gumbel_scale,
        args.gumbel_placement_scale,
    ))
    if args.opponent_checkpoint_folder:
        print("Contestant 2 search: {} ({} sims; Gumbel standard/placement scales {}/{}).".format(
            args.opponent_search_mode,
            args.opponent_sims or args.sims,
            args.opponent_gumbel_scale,
            args.opponent_gumbel_placement_scale,
        ))
    if uses_v3 and not args.no_search_symmetry_evaluation:
        print(
            'V3 search symmetry: random interior orientation; standard/placement '
            'root averages {}/{}.'.format(
                args.root_symmetry_samples,
                args.placement_root_symmetry_samples,
            )
        )

    seat_stats = None
    placement_diagnostics = None
    inference_diagnostics = None
    use_batched_arena = batched_arena_requested(args)
    if use_batched_arena:
        print("Using batched arena with batch size {}.".format(args.arena_batch_size))
        player1_args = search_args(
            args.sims,
            args.search_mode,
            args.gumbel_max_considered_actions,
            args.gumbel_scale,
            args.gumbel_placement_scale,
            search_symmetry_evaluation=(
                args.architecture == 'v3' and not args.no_search_symmetry_evaluation
            ),
            root_symmetry_samples=args.root_symmetry_samples,
            placement_root_symmetry_samples=args.placement_root_symmetry_samples,
            inference_deduplication=(
                args.architecture == 'v3' and not args.no_inference_deduplication
            ),
            inference_cache_size=args.inference_cache_size,
        )
        player2_args = search_args(
            args.opponent_sims or args.sims,
            args.opponent_search_mode,
            args.opponent_gumbel_max_considered_actions,
            args.opponent_gumbel_scale,
            args.opponent_gumbel_placement_scale,
            search_symmetry_evaluation=(
                args.opponent_architecture == 'v3'
                and not args.no_search_symmetry_evaluation
            ),
            root_symmetry_samples=args.root_symmetry_samples,
            placement_root_symmetry_samples=args.placement_root_symmetry_samples,
            inference_deduplication=(
                args.opponent_architecture == 'v3'
                and not args.no_inference_deduplication
            ),
            inference_cache_size=args.inference_cache_size,
        )
        standard_controller_nnet = None
        standard_controller_args = None
        if args.placement_only_comparison:
            standard_controller_path = os.path.join(
                args.standard_controller_folder,
                args.standard_controller_file,
            )
            if not os.path.exists(standard_controller_path):
                raise FileNotFoundError(
                    'No standard controller model in path {}'.format(standard_controller_path)
                )
            standard_controller_nnet = build_nnet(
                game,
                args.standard_controller_architecture,
            )
            standard_controller_nnet.load_checkpoint(
                args.standard_controller_folder,
                args.standard_controller_file,
            )
            standard_controller_args = search_args(
                args.standard_controller_sims or args.sims,
                args.standard_controller_search_mode,
                args.standard_controller_gumbel_max_considered_actions,
                args.standard_controller_gumbel_scale,
                args.standard_controller_gumbel_scale,
                search_symmetry_evaluation=(
                    args.standard_controller_architecture == 'v3'
                    and not args.no_search_symmetry_evaluation
                ),
                root_symmetry_samples=args.root_symmetry_samples,
                placement_root_symmetry_samples=args.placement_root_symmetry_samples,
                inference_deduplication=(
                    args.standard_controller_architecture == 'v3'
                    and not args.no_inference_deduplication
                ),
                inference_cache_size=args.inference_cache_size,
            )
            print(
                'Placement-only comparison: both sides switch after placement to {} '
                'using {} search ({} sims; Gumbel scale {}).'.format(
                    standard_controller_path,
                    args.standard_controller_search_mode,
                    args.standard_controller_sims or args.sims,
                    args.standard_controller_gumbel_scale,
                )
            )
        arena = BatchedMCTSArena(
            game,
            player1.nnet,
            player2.nnet,
            player1_args,
            batch_size=args.arena_batch_size,
            opening_boards=opening_boards,
            progress_file=sys.stdout,
            game_seeds=list(np.random.RandomState(args.opening_seed).randint(
                0,
                2**31 - 1,
                size=args.games // 2,
            )),
            player_args={1: player1_args, -1: player2_args},
            standard_controller_nnet=standard_controller_nnet,
            standard_controller_args=standard_controller_args,
            record_placement_diagnostics=bool(uses_v3 and opening_boards is None),
        )
        nnet_wins, opponent_wins, draws = arena.playGames(args.games)
        placement_diagnostics = arena.placementDiagnostics()
        inference_diagnostics = arena.inferenceDiagnostics()
        if inference_diagnostics['requested']:
            print(
                'Inference reuse: {} / {} requested evaluations ({:.1f}%).'.format(
                    inference_diagnostics['reused'],
                    inference_diagnostics['requested'],
                    100.0 * inference_diagnostics['reuse_rate'],
                )
            )
    elif opening_position is not None:
        nnet_wins, opponent_wins, draws, seat_stats = play_opening_games_by_seat(
            player1,
            player2,
            game,
            opening_board,
            args.games,
            contestant1_name,
            contestant2_name,
        )
    else:
        arena = Arena.Arena(
            player1,
            player2,
            game,
            display=SantoriniGame.display,
            opening_boards=opening_boards,
            progress_file=sys.stdout,
        )
        nnet_wins, opponent_wins, draws = arena.playGames(args.games, verbose=False)

    print("{} wins: {}".format(contestant1_name, nnet_wins))
    print("{} wins: {}".format(contestant2_name, opponent_wins))
    include_draws = bool(getattr(game, 'supports_draws', True))
    if include_draws:
        print("Draws: {}".format(draws))
    if placement_diagnostics is not None:
        print(
            'Learned placements: {} exact, {} symmetry-unique across {} games '
            '({} duplicate games).'.format(
                placement_diagnostics['distinct_exact_openings'],
                placement_diagnostics['distinct_symmetry_unique_openings'],
                placement_diagnostics['games_recorded'],
                placement_diagnostics['duplicate_game_count'],
            )
        )
        repeated_groups = placement_diagnostics['repeated_exact_labeled_opening_groups']
        if repeated_groups:
            print(
                'Repeated labeled openings: {} group(s); {} identical and {} divergent '
                'standard-play trajectory group(s).'.format(
                    repeated_groups,
                    placement_diagnostics[
                        'repeated_groups_with_identical_standard_trajectory'
                    ],
                    placement_diagnostics[
                        'repeated_groups_with_divergent_standard_trajectories'
                    ],
                )
            )
    if seat_stats is not None:
        print("Seat breakdown:")
        print("  {} as first player: {}".format(
            contestant1_name,
            format_seat_record(seat_stats["contestant1"]["first_player"], include_draws=include_draws),
        ))
        print("  {} as second player: {}".format(
            contestant1_name,
            format_seat_record(seat_stats["contestant1"]["second_player"], include_draws=include_draws),
        ))
        print("  {} as first player: {}".format(
            contestant2_name,
            format_seat_record(seat_stats["contestant2"]["first_player"], include_draws=include_draws),
        ))
        print("  {} as second player: {}".format(
            contestant2_name,
            format_seat_record(seat_stats["contestant2"]["second_player"], include_draws=include_draws),
        ))

    if args.json_out:
        result = {
            'baseline': args.baseline,
            'games': args.games,
            'sims': args.sims,
            'search_mode': args.search_mode,
            'gumbel_max_considered_actions': args.gumbel_max_considered_actions,
            'gumbel_scale': args.gumbel_scale,
            'gumbel_placement_scale': args.gumbel_placement_scale,
            'search_symmetry_evaluation': bool(
                uses_v3 and not args.no_search_symmetry_evaluation
            ),
            'root_symmetry_samples': args.root_symmetry_samples,
            'placement_root_symmetry_samples': args.placement_root_symmetry_samples,
            'action_temp': args.action_temp,
            'checkpoint_folder': args.checkpoint_folder,
            'checkpoint_file': args.checkpoint_file,
            'architecture': args.architecture,
            'opponent_checkpoint_folder': args.opponent_checkpoint_folder,
            'opponent_checkpoint_file': args.opponent_checkpoint_file,
            'opponent_architecture': args.opponent_architecture,
            'opponent_sims': args.opponent_sims or args.sims,
            'opponent_search_mode': args.opponent_search_mode,
            'opponent_gumbel_max_considered_actions': args.opponent_gumbel_max_considered_actions,
            'opponent_gumbel_scale': args.opponent_gumbel_scale,
            'opponent_gumbel_placement_scale': args.opponent_gumbel_placement_scale,
            'placement_only_comparison': args.placement_only_comparison,
            'standard_controller_folder': args.standard_controller_folder,
            'standard_controller_file': args.standard_controller_file,
            'standard_controller_architecture': args.standard_controller_architecture,
            'standard_controller_sims': args.standard_controller_sims or args.sims,
            'standard_controller_search_mode': args.standard_controller_search_mode,
            'standard_controller_gumbel_max_considered_actions': (
                args.standard_controller_gumbel_max_considered_actions
            ),
            'standard_controller_gumbel_scale': args.standard_controller_gumbel_scale,
            'arena_batch_size': args.arena_batch_size,
            'inference_deduplication': bool(
                uses_v3 and not args.no_inference_deduplication
            ),
            'inference_cache_size': args.inference_cache_size,
            'fresh': args.fresh,
            'contestant1_name': contestant1_name,
            'contestant2_name': contestant2_name,
            'opening_book_path': opening_book_path,
            'arena_opening_suite': args.arena_opening_suite,
            'opening_id': args.opening_id,
            'opening_source': args.opening_source,
            'opening_mode': opening_mode,
            'opening_seed': args.opening_seed,
            'opening_position_count': len(opening_boards) if opening_boards is not None else 1,
            'distinct_opening_position_count': (
                len({board.tobytes() for board in opening_boards})
                if opening_boards is not None else 1
            ),
            'contestant1_wins': int(nnet_wins),
            'contestant2_wins': int(opponent_wins),
            'neural_mcts_wins': int(nnet_wins),
            'baseline_wins': int(opponent_wins),
            'draws': int(draws),
        }
        if opening_position is not None:
            result.update({
                'opening_player1_rank': int(opening_position["player1_rank"]),
                'opening_player1': opening_position["player1"],
                'opening_player2': opening_position["player2"],
                'opening_player2_response_rank': int(opening_position["player2_response_rank"]),
            })
            if seat_stats is not None:
                result['seat_stats'] = seat_stats
        if placement_diagnostics is not None:
            result['learned_placement_diagnostics'] = placement_diagnostics
        if inference_diagnostics is not None:
            result['inference_diagnostics'] = inference_diagnostics
        json_dir = os.path.dirname(args.json_out)
        if json_dir:
            os.makedirs(json_dir, exist_ok=True)
        with open(args.json_out, 'w') as f:
            json.dump(result, f, indent=2, sort_keys=True)
        print("Wrote JSON results: {}".format(args.json_out))


if __name__ == "__main__":
    main()
