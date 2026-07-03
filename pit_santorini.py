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


class NeuralMCTSPlayer:
    def __init__(self, game, checkpoint_folder, checkpoint_file, sims, architecture='v2', action_temp=0.0):
        self.game = game
        self.nnet = build_nnet(game, architecture)
        self.nnet.load_checkpoint(checkpoint_folder, checkpoint_file)
        self.mcts_args = dotdict({'numMCTSSims': sims, 'cpuct': 1.0})
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
    if not batched_arena_requested(args):
        return
    if not args.opponent_checkpoint_folder:
        parser.error('--arena-batch-size > 1 currently requires --opponent-checkpoint-folder.')
    if args.action_temp != 0:
        parser.error('--arena-batch-size > 1 requires deterministic play with --action-temp 0.')
    if args.opponent_sims is not None and args.opponent_sims != args.sims:
        parser.error('--arena-batch-size > 1 requires --opponent-sims to match --sims.')


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


def board_from_opening_position(position, random_orientation_enabled=True):
    pieces = np.array(position["pieces"], dtype=int)
    heights = np.zeros_like(pieces, dtype=int)
    board = np.array([pieces, heights], dtype=int)
    if random_orientation_enabled:
        board = random_board_orientation(board)
    return board


def sample_opening_suite(opening_suite_path, count, random_orientation_enabled=True):
    suite = SantoriniOpeningSuite.load(opening_suite_path)
    count = int(count)
    if count <= 0:
        return suite, []

    replace = count > len(suite.positions)
    indices = np.random.choice(len(suite.positions), size=count, replace=replace)
    boards = [
        board_from_opening_position(
            suite.positions[int(index)],
            random_orientation_enabled=random_orientation_enabled,
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

    if args.opening_id is not None and getattr(args, 'arena_opening_suite', None):
        raise ValueError('--opening-id cannot be used with --arena-opening-suite.')

    if getattr(args, 'arena_opening_suite', None):
        suite, opening_boards = sample_opening_suite(
            args.arena_opening_suite,
            int(args.games / 2),
            random_orientation_enabled=not args.no_opening_random_orientation,
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
    parser.add_argument('--checkpoint-folder', default='./temp/santorini_quick/')
    parser.add_argument('--checkpoint-file', default='best.pth.tar')
    parser.add_argument('--architecture', choices=['v1', 'v2', 'v3'], default='v2')
    parser.add_argument('--opponent-checkpoint-folder')
    parser.add_argument('--opponent-checkpoint-file', default='best.pth.tar')
    parser.add_argument('--opponent-architecture', choices=['v1', 'v2', 'v3'], default='v2')
    parser.add_argument('--opponent-sims', type=int)
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
    parser.add_argument('--no-opening-book', action='store_true', help='Use game random starts instead of paired book openings.')
    parser.add_argument('--arena-opening-top-fraction', type=float, default=0.50)
    parser.add_argument('--arena-opening-max-abs-value', type=float, default=0.14)
    parser.add_argument('--no-opening-random-orientation', action='store_true')
    parser.add_argument('--json-out', help='Optional path to write evaluation results as JSON.')
    args = parser.parse_args()

    if args.opening_id is not None and args.no_opening_book:
        parser.error('--opening-id cannot be used with --no-opening-book.')
    if args.opening_id is not None and args.arena_opening_suite:
        parser.error('--opening-id cannot be used with --arena-opening-suite.')
    if args.action_temp < 0:
        parser.error('--action-temp must be non-negative.')
    if args.arena_batch_size < 1:
        parser.error('--arena-batch-size must be at least 1.')
    validate_batched_arena_args(parser, args)

    game = SantoriniGame(5, true_random_placement=True)
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
        )
        opponent_player_obj = NeuralMCTSPlayer(
            game,
            args.opponent_checkpoint_folder,
            args.opponent_checkpoint_file,
            args.opponent_sims or args.sims,
            architecture=args.opponent_architecture,
            action_temp=args.action_temp,
        )
        player1 = nnet_player_obj
        player2 = opponent_player_obj
        contestant2_name = display_name_from_folder(args.opponent_checkpoint_folder)
    else:
        nnet = build_nnet(game, args.architecture)
        if not args.fresh and os.path.exists(checkpoint_path):
            nnet.load_checkpoint(args.checkpoint_folder, args.checkpoint_file)
            print("Loaded checkpoint: {} ({})".format(checkpoint_path, args.architecture))
        else:
            print("Using fresh untrained network.")

        mcts_args = dotdict({'numMCTSSims': args.sims, 'cpuct': 1.0})
        mcts = MCTS(game, nnet, mcts_args)
        player1 = lambda x: select_legal_action(
            game,
            x,
            mcts.getActionProb(x, temp=args.action_temp),
            sample=args.action_temp > 0,
        )
        player2 = build_baseline(game, args.baseline)
        contestant2_name = args.baseline

    if args.action_temp > 0:
        print("Sampling neural MCTS actions with temperature {}.".format(args.action_temp))

    seat_stats = None
    use_batched_arena = batched_arena_requested(args)
    if use_batched_arena:
        print("Using batched arena with batch size {}.".format(args.arena_batch_size))
        arena = BatchedMCTSArena(
            game,
            player1.nnet,
            player2.nnet,
            dotdict({'numMCTSSims': args.sims, 'cpuct': 1.0}),
            batch_size=args.arena_batch_size,
            opening_boards=opening_boards,
            progress_file=sys.stdout,
        )
        nnet_wins, opponent_wins, draws = arena.playGames(args.games)
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
            'action_temp': args.action_temp,
            'checkpoint_folder': args.checkpoint_folder,
            'checkpoint_file': args.checkpoint_file,
            'architecture': args.architecture,
            'opponent_checkpoint_folder': args.opponent_checkpoint_folder,
            'opponent_checkpoint_file': args.opponent_checkpoint_file,
            'opponent_architecture': args.opponent_architecture,
            'opponent_sims': args.opponent_sims or args.sims,
            'arena_batch_size': args.arena_batch_size,
            'fresh': args.fresh,
            'contestant1_name': contestant1_name,
            'contestant2_name': contestant2_name,
            'opening_book_path': opening_book_path,
            'arena_opening_suite': args.arena_opening_suite,
            'opening_id': args.opening_id,
            'opening_mode': opening_mode,
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
        json_dir = os.path.dirname(args.json_out)
        if json_dir:
            os.makedirs(json_dir, exist_ok=True)
        with open(args.json_out, 'w') as f:
            json.dump(result, f, indent=2, sort_keys=True)
        print("Wrote JSON results: {}".format(args.json_out))


if __name__ == "__main__":
    main()
