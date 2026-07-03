import argparse
import os

import numpy as np

from pit_santorini import NeuralMCTSPlayer
from santorini.SantoriniGame import SantoriniGame
from santorini.SantoriniOpeningBook import SantoriniOpeningBook, find_opening_book
from santorini.SantoriniPlayers import (
    HumanSantoriniPlayer,
    SANTORINI_DIRECTIONS,
    coordinate_label,
    parse_coordinate,
)


DEFAULT_CHECKPOINT_FOLDER = './temp/santorini_colab_training2'
DEFAULT_OPENING_BOOK_DIR = './temp/santorini_opening_books'


def parse_args():
    parser = argparse.ArgumentParser(description='Play Santorini against a trained neural MCTS player.')
    parser.add_argument('--checkpoint-folder', default=DEFAULT_CHECKPOINT_FOLDER)
    parser.add_argument('--checkpoint-file', default='best.pth.tar')
    parser.add_argument('--sims', type=int, default=1024, help='MCTS simulations per AI move.')
    parser.add_argument('--human-first', action='store_true', help='Let the human play first.')
    parser.add_argument('--ai-first', action='store_true', help='Let the AI play first.')
    parser.add_argument('--opening-book', help='Optional opening book JSON path for worker placement.')
    parser.add_argument(
        '--fixed-start',
        action='store_true',
        help='Skip worker placement and use the legacy centered start.',
    )
    return parser.parse_args()


def opening_book_candidates(checkpoint_folder, explicit_path=None):
    checkpoint_name = os.path.basename(os.path.normpath(checkpoint_folder))
    return [
        explicit_path,
        os.path.join(DEFAULT_OPENING_BOOK_DIR, checkpoint_name, 'opening_book.json'),
        os.path.join(checkpoint_folder, 'opening_book.json'),
        os.path.join(checkpoint_folder, 'opening_books', 'opening_book.json'),
    ]


def load_opening_book(checkpoint_folder, explicit_path=None):
    opening_book_path = find_opening_book(opening_book_candidates(checkpoint_folder, explicit_path))
    if opening_book_path is None:
        candidates = [
            candidate
            for candidate in opening_book_candidates(checkpoint_folder, explicit_path)
            if candidate
        ]
        raise FileNotFoundError(
            "No opening book found. Checked: {}".format(", ".join(candidates))
        )
    return SantoriniOpeningBook.load(opening_book_path), opening_book_path


def transform_location(location, board_size, rotation, flip):
    row, col = location
    for _ in range(rotation):
        row, col = col, board_size - 1 - row
    if flip:
        col = board_size - 1 - col
    return row, col


def inverse_transform_location(location, board_size, rotation, flip):
    for row in range(board_size):
        for col in range(board_size):
            candidate = (row, col)
            if transform_location(candidate, board_size, rotation, flip) == tuple(location):
                return candidate
    raise ValueError("Could not invert location {}".format(location))


def normalize_locations(locations):
    return tuple(sorted(tuple(location) for location in locations))


def locations_from_labels(labels, board_size):
    return [parse_coordinate(label, board_size) for label in labels]


def choice_player1_locations(choice, board_size):
    if "player1_locations" in choice:
        return [tuple(location) for location in choice["player1_locations"]]
    return locations_from_labels(choice["player1"], board_size)


def response_player2_locations(response):
    pieces = np.asarray(response["pieces"])
    locations = []
    for piece in (-1, -2):
        matches = np.argwhere(pieces == piece)
        if len(matches) != 1:
            raise ValueError("Opening response {} has invalid worker labels.".format(response.get("id")))
        locations.append(tuple(int(value) for value in matches[0]))
    return locations


class OpeningBookPlacementSelector:
    def __init__(self, book, board_size):
        self.book = book
        self.board_size = board_size
        self.choices_by_player1 = {
            normalize_locations(choice_player1_locations(choice, board_size)): choice
            for choice in book.player1_choices
        }

    def best_player1_placement(self):
        choice = min(
            self.book.player1_choices,
            key=lambda item: (int(item["player1_rank"]), -float(item["minimax_value"])),
        )
        return choice_player1_locations(choice, self.board_size), choice

    def best_response_to_player1(self, player1_locations):
        for rotation in range(4):
            for flip in (False, True):
                transformed_player1 = normalize_locations(
                    transform_location(location, self.board_size, rotation, flip)
                    for location in player1_locations
                )
                choice = self.choices_by_player1.get(transformed_player1)
                if choice is None:
                    continue

                response = min(
                    choice["responses"],
                    key=lambda item: (
                        int(item["player2_response_rank"]),
                        float(item["value_mean"]),
                        int(item["id"]),
                    ),
                )
                response_locations = [
                    inverse_transform_location(location, self.board_size, rotation, flip)
                    for location in response_player2_locations(response)
                ]
                return response_locations, choice, response

        raise ValueError(
            "Opening book has no player-2 response for Player 1 placement {}.".format(
                ", ".join(coordinate_label(location) for location in player1_locations)
            )
        )


def empty_placement_board(board_size):
    return np.zeros((2, board_size, board_size), dtype=int)


def place_workers(board, player, locations):
    board = board.copy()
    board[0][tuple(locations[0])] = player
    board[0][tuple(locations[1])] = 2 * player
    return board


def parse_placement(text, board_size, occupied_locations=()):
    parts = text.replace(',', ' ').split()
    if len(parts) != 2:
        raise ValueError("use exactly two coordinates, like B2 D3")

    locations = [parse_coordinate(part, board_size) for part in parts]
    if locations[0] == locations[1]:
        raise ValueError("workers must be placed on two different squares")

    occupied = set(tuple(location) for location in occupied_locations)
    blocked = [location for location in locations if location in occupied]
    if blocked:
        raise ValueError("{} is already occupied".format(coordinate_label(blocked[0])))
    return locations


def prompt_human_placement(game, occupied_locations=()):
    if occupied_locations:
        occupied = ", ".join(coordinate_label(location) for location in occupied_locations)
        print("Occupied squares: {}.".format(occupied))

    while True:
        entered = input("\nPlace your workers as '<O-square> <U-square>' (example: B2 D3), or q to quit: ").strip()
        if entered.lower() in ('q', 'quit', 'exit'):
            raise KeyboardInterrupt
        try:
            return parse_placement(entered, game.n, occupied_locations=occupied_locations)
        except ValueError as error:
            print("Sorry, {}".format(error))


def placement_worker_names(player, human_player):
    return ('O', 'U') if player == human_player else ('X', 'Y')


def describe_placement(player, locations, human_player):
    names = placement_worker_names(player, human_player)
    return "{0} at {1}, {2} at {3}".format(
        names[0],
        coordinate_label(locations[0]),
        names[1],
        coordinate_label(locations[1]),
    )


def run_worker_placement(game, selector, human_player):
    board = empty_placement_board(game.n)

    print("\nWorker placement phase")
    if human_player == 1:
        human_locations = prompt_human_placement(game)
        board = place_workers(board, 1, human_locations)
        print("Human places {}.".format(describe_placement(1, human_locations, human_player)))

        ai_locations, choice, response = selector.best_response_to_player1(human_locations)
        board = place_workers(board, -1, ai_locations)
        print(
            "AI responds with {} (book P1 rank {}, response rank {}).".format(
                describe_placement(-1, ai_locations, human_player),
                choice["player1_rank"],
                response["player2_response_rank"],
            )
        )
    else:
        ai_locations, choice = selector.best_player1_placement()
        board = place_workers(board, 1, ai_locations)
        print(
            "AI opens with {} (book P1 rank {}).".format(
                describe_placement(1, ai_locations, human_player),
                choice["player1_rank"],
            )
        )
        print("\nBoard from your perspective after AI placement:")
        SantoriniGame.display(game.getCanonicalForm(board, human_player))

        human_locations = prompt_human_placement(game, occupied_locations=ai_locations)
        board = place_workers(board, -1, human_locations)
        print("Human places {}.".format(describe_placement(-1, human_locations, human_player)))

    print("\nInitial board from your perspective:")
    SantoriniGame.display(game.getCanonicalForm(board, human_player))
    return board


def describe_action(game, board, player, action, perspective_player):
    origin, move_direction, build_direction = game.decodeAction(action)

    worker_idx = game.getCharacterLocations(board, player).index(origin)
    move_delta = SANTORINI_DIRECTIONS[move_direction]
    build_delta = SANTORINI_DIRECTIONS[build_direction]
    move = (origin[0] + move_delta[0], origin[1] + move_delta[1])
    build = (move[0] + build_delta[0], move[1] + build_delta[1])

    if player == perspective_player:
        worker = ('O', 'U')[worker_idx]
    else:
        worker = ('X', 'Y')[worker_idx]
    return "{} {} {}".format(worker, coordinate_label(move), coordinate_label(build))


def play_game(game, human, ai, human_player, initial_board=None):
    board = initial_board.copy() if initial_board is not None else game.getInitBoard()
    cur_player = 1
    turn = 0

    ai.startGame()

    while game.getGameEnded(board, cur_player) == 0:
        turn += 1
        actor = 'Human' if cur_player == human_player else 'AI'
        print("\nTurn {}: {} to move".format(turn, actor))
        print("Board from your perspective: pieces first, then tower heights.")
        SantoriniGame.display(game.getCanonicalForm(board, human_player))

        canonical_board = game.getCanonicalForm(board, cur_player)
        if cur_player == human_player:
            action = human(canonical_board)
        else:
            print("AI thinking...")
            action = ai(canonical_board)
            print("AI plays {}.".format(describe_action(game, board, cur_player, action, human_player)))

        valids = game.getValidMoves(canonical_board, 1)
        if valids[action] == 0:
            raise AssertionError("Player returned illegal action {}".format(action))
        board, cur_player = game.getNextState(board, cur_player, action)

    result = cur_player * game.getGameEnded(board, cur_player)
    print("\nFinal board from your perspective:")
    SantoriniGame.display(game.getCanonicalForm(board, human_player))
    return result


def main():
    args = parse_args()
    if args.human_first and args.ai_first:
        raise ValueError('Choose at most one of --human-first or --ai-first.')

    checkpoint_path = os.path.join(args.checkpoint_folder, args.checkpoint_file)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("No model in path {}".format(checkpoint_path))

    game = SantoriniGame(5)
    human = HumanSantoriniPlayer(game).play
    ai = NeuralMCTSPlayer(game, args.checkpoint_folder, args.checkpoint_file, args.sims)

    human_starts = args.human_first or not args.ai_first
    human_player = 1 if human_starts else -1

    print("Loaded checkpoint: {}".format(checkpoint_path))
    print("AI MCTS sims per move: {}".format(args.sims))
    print("Human is Player {}.".format(human_player))
    print("Your workers are O and U. AI workers are X and Y.")
    print("Coordinates use lettered columns and numbered rows; the top-left corner is A1.")

    try:
        if args.fixed_start:
            initial_board = game.getInitBoard()
        else:
            opening_book, opening_book_path = load_opening_book(args.checkpoint_folder, args.opening_book)
            print("Loaded opening book: {}".format(opening_book_path))
            selector = OpeningBookPlacementSelector(opening_book, game.n)
            initial_board = run_worker_placement(game, selector, human_player)

        result = play_game(game, human, ai, human_player, initial_board=initial_board)
    except KeyboardInterrupt:
        print("\nGame aborted.")
        return

    if result == human_player:
        winner = 'Human'
    elif result == -human_player:
        winner = 'AI'
    else:
        winner = 'Nobody'
    print("Winner: {}".format(winner))


if __name__ == '__main__':
    main()
