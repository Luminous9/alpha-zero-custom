import argparse
import json
import os
import re

import numpy as np

from play_santorini import DEFAULT_CHECKPOINT_FOLDER, describe_action
from santorini.SantoriniGame import SantoriniGame
from santorini.pytorch.NNet import build_nnet


ROW_SPLIT_RE = re.compile(r"\s*[;/]\s*")
VALUE_SPLIT_RE = re.compile(r"[\s,]+")


def parse_grid(text, name, board_size):
    rows = [row.strip() for row in ROW_SPLIT_RE.split(text.strip()) if row.strip()]
    if len(rows) != board_size:
        raise ValueError("{} must have exactly {} rows.".format(name, board_size))

    grid = []
    for row_number, row in enumerate(rows, start=1):
        values = [value for value in VALUE_SPLIT_RE.split(row) if value]
        if len(values) != board_size:
            raise ValueError(
                "{} row {} must have exactly {} values.".format(name, row_number, board_size)
            )
        try:
            grid.append([int(value) for value in values])
        except ValueError:
            raise ValueError("{} row {} contains a non-integer value.".format(name, row_number))

    return np.array(grid, dtype=int)


def load_board_from_json(path, board_size):
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        board = np.array(data, dtype=int)
        if board.shape != (2, board_size, board_size):
            raise ValueError(
                "JSON board list must have shape (2, {}, {}).".format(board_size, board_size)
            )
        return board

    if not isinstance(data, dict):
        raise ValueError("JSON board must be either a 2-layer list or an object.")

    try:
        pieces = np.array(data["pieces"], dtype=int)
        heights = np.array(data["heights"], dtype=int)
    except KeyError as error:
        raise ValueError("JSON board object is missing key {}.".format(error))

    return make_board(pieces, heights, board_size)


def make_board(pieces, heights, board_size):
    if pieces.shape != (board_size, board_size):
        raise ValueError("pieces must have shape ({}, {}).".format(board_size, board_size))
    if heights.shape != (board_size, board_size):
        raise ValueError("heights must have shape ({}, {}).".format(board_size, board_size))
    return np.array([pieces, heights], dtype=int)


def validate_board(board):
    pieces = board[0]
    heights = board[1]

    piece_values = set(np.unique(pieces))
    invalid_pieces = piece_values - {-2, -1, 0, 1, 2}
    if invalid_pieces:
        raise ValueError("pieces contains invalid values: {}".format(sorted(invalid_pieces)))

    for worker in [1, 2, -1, -2]:
        count = int(np.sum(pieces == worker))
        if count != 1:
            raise ValueError("pieces must contain worker {} exactly once; found {}.".format(worker, count))

    if np.any(heights < 0) or np.any(heights > 4):
        raise ValueError("heights must be between 0 and 4.")

    if np.any((pieces != 0) & (heights >= 4)):
        raise ValueError("workers cannot stand on domes, which are represented by height 4.")


def build_board(args):
    if args.board_json:
        return load_board_from_json(args.board_json, args.board_size)

    if args.pieces is None or args.heights is None:
        raise ValueError("provide either --board-json, or both --pieces and --heights.")

    pieces = parse_grid(args.pieces, "pieces", args.board_size)
    heights = parse_grid(args.heights, "heights", args.board_size)
    return make_board(pieces, heights, args.board_size)


def decode_top_actions(game, canonical_board, policy, top_actions):
    if top_actions <= 0:
        return []

    valids = game.getValidMoves(canonical_board, 1)
    action_ids = np.flatnonzero(valids)
    ranked = sorted(action_ids, key=lambda action: float(policy[action]), reverse=True)

    return [
        {
            "action": int(action),
            "policy": float(policy[action]),
            "move": describe_action(game, canonical_board, 1, int(action), 1),
        }
        for action in ranked[:top_actions]
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a Santorini position with a trained neural net.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python evaluate_santorini.py --player 1 \\
    --pieces "0 0 0 0 0; 0 0 1 0 0; 0 -1 0 -2 0; 0 0 2 0 0; 0 0 0 0 0" \\
    --heights "0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0; 0 0 0 0 0"

  python evaluate_santorini.py --player -1 --board-json position.json --top-actions 5

Board format:
  --pieces uses 1 and 2 for player +1 workers, -1 and -2 for player -1 workers.
  --heights uses 0..4, where 4 is a dome.
  Rows may be separated by semicolons or slashes; values may be separated by spaces or commas.
""",
    )
    parser.add_argument("--player", type=int, choices=[1, -1], required=True, help="Player to move.")
    parser.add_argument("--pieces", help="5 row piece grid, quoted as one string.")
    parser.add_argument("--heights", help="5 row height grid, quoted as one string.")
    parser.add_argument("--board-json", help="Path to JSON containing either [pieces, heights] or {pieces, heights}.")
    parser.add_argument("--board-size", type=int, default=5)
    parser.add_argument("--checkpoint-folder", default=DEFAULT_CHECKPOINT_FOLDER)
    parser.add_argument("--checkpoint-file", default="best.pth.tar")
    parser.add_argument("--architecture", choices=["v2", "v3"], default="v2")
    parser.add_argument("--top-actions", type=int, default=0, help="Print the top N legal policy moves.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main():
    try:
        args = parse_args()
        if args.board_json and (args.pieces or args.heights):
            raise ValueError("use --board-json or --pieces/--heights, not both.")

        checkpoint_path = os.path.join(args.checkpoint_folder, args.checkpoint_file)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError("No model in path {}".format(checkpoint_path))

        board = build_board(args)
        validate_board(board)

        game = SantoriniGame(args.board_size, sequential_placement=args.architecture == 'v3')
        nnet = build_nnet(game, args.architecture)
        nnet.load_checkpoint(args.checkpoint_folder, args.checkpoint_file)

        canonical_board = game.getCanonicalForm(board, args.player)
        policy, value = nnet.predict(canonical_board)
        top_actions = decode_top_actions(game, canonical_board, policy, args.top_actions)

        result = {
            "checkpoint": checkpoint_path,
            "player_to_move": args.player,
            "value": float(value),
            "value_perspective": "player_to_move",
            "top_actions": top_actions,
        }

        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
            return

        print("Checkpoint: {}".format(checkpoint_path))
        print("Player to move: {}".format(args.player))
        print("Value for player to move: {:.6f}".format(value))
        if top_actions:
            print("Top legal policy moves:")
            for item in top_actions:
                print("  {move}: action={action} policy={policy:.6f}".format(**item))
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit("error: {}".format(error))


if __name__ == "__main__":
    main()
