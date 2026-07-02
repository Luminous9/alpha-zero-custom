import argparse
import json
import os
from datetime import datetime, timezone
from itertools import combinations

import numpy as np

from santorini.pytorch.NNet import NNetWrapper
from santorini.SantoriniGame import SantoriniGame
from santorini.SantoriniPlayers import coordinate_label

DEFAULT_CHECKPOINT_FOLDER = "./temp/santorini_bootstrap_result"
DEFAULT_OUTPUT_DIR = "./temp/santorini_opening_books"
LABELINGS = ((False, False), (False, True), (True, False), (True, True))


def transform_location(location, board_size, rotation, flip):
    row, col = location

    for _ in range(rotation):
        row, col = col, board_size - 1 - row

    if flip:
        col = board_size - 1 - col

    return row, col


def normalize_locations(locations):
    return tuple(sorted(locations))


def all_transforms():
    for rotation in range(4):
        for flip in (False, True):
            yield rotation, flip


def canonical_locations(locations, board_size):
    return min(
        normalize_locations(
            transform_location(location, board_size, rotation, flip)
            for location in locations
        )
        for rotation, flip in all_transforms()
    )


def canonical_key(p1_locations, p2_locations, board_size):
    keys = []
    for rotation, flip in all_transforms():
        transformed_p1 = normalize_locations(
            transform_location(location, board_size, rotation, flip)
            for location in p1_locations
        )
        transformed_p2 = normalize_locations(
            transform_location(location, board_size, rotation, flip)
            for location in p2_locations
        )
        keys.append((transformed_p1, transformed_p2))
    return min(keys)


def stabilizer_transforms(locations, board_size):
    normalized = normalize_locations(locations)
    return [
        (rotation, flip)
        for rotation, flip in all_transforms()
        if normalize_locations(
            transform_location(location, board_size, rotation, flip)
            for location in normalized
        ) == normalized
    ]


def canonical_response_for_player1(p2_locations, p1_locations, board_size):
    return min(
        normalize_locations(
            transform_location(location, board_size, rotation, flip)
            for location in p2_locations
        )
        for rotation, flip in stabilizer_transforms(p1_locations, board_size)
    )


def iter_player1_choices(board_size):
    squares = [(row, col) for row in range(board_size) for col in range(board_size)]
    seen = set()
    for locations in combinations(squares, 2):
        key = canonical_locations(locations, board_size)
        if key in seen:
            continue
        seen.add(key)
        yield key


def iter_player2_responses(p1_locations, board_size):
    p1_locations = normalize_locations(p1_locations)
    squares = [(row, col) for row in range(board_size) for col in range(board_size)]
    remaining = [square for square in squares if square not in p1_locations]
    seen = set()
    for locations in combinations(remaining, 2):
        key = canonical_response_for_player1(locations, p1_locations, board_size)
        if key in seen:
            continue
        seen.add(key)
        yield key


def iter_opening_positions(board_size):
    for p1_locations in iter_player1_choices(board_size):
        for p2_locations in iter_player2_responses(p1_locations, board_size):
            yield p1_locations, p2_locations


def make_board(p1_locations, p2_locations, p1_swap=False, p2_swap=False, board_size=5):
    p1 = list(p1_locations)
    p2 = list(p2_locations)
    if p1_swap:
        p1.reverse()
    if p2_swap:
        p2.reverse()

    board = np.zeros((2, board_size, board_size), dtype=int)
    board[0][p1[0]] = 1
    board[0][p1[1]] = 2
    board[0][p2[0]] = -1
    board[0][p2[1]] = -2
    return board


def location_labels(locations):
    return [coordinate_label(location) for location in locations]


def board_record(position_id, p1_locations, p2_locations, value, board_size, label_values=None):
    if label_values is None:
        raw_value = np.asarray(value, dtype=np.float32)
        if raw_value.ndim == 0:
            values = np.asarray([float(raw_value)], dtype=np.float32)
        else:
            values = raw_value.reshape(-1)
            label_values = values
    else:
        values = np.asarray(label_values, dtype=np.float32)

    value_mean = float(np.mean(values))
    value_std = float(np.std(values))
    value_abs = abs(value_mean)
    score = value_abs + value_std
    selected_label_index = int(np.argmin(np.abs(values - value_mean)))
    if len(values) == len(LABELINGS):
        selected_p1_swap, selected_p2_swap = LABELINGS[selected_label_index]
    else:
        selected_p1_swap, selected_p2_swap = False, False
    selected_value = float(values[selected_label_index])
    return {
        "id": position_id,
        "player1": location_labels(p1_locations),
        "player2": location_labels(p2_locations),
        "pieces": make_board(
            p1_locations,
            p2_locations,
            p1_swap=selected_p1_swap,
            p2_swap=selected_p2_swap,
            board_size=board_size,
        )[0].tolist(),
        "selected_label_index": selected_label_index,
        "selected_p1_swap": selected_p1_swap,
        "selected_p2_swap": selected_p2_swap,
        "selected_value": selected_value,
        "selected_value_abs": abs(selected_value),
        "value_mean": value_mean,
        "value_abs": value_abs,
        "value_std": value_std,
        "score": score,
        "label_invariance_checked": label_values is not None,
        "label_values": [float(value) for value in values],
    }


def evaluate_openings(nnet, board_size, batch_size, max_positions=None, check_label_invariance=False):
    records_by_player1 = {}
    pending_boards = []
    pending_specs = []

    def flush():
        if not pending_boards:
            return

        _, values = nnet.predict_batch(pending_boards)
        offset = 0
        for position_id, p1_locations, p2_locations, label_count in pending_specs:
            position_values = values[offset:offset + label_count]
            offset += label_count
            record = board_record(
                position_id,
                p1_locations,
                p2_locations,
                float(position_values[0]),
                board_size,
                label_values=position_values if check_label_invariance else None,
            )
            records_by_player1.setdefault(p1_locations, []).append(record)

        pending_boards.clear()
        pending_specs.clear()

    for position_id, (p1_locations, p2_locations) in enumerate(iter_opening_positions(board_size), start=1):
        if max_positions is not None and position_id > max_positions:
            break

        labelings = LABELINGS if check_label_invariance else ((False, False),)
        pending_specs.append((position_id, p1_locations, p2_locations, len(labelings)))
        for p1_swap, p2_swap in labelings:
            pending_boards.append(
                make_board(
                    p1_locations,
                    p2_locations,
                    p1_swap=p1_swap,
                    p2_swap=p2_swap,
                    board_size=board_size,
                )
            )

        if len(pending_boards) >= batch_size:
            flush()

    flush()
    return build_minimax_book_records(records_by_player1)


def build_minimax_book_records(records_by_player1):
    choices = []
    for player1_locations, responses in records_by_player1.items():
        responses.sort(key=lambda record: (record["value_mean"], record["value_std"], record["id"]))
        for response_rank, response in enumerate(responses, start=1):
            response["player2_response_rank"] = response_rank

        best_response = responses[0]
        choices.append({
            "player1": location_labels(player1_locations),
            "player1_locations": [list(location) for location in player1_locations],
            "minimax_value": best_response["value_mean"],
            "best_response_id": best_response["id"],
            "response_count": len(responses),
            "responses": responses,
        })

    choices.sort(key=lambda choice: (-choice["minimax_value"], choice["player1"]))
    for player1_rank, choice in enumerate(choices, start=1):
        choice["player1_rank"] = player1_rank

    return choices


def write_json(path, payload):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w") as f:
        f.write(format_json(payload))


def checkpoint_output_dir(output_dir, checkpoint_folder):
    normalized = os.path.normpath(checkpoint_folder)
    checkpoint_name = os.path.basename(normalized)
    if checkpoint_name in ("", os.curdir, os.pardir):
        checkpoint_name = "checkpoint"
    return os.path.join(output_dir, checkpoint_name)


def format_json(payload):
    return _compact_piece_rows(json.dumps(payload, indent=2, sort_keys=True)) + "\n"


def _compact_piece_rows(text):
    lines = text.splitlines()
    compacted = []
    i = 0

    while i < len(lines):
        compacted.append(lines[i])
        if lines[i].strip() != '"pieces": [':
            i += 1
            continue

        i += 1
        while i < len(lines):
            stripped = lines[i].strip()
            if stripped == '[':
                row_indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
                row_values = []
                i += 1
                while i < len(lines):
                    value = lines[i].strip()
                    if value.startswith(']'):
                        compacted.append(
                            "{}[{}]{}".format(
                                row_indent,
                                ", ".join(row_values),
                                "," if value.endswith(",") else "",
                            )
                        )
                        i += 1
                        break
                    row_values.append(value.rstrip(","))
                    i += 1
                continue

            compacted.append(lines[i])
            i += 1
            if stripped in (']', '],'):
                break

    return "\n".join(compacted)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a symmetry-reduced Santorini opening book ranked by a model value head.",
    )
    parser.add_argument("--board-size", type=int, default=5)
    parser.add_argument("--checkpoint-folder", default=DEFAULT_CHECKPOINT_FOLDER)
    parser.add_argument("--checkpoint-file", default="best.pth.tar")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-positions", type=int, help="Optional smoke-test limit.")
    parser.add_argument(
        "--check-label-invariance",
        action="store_true",
        help="Evaluate all four same-color worker labelings and report value variance.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = os.path.join(args.checkpoint_folder, args.checkpoint_file)
    if not os.path.exists(checkpoint_path):
        raise SystemExit("error: no model in path {}".format(checkpoint_path))

    game = SantoriniGame(args.board_size)
    nnet = NNetWrapper(game)
    nnet.load_checkpoint(args.checkpoint_folder, args.checkpoint_file)

    player1_choices = evaluate_openings(
        nnet,
        board_size=args.board_size,
        batch_size=args.batch_size,
        max_positions=args.max_positions,
        check_label_invariance=args.check_label_invariance,
    )

    metadata = {
        "board_size": args.board_size,
        "checkpoint": checkpoint_path,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "player1_choice_count": len(player1_choices),
        "position_count": sum(choice["response_count"] for choice in player1_choices),
        "symmetry": (
            "Player 1 choices are D4-canonical; Player 2 responses are canonical under symmetries "
            "that preserve Player 1's placement."
        ),
        "ranking": (
            "Player 2 responses sort by value_mean ascending. Player 1 choices sort by minimax_value descending."
        ),
        "value_perspective": "player +1 after both players have placed workers",
        "label_value_policy": (
            "V2 default evaluates one anonymous-worker canonical labeling. "
            "Use --check-label-invariance to evaluate all four same-color worker labelings."
        ),
        "label_invariance_checked": args.check_label_invariance,
        "recommended_use": (
            "Use high-minimax openings for strong-player-like self-play sampling. "
            "For arena suites, sample from this book while filtering out very lopsided values, "
            "then play each selected opening once from each side."
        ),
    }
    full_book = {
        "metadata": metadata,
        "player1_choices": player1_choices,
    }

    book_output_dir = checkpoint_output_dir(args.output_dir, args.checkpoint_folder)
    full_book_path = os.path.join(book_output_dir, "opening_book.json")
    write_json(full_book_path, full_book)

    print(
        "Evaluated {} player-1 choices and {} response positions.".format(
            len(player1_choices),
            metadata["position_count"],
        )
    )
    print("Wrote full opening book: {}".format(full_book_path))
    if player1_choices:
        best = player1_choices[0]
        print(
            "Best minimax P1: rank={player1_rank} minimax_value={minimax_value:.6f} "
            "P1={player1} best_response_id={best_response_id}".format(**best)
        )


if __name__ == "__main__":
    main()
