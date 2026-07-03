import json
import math
import os

import numpy as np


class SantoriniOpeningBook:
    def __init__(self, path, metadata, player1_choices):
        self.path = path
        self.metadata = metadata
        self.player1_choices = player1_choices
        self.positions = self._flatten_positions(player1_choices)
        self.best_response_positions = [
            position
            for position in self.positions
            if position["player2_response_rank"] == 1
        ]

        if not self.positions:
            raise ValueError("Opening book contains no response positions.")

    @classmethod
    def load(cls, path):
        with open(path) as book_file:
            payload = json.load(book_file)

        if "player1_choices" not in payload:
            raise ValueError("Opening book is missing player1_choices.")
        return cls(path, payload.get("metadata", {}), payload["player1_choices"])

    def _flatten_positions(self, player1_choices):
        positions = []
        for choice in player1_choices:
            for response in choice.get("responses", []):
                pieces = np.array(response["pieces"], dtype=int)
                positions.append({
                    "id": response["id"],
                    "pieces": pieces,
                    "value_mean": float(response["value_mean"]),
                    "value_abs": abs(float(response["value_mean"])),
                    "minimax_value": float(choice["minimax_value"]),
                    "player1_rank": int(choice["player1_rank"]),
                    "player1": choice["player1"],
                    "player2": response["player2"],
                    "player2_response_rank": int(response["player2_response_rank"]),
                })
        return positions


class SantoriniOpeningSuite:
    def __init__(self, path, metadata, positions):
        self.path = path
        self.metadata = metadata
        self.positions = positions

        if not self.positions:
            raise ValueError("Opening suite contains no positions.")

    @classmethod
    def load(cls, path):
        with open(path) as suite_file:
            payload = json.load(suite_file)

        if "positions" not in payload:
            raise ValueError("Opening suite is missing positions.")

        positions = []
        for position in payload["positions"]:
            record = dict(position)
            record["pieces"] = np.array(record["pieces"], dtype=int)
            record["value_mean"] = float(record["value_mean"])
            record["value_abs"] = abs(float(record["value_mean"]))
            record["player1_rank"] = int(record["player1_rank"])
            positions.append(record)

        return cls(path, payload.get("metadata", {}), positions)


class SantoriniOpeningSampler:
    def __init__(
        self,
        book,
        arena_suite=None,
        self_play_max_abs_value=0.30,
        self_play_old_filter_probability=0.70,
        self_play_value_probability=0.25,
        self_play_tail_probability=0.05,
        arena_top_fraction=0.50,
        arena_max_abs_value=0.14,
        random_orientation=True,
        rng=None,
    ):
        self.book = book
        self.arena_suite = arena_suite
        self.self_play_max_abs_value = self_play_max_abs_value
        self.self_play_old_filter_probability = self_play_old_filter_probability
        self.self_play_value_probability = self_play_value_probability
        self.self_play_tail_probability = self_play_tail_probability
        self.arena_top_fraction = arena_top_fraction
        self.arena_max_abs_value = arena_max_abs_value
        self.random_orientation = random_orientation
        self.rng = rng if rng is not None else np.random
        self._self_play_value_candidates_cache = None
        self._old_filter_candidates_cache = None
        self._arena_candidates_cache = None

    @classmethod
    def load(cls, path, **kwargs):
        return cls(SantoriniOpeningBook.load(path), **kwargs)

    def sample_self_play_board(self):
        total_probability = (
            self.self_play_old_filter_probability
            + self.self_play_value_probability
            + self.self_play_tail_probability
        )
        roll = self.rng.random() * total_probability if total_probability > 0 else 0.0
        old_filter_cutoff = self.self_play_old_filter_probability
        value_cutoff = old_filter_cutoff + self.self_play_value_probability

        if roll < old_filter_cutoff:
            candidates = self._old_filter_candidates()
        elif roll < value_cutoff:
            candidates = self._self_play_value_candidates()
        else:
            candidates = self.book.positions

        if not candidates:
            candidates = self._self_play_value_candidates()
        if not candidates:
            candidates = self.book.positions

        position = candidates[int(self.rng.randint(len(candidates)))]
        return self._board_from_position(position)

    def sample_arena_suite(self, count):
        count = int(count)
        if count <= 0:
            return []

        candidates = self._arena_candidates()
        if not candidates:
            candidates = self.book.positions

        replace = count > len(candidates)
        probabilities = self._arena_probabilities(candidates)
        indices = self.rng.choice(
            len(candidates),
            size=count,
            replace=replace,
            p=probabilities,
        )
        return [self._board_from_position(candidates[int(index)]) for index in indices]

    def _self_play_value_candidates(self):
        if self._self_play_value_candidates_cache is None:
            self._self_play_value_candidates_cache = [
                position
                for position in self.book.positions
                if position["value_abs"] <= self.self_play_max_abs_value
            ]
        return self._self_play_value_candidates_cache

    def _old_filter_candidates(self):
        if self._old_filter_candidates_cache is None:
            self._old_filter_candidates_cache = [
                position
                for position in self.book.positions
                if passes_old_opening_filter(position["pieces"])
            ]
        return self._old_filter_candidates_cache

    def _arena_candidates(self):
        if self.arena_suite is not None:
            return self.arena_suite.positions

        if self._arena_candidates_cache is not None:
            return self._arena_candidates_cache

        max_rank = max(position["player1_rank"] for position in self.book.positions)
        rank_cutoff = max(1, int(math.ceil(max_rank * self.arena_top_fraction)))
        self._arena_candidates_cache = [
            position
            for position in self.book.positions
            if position["player1_rank"] <= rank_cutoff
            and position["value_abs"] <= self.arena_max_abs_value
        ]
        return self._arena_candidates_cache

    def _arena_probabilities(self, candidates):
        if self.arena_suite is not None:
            return None

        ranks = np.array([position["player1_rank"] for position in candidates], dtype=np.float64)
        weights = 1.0 / ranks
        return weights / np.sum(weights)

    def _board_from_position(self, position):
        heights = np.zeros_like(position["pieces"], dtype=int)
        board = np.array([position["pieces"], heights], dtype=int)
        if self.random_orientation:
            board = random_board_orientation(board, self.rng)
        return board


def is_outer_square(location, board_size):
    row, col = location
    return row == 0 or col == 0 or row == board_size - 1 or col == board_size - 1


def worker_locations(pieces, player):
    return list(zip(*np.where(np.sign(pieces) == player)))


def passes_old_opening_filter(pieces):
    board_size = pieces.shape[0]
    for player in (1, -1):
        locations = worker_locations(pieces, player)
        if len(locations) == 2 and all(is_outer_square(location, board_size) for location in locations):
            return False
    return True


def random_board_orientation(board, rng=None):
    rng = rng if rng is not None else np.random
    rotations = int(rng.randint(4))
    flip = bool(rng.randint(2))
    transformed = np.array([
        np.rot90(board[0], rotations),
        np.rot90(board[1], rotations),
    ])
    if flip:
        transformed = np.array([
            np.fliplr(transformed[0]),
            np.fliplr(transformed[1]),
        ])
    return transformed


def find_opening_book(candidates):
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None
