import json
import math
import os
from itertools import combinations

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


class SantoriniRandomOpeningSampler:
    _positions_cache = {}

    def __init__(self, board_size=5, random_orientation=True, rng=None):
        self.board_size = board_size
        self.random_orientation = random_orientation
        self.rng = rng if rng is not None else np.random
        self.positions = unique_opening_positions(board_size)

    def sample_self_play_board(self):
        return self._sample_board()

    def sample_arena_suite(self, count):
        count = int(count)
        if count <= 0:
            return []
        return [self._sample_board() for _ in range(count)]

    def sample_distinct_arena_suite(self, count):
        """Sample symmetry-distinct completed openings without replacement."""
        count = int(count)
        if count <= 0:
            return []
        if count > len(self.positions):
            raise ValueError(
                'Requested {} distinct openings, but only {} are available.'.format(
                    count,
                    len(self.positions),
                )
            )
        indices = self.rng.choice(len(self.positions), size=count, replace=False)
        return [self._board_from_index(int(index)) for index in indices]

    def _sample_board(self):
        return self._board_from_index(int(self.rng.randint(len(self.positions))))

    def _board_from_index(self, index):
        p1_locations, p2_locations = self.positions[index]
        board = opening_board_from_locations(p1_locations, p2_locations, self.board_size)
        if self.random_orientation:
            board = random_board_orientation(board, self.rng)
        return board


class SantoriniMixedOpeningSampler:
    def __init__(self, primary_sampler, unique_sampler, unique_probability=0.20, rng=None):
        if unique_probability < 0.0 or unique_probability > 1.0:
            raise ValueError("unique_probability must be between 0 and 1.")
        self.primary_sampler = primary_sampler
        self.unique_sampler = unique_sampler
        self.unique_probability = unique_probability
        self.rng = rng if rng is not None else np.random

    def sample_self_play_board(self):
        return self._sample_sampler().sample_self_play_board()

    def sample_arena_suite(self, count):
        count = int(count)
        if count <= 0:
            return []
        unique_count = int(round(count * self.unique_probability))
        primary_count = count - unique_count
        boards = (
            self.primary_sampler.sample_arena_suite(primary_count)
            + self.unique_sampler.sample_arena_suite(unique_count)
        )
        self.rng.shuffle(boards)
        return boards

    def _sample_sampler(self):
        if self.rng.random() < self.unique_probability:
            return self.unique_sampler
        return self.primary_sampler


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


def iter_unique_player1_choices(board_size):
    squares = [(row, col) for row in range(board_size) for col in range(board_size)]
    seen = set()
    for locations in combinations(squares, 2):
        key = canonical_locations(locations, board_size)
        if key in seen:
            continue
        seen.add(key)
        yield key


def iter_unique_player2_responses(p1_locations, board_size):
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


def iter_unique_opening_positions(board_size):
    for p1_locations in iter_unique_player1_choices(board_size):
        for p2_locations in iter_unique_player2_responses(p1_locations, board_size):
            yield p1_locations, p2_locations


def unique_opening_positions(board_size):
    cache = SantoriniRandomOpeningSampler._positions_cache
    if board_size not in cache:
        cache[board_size] = list(iter_unique_opening_positions(board_size))
    return cache[board_size]


def opening_board_from_locations(p1_locations, p2_locations, board_size=5):
    p1 = list(p1_locations)
    p2 = list(p2_locations)
    board = np.zeros((2, board_size, board_size), dtype=int)
    board[0][p1[0]] = 1
    board[0][p1[1]] = 2
    board[0][p2[0]] = -1
    board[0][p2[1]] = -2
    return board


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
