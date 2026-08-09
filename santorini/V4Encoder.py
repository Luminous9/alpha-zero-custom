"""Rule-derived, D4-covariant 13-plane input encoding for Santorini V4."""

import numpy as np


INPUT_CHANNELS = 13
PHASE_BUILD_CLIP = 40.0
DIRECTIONS = (
    (-1, -1), (-1, 0), (-1, 1), (0, -1),
    (0, 1), (1, -1), (1, 0), (1, 1),
)


def _on_board(x, y, size):
    return 0 <= x < size and 0 <= y < size


def _legal_destinations(board, origin):
    pieces, heights = board
    size = pieces.shape[0]
    x, y = origin
    origin_height = int(heights[x, y])
    destinations = []
    for dx, dy in DIRECTIONS:
        target_x, target_y = x + dx, y + dy
        if not _on_board(target_x, target_y, size):
            continue
        if pieces[target_x, target_y] != 0 or heights[target_x, target_y] >= 4:
            continue
        if int(heights[target_x, target_y]) <= origin_height + 1:
            destinations.append((target_x, target_y))
    return destinations


def _player_features(board, sign):
    pieces, heights = board
    threats = np.zeros_like(heights, dtype=np.float32)
    mobility = np.zeros_like(heights, dtype=np.float32)
    for origin in map(tuple, np.argwhere(pieces * sign > 0)):
        destinations = _legal_destinations(board, origin)
        mobility[origin] = len(destinations) / 8.0
        if int(heights[origin]) == 2:
            for destination in destinations:
                if int(heights[destination]) == 3:
                    threats[destination] = 1.0
    return threats, mobility


def _climb_access(board):
    pieces, heights = board
    size = pieces.shape[0]
    access = np.zeros_like(heights, dtype=np.float32)
    for x in range(size):
        for y in range(size):
            count = 0
            origin_height = int(heights[x, y])
            # The hypothetical origin may currently be occupied (we are asking
            # about the square's geometry), but a worker can never stand on a dome.
            if origin_height >= 4:
                continue
            for dx, dy in DIRECTIONS:
                target_x, target_y = x + dx, y + dy
                if not _on_board(target_x, target_y, size):
                    continue
                if pieces[target_x, target_y] != 0 or heights[target_x, target_y] >= 4:
                    continue
                if int(heights[target_x, target_y]) <= origin_height + 1:
                    count += 1
            access[x, y] = count / 8.0
    return access


def encode_v4_board(board):
    """Encode one current-player canonical board into the declared 13 planes.

    Derived tactical planes are zero until all four workers are placed. During
    standard play, the phase scalar is ``min(sum(heights), 40) / 40``; 40 is an
    explicit saturation point, not a hidden learned convention.
    """
    board = np.asarray(board)
    if board.shape != (2, 5, 5):
        raise ValueError("V4 encoder expects a board with shape (2, 5, 5).")
    pieces, heights = board
    if np.any(heights < 0) or np.any(heights > 4):
        raise ValueError("Santorini heights must be in [0, 4].")
    encoded = np.zeros((INPUT_CHANNELS, 5, 5), dtype=np.float32)
    encoded[0] = pieces > 0
    encoded[1] = pieces < 0
    encoded[2] = heights == 1
    encoded[3] = heights == 2
    encoded[4] = heights == 3
    encoded[5] = heights >= 4
    encoded[6] = heights.astype(np.float32) / 4.0

    workers_placed = int(np.count_nonzero(pieces))
    if workers_placed == 4:
        encoded[7], encoded[9] = _player_features(board, 1)
        encoded[8], encoded[10] = _player_features(board, -1)
        encoded[11] = _climb_access(board)
        phase = min(float(np.sum(heights)), PHASE_BUILD_CLIP) / PHASE_BUILD_CLIP
    else:
        phase = workers_placed / 4.0
    encoded[12].fill(phase)
    return encoded


def encode_v4_boards(boards):
    return np.asarray([encode_v4_board(board) for board in boards], dtype=np.float32)
