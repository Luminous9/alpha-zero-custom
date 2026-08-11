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
    """Vectorized equivalent of :func:`encode_v4_board` for inference batches."""
    boards = np.asarray(boards)
    if boards.ndim != 4 or boards.shape[1:] != (2, 5, 5):
        raise ValueError("V4 encoder expects boards with shape (N, 2, 5, 5).")
    if not len(boards):
        return np.empty((0, INPUT_CHANNELS, 5, 5), dtype=np.float32)
    if len(boards) == 1:
        return encode_v4_board(boards[0])[None]
    pieces = boards[:, 0]
    heights = boards[:, 1]
    if np.any(heights < 0) or np.any(heights > 4):
        raise ValueError("Santorini heights must be in [0, 4].")

    encoded = np.zeros((len(boards), INPUT_CHANNELS, 5, 5), dtype=np.float32)
    encoded[:, 0] = pieces > 0
    encoded[:, 1] = pieces < 0
    encoded[:, 2] = heights == 1
    encoded[:, 3] = heights == 2
    encoded[:, 4] = heights == 3
    encoded[:, 5] = heights >= 4
    encoded[:, 6] = heights.astype(np.float32) / 4.0

    workers_placed = np.count_nonzero(pieces, axis=(1, 2))
    standard = workers_placed == 4

    def shifted_slices(delta):
        if delta >= 0:
            return slice(0, 5 - delta), slice(delta, 5)
        return slice(-delta, 5), slice(0, 5 + delta)

    for dx, dy in DIRECTIONS:
        origin_x, target_x = shifted_slices(dx)
        origin_y, target_y = shifted_slices(dy)
        origin = (slice(None), origin_x, origin_y)
        target = (slice(None), target_x, target_y)
        origin_heights = heights[origin]
        target_heights = heights[target]
        destination_open = (
            (pieces[target] == 0)
            & (target_heights < 4)
            & (target_heights <= origin_heights + 1)
        )

        climb_legal = destination_open & (origin_heights < 4)
        encoded[:, 11][origin] += climb_legal.astype(np.float32) / 8.0

        for sign, threat_plane, mobility_plane in (
            (1, 7, 9), (-1, 8, 10)
        ):
            legal = destination_open & (pieces[origin] * sign > 0)
            encoded[:, mobility_plane][origin] += legal.astype(np.float32) / 8.0
            threat = legal & (origin_heights == 2) & (target_heights == 3)
            encoded[:, threat_plane][target] = np.maximum(
                encoded[:, threat_plane][target], threat.astype(np.float32)
            )

    encoded[:, 7:12] *= standard[:, None, None, None]
    phase = np.where(
        standard,
        np.minimum(np.sum(heights, axis=(1, 2)), PHASE_BUILD_CLIP)
        / PHASE_BUILD_CLIP,
        workers_placed / 4.0,
    ).astype(np.float32)
    encoded[:, 12] = phase[:, None, None]
    return encoded
