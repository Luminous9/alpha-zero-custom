"""Adapter for using JPricey's native Santorini engine as a search oracle."""

import atexit
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from .SantoriniPlayers import SANTORINI_DIRECTIONS, coordinate_label, parse_coordinate


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORACLE_BINARY = (
    REPOSITORY_ROOT / "tools" / "santorini_oracle" / "target" / "release" / "santorini-oracle"
)


def _validate_board(board):
    board = np.asarray(board)
    if board.shape != (2, 5, 5):
        raise ValueError("Santorini oracle boards must have shape (2, 5, 5).")
    if np.any(board[1] < 0) or np.any(board[1] > 4):
        raise ValueError("Santorini building heights must be between zero and four.")
    return board


def _worker_coordinates(pieces, sign):
    locations = [tuple(map(int, location)) for location in np.argwhere(pieces * sign > 0)]
    return sorted(coordinate_label(location) for location in locations)


def canonical_board_to_fen(board):
    """Encode a canonical Mortal board at a legal joint-action boundary.

    santorini-ai treats each player's two worker placements as one move.  It can
    therefore search the empty board and the state after player one's complete
    pair, but not V4's one- or three-worker intermediate states.
    """
    board = _validate_board(board)
    pieces, heights = board
    positive = _worker_coordinates(pieces, 1)
    negative = _worker_coordinates(pieces, -1)
    # The Rust engine serializes A5..E5 first; this project stores A1..E1 in row zero.
    height_text = "".join(
        str(int(heights[row, col]))
        for row in range(4, -1, -1)
        for col in range(5)
    )
    if len(positive) == 2 and len(negative) == 2:
        return "{}/1/mortal:{}/mortal:{}".format(
            height_text, ",".join(positive), ",".join(negative)
        )
    if np.any(heights):
        raise ValueError("Oracle placement boundaries cannot contain buildings.")
    if not positive and not negative:
        return "{}/1/mortal:/mortal:".format(height_text)
    if not positive and len(negative) == 2:
        # V4 has already canonicalized to player two, so player one's placed
        # pair is negative in the network-visible board.
        return "{}/2/mortal:{}/mortal:".format(
            height_text, ",".join(negative)
        )
    raise ValueError(
        "santorini-ai supports only empty, two-worker joint-boundary, or "
        "completed-placement boards; one/three-worker states are factored locally."
    )


def external_joint_placement_locations(actions, board_size=5):
    """Decode an unordered standard Mortal worker pair from an action path."""
    coordinates = [
        action.get("value")
        for action in actions
        if action.get("type") == "place_worker"
    ]
    if len(coordinates) != 2 or any(not isinstance(value, str) for value in coordinates):
        raise ValueError("Oracle placement action must contain exactly two worker squares.")
    locations = tuple(sorted(parse_coordinate(value, board_size) for value in coordinates))
    if locations[0] == locations[1]:
        raise ValueError("Oracle placement pair cannot reuse one square.")
    return locations


def _parse_worker_section(section):
    if ":" not in section:
        return []
    _, coordinates = section.split(":", 1)
    return [coordinate.strip() for coordinate in coordinates.split(",") if coordinate.strip()]


def _place_labeled_workers(pieces, coordinates, sign):
    locations = sorted(parse_coordinate(coordinate, 5) for coordinate in coordinates)
    for label, location in enumerate(locations, start=1):
        pieces[location] = sign * label


def fen_to_canonical_board(fen):
    """Decode an oracle FEN into this project's current-player canonical board."""
    sections = str(fen).split("/")
    if len(sections) != 4:
        raise ValueError("Oracle FEN must have four slash-separated sections.")
    height_digits = [int(char) for char in sections[0] if char.isdigit()]
    if len(height_digits) != 25:
        raise ValueError("Oracle FEN height section must contain exactly 25 digits.")

    external_heights = np.asarray(height_digits, dtype=int).reshape(5, 5)
    heights = np.flipud(external_heights).copy()
    pieces = np.zeros((5, 5), dtype=int)
    current_player = sections[1].strip()
    if current_player not in ("1", "2"):
        raise ValueError("Oracle FEN current-player marker must be 1 or 2.")

    p1_sign = 1 if current_player == "1" else -1
    p2_sign = -p1_sign
    _place_labeled_workers(pieces, _parse_worker_section(sections[2]), p1_sign)
    _place_labeled_workers(pieces, _parse_worker_section(sections[3]), p2_sign)
    return np.asarray([pieces, heights], dtype=int)


def anonymous_board_key(board):
    """Return a worker-label-independent key suitable for cross-engine comparisons."""
    board = _validate_board(board)
    normalized = np.asarray([np.sign(board[0]), board[1]], dtype=np.int8)
    return normalized.tobytes()


def _action_value(actions, action_type):
    for action in actions:
        if action.get("type") == action_type:
            return action.get("value")
    return None


def _move_destination(value):
    if isinstance(value, dict):
        return value.get("dest")
    return value


def external_actions_to_v3_actions(game, board, actions):
    """Map one Mortal action path to every equivalent legal V3 action index."""
    board = _validate_board(board)
    if game.isPlacementPhase(board):
        raise ValueError("Joint external placements are not supported by the first bridge.")

    origin_text = _action_value(actions, "select_worker")
    destination_text = _move_destination(_action_value(actions, "move_worker"))
    build_text = _action_value(actions, "build")
    if origin_text is None or destination_text is None:
        raise ValueError("Oracle action path is missing a selected worker or destination.")

    origin = parse_coordinate(origin_text, game.n)
    destination = parse_coordinate(destination_text, game.n)
    move_delta = (destination[0] - origin[0], destination[1] - origin[1])
    if move_delta not in SANTORINI_DIRECTIONS:
        raise ValueError("Oracle worker move is not adjacent in the V3 coordinate system.")
    move_direction = SANTORINI_DIRECTIONS.index(move_delta)
    valids = game.getValidMoves(board, 1)

    if build_text is not None:
        build = parse_coordinate(build_text, game.n)
        build_delta = (build[0] - destination[0], build[1] - destination[1])
        if build_delta not in SANTORINI_DIRECTIONS:
            raise ValueError("Oracle build is not adjacent to the destination square.")
        action = game.getActionFromOrigin(
            origin,
            move_direction,
            SANTORINI_DIRECTIONS.index(build_delta),
        )
        if not valids[action]:
            raise ValueError("Mapped oracle action is not legal in the V3 engine.")
        return [int(action)]

    # A winning Mortal move has no build. V3 has one equivalent action for each
    # on-board build direction and intentionally skips all of those builds.
    equivalents = []
    for build_direction in range(8):
        action = game.getActionFromOrigin(origin, move_direction, build_direction)
        if valids[action]:
            equivalents.append(int(action))
    if not equivalents:
        raise ValueError("Winning oracle move has no equivalent legal V3 action.")
    return equivalents


class SantoriniOracleProcess:
    """Persistent JSONL client for the fixed-node Rust oracle."""

    def __init__(self, binary_path=None):
        configured_path = binary_path or os.environ.get("SANTORINI_ORACLE_BINARY")
        self.binary_path = Path(configured_path) if configured_path else DEFAULT_ORACLE_BINARY
        if not self.binary_path.is_file():
            raise FileNotFoundError(
                "Santorini oracle binary not found at {}. Build it with: cargo build "
                "--release --manifest-path tools/santorini_oracle/Cargo.toml".format(
                    self.binary_path
                )
            )
        self.process = subprocess.Popen(
            [str(self.binary_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._next_id = 1
        atexit.register(self.close)
        self.info = self.request("ping")

    def request(self, command, **payload):
        if self.process.poll() is not None:
            raise RuntimeError("Santorini oracle process exited unexpectedly.")
        request_id = self._next_id
        self._next_id += 1
        request = {"command": command, "id": request_id}
        request.update(payload)
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        response_line = self.process.stdout.readline()
        if not response_line:
            raise RuntimeError("Santorini oracle closed stdout without a response.")
        response = json.loads(response_line)
        if response.get("id") != request_id:
            raise RuntimeError("Santorini oracle response id did not match its request.")
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "unknown Santorini oracle error"))
        return response

    def analyze_fen(self, fen, nodes=20_000):
        return self.request("analyze", fen=fen, nodes=int(nodes))

    def analyze(self, canonical_board, nodes=20_000):
        return self.analyze_fen(canonical_board_to_fen(canonical_board), nodes=nodes)

    def analyze_root_moves_fen(self, fen, nodes_per_move=20_000, top_k=8):
        return self.request(
            "analyze_root_moves",
            fen=fen,
            nodes=int(nodes_per_move),
            top_k=int(top_k),
        )

    def analyze_root_moves(self, canonical_board, nodes_per_move=20_000, top_k=8):
        return self.analyze_root_moves_fen(
            canonical_board_to_fen(canonical_board),
            nodes_per_move=nodes_per_move,
            top_k=top_k,
        )

    def legal_moves_fen(self, fen):
        return self.request("legal_moves", fen=fen)

    def legal_moves(self, canonical_board):
        return self.legal_moves_fen(canonical_board_to_fen(canonical_board))

    def reset(self):
        return self.request("reset")

    def close(self):
        process = getattr(self, "process", None)
        if process is None:
            return
        self.process = None
        if process.poll() is None:
            try:
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=2)
        if process.stdout is not None:
            process.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class SantoriniOraclePlayer:
    """Arena-compatible deterministic fixed-node oracle player."""

    def __init__(self, game, oracle, nodes=20_000):
        self.game = game
        self.oracle = oracle
        self.nodes = int(nodes)

    def startGame(self):
        self.oracle.reset()

    def play(self, canonical_board):
        response = self.oracle.analyze(canonical_board, nodes=self.nodes)
        equivalents = external_actions_to_v3_actions(
            self.game,
            canonical_board,
            response["best_move"]["actions"],
        )
        return min(equivalents)

    __call__ = play


def compare_legal_successors(game, canonical_board, oracle):
    """Compare unique legal successor states from both rule implementations."""
    canonical_board = _validate_board(canonical_board)
    valids = game.getValidMoves(canonical_board, 1)
    ours = set()
    for action in np.flatnonzero(valids):
        next_board, next_player = game.getNextState(canonical_board, 1, int(action))
        next_canonical = game.getCanonicalForm(next_board, next_player)
        ours.add(anonymous_board_key(next_canonical))

    response = oracle.legal_moves(canonical_board)
    theirs = {
        anonymous_board_key(fen_to_canonical_board(move["next_fen"]))
        for move in response["moves"]
        if not move.get("no_moves")
    }
    return {
        "matches": ours == theirs,
        "ours": ours,
        "theirs": theirs,
        "ours_count": len(ours),
        "theirs_count": len(theirs),
        "only_ours": ours - theirs,
        "only_theirs": theirs - ours,
    }
