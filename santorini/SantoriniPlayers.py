import numpy as np

SANTORINI_DIRECTIONS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]


def coordinate_label(location):
    row, col = location
    return "{}{}".format(chr(ord('A') + col), row + 1)


def parse_coordinate(text, board_size):
    text = text.strip().upper()
    if len(text) < 2:
        raise ValueError("Coordinates should look like A1, B3, or E5.")

    col = ord(text[0]) - ord('A')
    try:
        row = int(text[1:]) - 1
    except ValueError:
        raise ValueError("Coordinates should use a letter column and numeric row, like B3.")

    if not (0 <= row < board_size and 0 <= col < board_size):
        max_coordinate = "{}{}".format(chr(ord('A') + board_size - 1), board_size)
        raise ValueError("Coordinates must be between A1 and {}.".format(max_coordinate))
    return row, col


# Renamed OthelloPlayers.py Function
class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanSantoriniPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        if hasattr(self.game, 'isPlacementPhase') and self.game.isPlacementPhase(board):
            while True:
                entered = input("\nPlace your next worker on a square (example: B3), or q to quit: ").strip()
                if entered.lower() in ('q', 'quit', 'exit'):
                    raise KeyboardInterrupt
                try:
                    location = parse_coordinate(entered, self.game.n)
                    action = self.game.getPlacementAction(location)
                except ValueError as error:
                    print("Sorry, {}".format(error))
                    continue
                if valids[action]:
                    return action
                print("Sorry, that square is already occupied.")
        worker_locations = self.game.getCharacterLocations(board, 1)
        print(
            "Your workers: O at {}, U at {}.".format(
                coordinate_label(worker_locations[0]),
                coordinate_label(worker_locations[1]),
            )
        )
        while True:
            entered = input("\nEnter move as '<worker> <move-to> <build-at>' (example: O B3 C2), or q to quit: ").strip()
            if entered.lower() in ('q', 'quit', 'exit'):
                raise KeyboardInterrupt
            if entered.lower() in ('h', 'help', '?'):
                print("Example: O B3 C2 means move worker O to B3, then build on C2.")
                continue
            try:
                action = self._parse_action(board, entered)
            except ValueError as error:
                print("Sorry, {}".format(error))
                continue
            if valids[action]:
                return action
            print("Sorry, that move/build is not legal in the current position.")

    def _parse_action(self, board, entered):
        parts = entered.replace(',', ' ').split()
        if len(parts) != 3:
            raise ValueError("use exactly three parts, like O B3 C2")

        worker_name = parts[0].upper()
        if worker_name not in ('O', 'U'):
            raise ValueError("worker must be O or U")

        worker_idx = 0 if worker_name == 'O' else 1
        worker_location = self.game.getCharacterLocations(board, 1)[worker_idx]
        move_location = parse_coordinate(parts[1], self.game.n)
        build_location = parse_coordinate(parts[2], self.game.n)

        move_delta = (
            move_location[0] - worker_location[0],
            move_location[1] - worker_location[1],
        )
        build_delta = (
            build_location[0] - move_location[0],
            build_location[1] - move_location[1],
        )

        if move_delta not in SANTORINI_DIRECTIONS:
            raise ValueError("move-to square must be adjacent to worker {}".format(worker_name))
        if build_delta not in SANTORINI_DIRECTIONS:
            raise ValueError("build square must be adjacent to the move-to square")

        move_direction = SANTORINI_DIRECTIONS.index(move_delta)
        build_direction = SANTORINI_DIRECTIONS.index(build_delta)
        return self.game.getActionFromOrigin(worker_location, move_direction, build_direction)

# Renamed OthelloPlayers.py Function
class GreedySantoriniPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        if hasattr(self.game, 'isPlacementPhase') and self.game.isPlacementPhase(board):
            # Baseline compatibility: prefer central placement squares.
            legal = np.flatnonzero(valids)
            center = (self.game.n - 1) / 2.0
            return min(
                legal,
                key=lambda action: sum(
                    (coordinate - center) ** 2
                    for coordinate in (
                        int(action) // self.game.local_action_size // self.game.n,
                        int(action) // self.game.local_action_size % self.game.n,
                    )
                ),
            )
        candidates = []
        for a in range(self.game.getActionSize()):
            if valids[a]==0:
                continue
            nextBoard, _ = self.game.getNextState(board, 1, a)
            score = self.game.getScore(nextBoard, 1)
            candidates += [(-score, a)]
        candidates.sort()
       
        return candidates[0][1]

            
