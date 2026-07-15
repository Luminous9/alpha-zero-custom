from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .SantoriniLogic import Board
import numpy as np

class SantoriniGame(Game):
    """
    Many of thes functions are based on those from OthelloGame.py:
        https://github.com/suragnair/alpha-zero-general/blob/master/othello/OthelloGame.py
    """
    supports_draws = False

    square_content = {
        -2: 'Y',
        -1: 'X',
        +0: '-',
        +1: 'O',
        +2: 'U'
    }

    # NOTE THESE ARE NEITHER CCW NOR CW!
    __directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    @staticmethod
    def getSquarePiece(piece):
        return SantoriniGame.square_content[piece]

    STANDARD_LOCAL_ACTIONS = 64
    PLACEMENT_LOCAL_ACTION = 64

    def __init__(self, board_length=5, true_random_placement=False, sequential_placement=False):
        self.n = board_length
        self.true_random_placement = true_random_placement
        self.sequential_placement = bool(sequential_placement)
        self.local_action_size = 65 if self.sequential_placement else 64
        self._directions_array = np.array(self.__directions, dtype=np.int8)
        self._action_origin_x = np.zeros(self.getActionSize(), dtype=np.int8)
        self._action_origin_y = np.zeros(self.getActionSize(), dtype=np.int8)
        self._action_move_direction = np.zeros(self.getActionSize(), dtype=np.int8)
        self._action_build_direction = np.zeros(self.getActionSize(), dtype=np.int8)
        self._action_move_dx = np.zeros(self.getActionSize(), dtype=np.int8)
        self._action_move_dy = np.zeros(self.getActionSize(), dtype=np.int8)
        self._action_build_dx = np.zeros(self.getActionSize(), dtype=np.int8)
        self._action_build_dy = np.zeros(self.getActionSize(), dtype=np.int8)
        self._local_action_move_on_board = np.zeros((self.n, self.n, 64), dtype=bool)
        self._local_action_move_x = np.zeros((self.n, self.n, 64), dtype=np.int8)
        self._local_action_move_y = np.zeros((self.n, self.n, 64), dtype=np.int8)
        self._local_action_build_on_board = np.zeros((self.n, self.n, 64), dtype=bool)
        self._local_action_build_x = np.zeros((self.n, self.n, 64), dtype=np.int8)
        self._local_action_build_y = np.zeros((self.n, self.n, 64), dtype=np.int8)
        self._local_action_build_on_origin = np.zeros((self.n, self.n, 64), dtype=bool)
        self._canonical_multiplier = {
            1: np.array(
                [np.ones((self.n, self.n), dtype=int), np.ones((self.n, self.n), dtype=int)]
            ),
            -1: np.array(
                [-np.ones((self.n, self.n), dtype=int), np.ones((self.n, self.n), dtype=int)]
            ),
        }
        self._init_action_tables()
        self._init_local_action_geometry_tables()
        self._policy_symmetry_permutations = self._init_policy_symmetry_permutations()
        
    def _init_action_tables(self):
        for action in range(self.getActionSize()):
            origin = action // self.local_action_size
            origin_x = origin // self.n
            origin_y = origin % self.n
            local_action = action % self.local_action_size
            if local_action == self.PLACEMENT_LOCAL_ACTION:
                self._action_origin_x[action] = origin_x
                self._action_origin_y[action] = origin_y
                continue
            move_direction = local_action // 8
            build_direction = local_action % 8
            move_dx, move_dy = self.__directions[move_direction]
            build_dx, build_dy = self.__directions[build_direction]

            self._action_origin_x[action] = origin_x
            self._action_origin_y[action] = origin_y
            self._action_move_direction[action] = move_direction
            self._action_build_direction[action] = build_direction
            self._action_move_dx[action] = move_dx
            self._action_move_dy[action] = move_dy
            self._action_build_dx[action] = build_dx
            self._action_build_dy[action] = build_dy

    def _init_local_action_geometry_tables(self):
        for x in range(self.n):
            for y in range(self.n):
                for local_action in range(64):
                    move_direction = local_action // 8
                    build_direction = local_action % 8
                    move_dx, move_dy = self.__directions[move_direction]
                    build_dx, build_dy = self.__directions[build_direction]
                    move_x = x + move_dx
                    move_y = y + move_dy

                    if not self._is_on_board(move_x, move_y):
                        continue

                    build_x = move_x + build_dx
                    build_y = move_y + build_dy
                    self._local_action_move_on_board[x, y, local_action] = True
                    self._local_action_move_x[x, y, local_action] = move_x
                    self._local_action_move_y[x, y, local_action] = move_y

                    if self._is_on_board(build_x, build_y):
                        self._local_action_build_on_board[x, y, local_action] = True
                        self._local_action_build_x[x, y, local_action] = build_x
                        self._local_action_build_y[x, y, local_action] = build_y
                        self._local_action_build_on_origin[x, y, local_action] = (
                            build_x == x and build_y == y
                        )

    def _init_policy_symmetry_permutations(self):
        permutations = {}
        for rotations in range(4):
            for flip in (False, True):
                direction_map = self._direction_transform_indices(rotations, flip)
                old_indices = np.arange(self.getActionSize())
                new_indices = np.zeros(self.getActionSize(), dtype=np.int32)

                for x in range(self.n):
                    for y in range(self.n):
                        new_x, new_y = self._transform_square(x, y, rotations, flip)
                        origin_offset = (x * self.n + y) * self.local_action_size
                        new_origin_offset = (new_x * self.n + new_y) * self.local_action_size

                        for move_direction in range(8):
                            for build_direction in range(8):
                                old_action = origin_offset + move_direction * 8 + build_direction
                                new_action = (
                                    new_origin_offset
                                    + direction_map[move_direction] * 8
                                    + direction_map[build_direction]
                                )
                                new_indices[old_action] = new_action

                        if self.sequential_placement:
                            new_indices[origin_offset + self.PLACEMENT_LOCAL_ACTION] = (
                                new_origin_offset + self.PLACEMENT_LOCAL_ACTION
                            )

                permutations[(rotations, flip)] = (old_indices, new_indices)

        return permutations

    def getInitBoard(self):
        # return initial board (numpy board)
        if self.sequential_placement:
            return np.zeros((2, self.n, self.n), dtype=int)
        b = Board(self.n, true_random_placement=self.true_random_placement)
        return np.array(b.pieces)

    def getBoardSize(self):
        # (dimension,a,b) tuple
        return (2, self.n, self.n)


    def getActionSize(self):
        # return number of actions
        return self.n * self.n * self.local_action_size

    def getActionFromOrigin(self, origin, move_direction, build_direction):
        x, y = origin
        return (x * self.n + y) * self.local_action_size + move_direction * 8 + build_direction

    def getPlacementAction(self, location):
        if not self.sequential_placement:
            raise ValueError("Placement actions require sequential_placement=True.")
        x, y = location
        return (x * self.n + y) * self.local_action_size + self.PLACEMENT_LOCAL_ACTION

    def isPlacementAction(self, action):
        return self.sequential_placement and int(action) % self.local_action_size == self.PLACEMENT_LOCAL_ACTION

    def isPlacementPhase(self, board):
        return self.sequential_placement and int(np.count_nonzero(board[0])) < 4

    def decodeAction(self, action):
        action = int(action)
        if self.isPlacementAction(action):
            raise ValueError("Placement actions do not have move/build directions.")
        origin = (
            int(self._action_origin_x[action]),
            int(self._action_origin_y[action]),
        )
        local_action = action % self.local_action_size
        return origin, local_action // 8, local_action % 8

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move

        action = int(action)
        if self.isPlacementPhase(board):
            if not self.isPlacementAction(action):
                raise ValueError("Only placement actions are valid during worker placement.")
            location = (
                int(self._action_origin_x[action]),
                int(self._action_origin_y[action]),
            )
            if board[0][location] != 0:
                raise ValueError("Cannot place a worker on an occupied square.")

            next_board = board.copy()
            worker_count = len(self.getCharacterLocations(next_board, player))
            if worker_count >= 2:
                raise ValueError("The current player has already placed both workers.")
            next_board[0][location] = player * (worker_count + 1)
            if worker_count == 1:
                self._normalize_worker_labels(next_board, player)
                return next_board, -player
            return next_board, player

        if self.isPlacementAction(action):
            raise ValueError("Placement actions are invalid after setup is complete.")
        current = (
            int(self._action_origin_x[action]),
            int(self._action_origin_y[action]),
        )
        move_dx = int(self._action_move_dx[action])
        move_dy = int(self._action_move_dy[action])
        build_dx = int(self._action_build_dx[action])
        build_dy = int(self._action_build_dy[action])

        move = (current[0] + move_dx, current[1] + move_dy)
        build = (move[0] + build_dx, move[1] + build_dy)

        next_board = board.copy()
        piece = next_board[0][current]
        next_board[0][current] = 0
        next_board[0][move] = piece

        if next_board[1][move] != 3:
            next_board[1][build] += 1

        return (next_board, -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        return self._get_valid_moves_fast(board, player)

    def getGameEndedAndValidMoves(self, board, player):
        if self.isPlacementPhase(board):
            return 0, self._get_valid_placement_moves(board)

        player_pieces = self.getCharacterLocations(board, player)
        opponent_pieces = self.getCharacterLocations(board, -1 * player)

        for piece in player_pieces:
            if board[1][piece] == 3:
                return 1, None

        for piece in opponent_pieces:
            if board[1][piece] == 3:
                return -1, None

        valids = self._get_valid_moves_fast(board, player, player_pieces=player_pieces)
        if not np.any(valids):
            return -1, valids
        return 0, valids

    def _get_valid_moves_fast(self, board, player, player_pieces=None):
        if self.isPlacementPhase(board):
            return self._get_valid_placement_moves(board)

        pieces = board[0]
        heights = board[1]
        valids = np.zeros(self.getActionSize(), dtype=np.int8)
        player_pieces = player_pieces or self.getCharacterLocations(board, player)

        for worker in player_pieces:
            x, y = worker
            origin_height = heights[x, y]
            action_offset = (x * self.n + y) * self.local_action_size

            move_x = self._local_action_move_x[x, y]
            move_y = self._local_action_move_y[x, y]
            build_x = self._local_action_build_x[x, y]
            build_y = self._local_action_build_y[x, y]

            move_heights = heights[move_x, move_y]
            build_heights = heights[build_x, build_y]
            moved_to_level_three = move_heights == 3

            move_legal = (
                self._local_action_move_on_board[x, y]
                & (pieces[move_x, move_y] == 0)
                & (move_heights <= 3)
                & ((move_heights - origin_height) <= 1)
            )
            build_legal = (
                self._local_action_build_on_board[x, y]
                & (
                    moved_to_level_three
                    | (
                        (
                            self._local_action_build_on_origin[x, y]
                            | (pieces[build_x, build_y] == 0)
                        )
                        & (build_heights <= 3)
                    )
                )
            )

            valids[action_offset:action_offset + 64] = move_legal & build_legal

        return valids

    def _get_valid_placement_moves(self, board):
        valids = np.zeros(self.getActionSize(), dtype=np.int8)
        for x, y in np.argwhere(board[0] == 0):
            valids[self.getPlacementAction((int(x), int(y)))] = 1
        return valids

    @staticmethod
    def _normalize_worker_labels(board, player):
        locations = sorted(map(tuple, np.argwhere(board[0] * player > 0)))
        for label, location in enumerate(locations, start=1):
            board[0][location] = player * label

    def _is_on_board(self, x, y):
        return 0 <= x < self.n and 0 <= y < self.n

    def getValidMovesHuman(self, board, player):
        b = Board(self.n)
        b.pieces = np.copy(board)
        color = player

        return b.get_all_moves(color)


            
    def getCharacterLocations(self, board, player):  
        """
        Returns a list of both character's locations as tuples for the player
        """
        locations = []
        for label in (1, 2):
            rows, cols = np.where(board[0] == label * player)
            if len(rows):
                locations.append((int(rows[0]), int(cols[0])))
        return locations

    def getGameEnded(self, board, player):
        """
        Assumes player is about to move. THIS IS NOT COMPATIBLE with the prior implementation of Arena.py
        which returned self.game.getGameEnded(board, 1). 
        Input:
            board: current board
            player: current player (1 or -1)
        Returns:
            r: 0 if game has not ended. 1 if THIS player has won, -1 if player THIS lost,
               small non-zero value for draw.
        """
        ended, _ = self.getGameEndedAndValidMoves(board, player)
        return ended





       
    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        board = board * self._canonical_multiplier[player]
        
        return board


    def getRandomBoardSymmetry(self, board):
        """
        Returns a random board symmetry.
        """
        b = Board(self.n)
        b.pieces = np.copy(board)
        i = np.random.randint(0, 4)
        k = np.random.choice([True, False])
        newB0 = np.rot90(b.pieces[0], i)
        newB1 = np.rot90(b.pieces[1], i)
        if k:
            newB0 = np.fliplr(newB0)
            newB1 = np.fliplr(newB1)
        
        return np.array([newB0, newB1])

    def getSymmetries(self, board, pi):
        # mirror, rotational

        assert(len(pi) == self.getActionSize())

        syms = []
        pi = np.asarray(pi)

        for rotations in range(4):
            for flip in [False, True]:
                newB0 = np.rot90(board[0], rotations)
                newB1 = np.rot90(board[1], rotations)
                if flip:
                    newB0 = np.fliplr(newB0)
                    newB1 = np.fliplr(newB1)

                syms.append((
                    np.array([newB0, newB1]),
                    self._transform_policy_array(pi, rotations, flip),
                ))

        return syms

    def getPolicySymmetryPermutation(self, rotations, flip):
        """Return old/new policy indices for one dihedral board transform."""
        return self._policy_symmetry_permutations[(int(rotations) % 4, bool(flip))]

    def _transform_policy(self, pi, rotations, flip):
        return self._transform_policy_array(np.asarray(pi), rotations, flip)

    def _transform_policy_array(self, pi, rotations, flip):
        transformed = np.zeros(self.getActionSize(), dtype=pi.dtype)
        old_indices, new_indices = self._policy_symmetry_permutations[(rotations, flip)]
        transformed[new_indices] = pi[old_indices]
        return transformed

    def _transform_square(self, x, y, rotations, flip):
        for _ in range(rotations):
            x, y = self.n - 1 - y, x
        if flip:
            y = self.n - 1 - y
        return x, y

    def _direction_transform_indices(self, rotations, flip):
        direction_to_index = {direction: i for i, direction in enumerate(self.__directions)}
        transformed_indices = []

        for dx, dy in self.__directions:
            for _ in range(rotations):
                dx, dy = -dy, dx
            if flip:
                dy = -dy
            transformed_indices.append(direction_to_index[(dx, dy)])

        return transformed_indices
                
    def rotate(self, pi_64):
        """
        Input: first XOR second half of Pi
        Returns: the half of pie in a reordered list that corresponds
                 to a counterclockwise rotation of the board
        """
        assert (len(pi_64) == 64)
  
        rotation_indices = [18, 20, 23, 17, 22, 16, 19, 21, 34, 36, 39, 33, 
                            38, 32, 35, 37, 58, 60, 63, 57, 62, 56, 59, 61, 
                            10, 12, 15,  9, 14,  8, 11, 13, 50, 52, 55, 49, 
                            54, 48, 51, 53,  2,  4,  7,  1,  6,  0,  3,  5, 
                            26, 28, 31, 25, 30, 24, 27, 29, 42, 44, 47, 41, 
                            46, 40, 43, 45]
    
        pi_new = [pi_64[i] for i in rotation_indices]
  
        return pi_new
    
    
    def flip(self, pi_64):
        """
        Input: first XOR second half of Pi
        Returns: the half of pie in a reordered list that corresponds
                 to a left<--->right flip of the board
        """
        assert (len(pi_64) == 64)
    
        flip_indices = [18, 17, 16, 20, 19, 23, 22, 21, 10, 9, 8, 12, 11, 
                         15, 14, 13, 2, 1, 0, 4, 3, 7, 6, 5, 34, 33, 32, 36, 
                         35, 39, 38, 37, 26, 25, 24, 28, 27, 31, 30, 29, 58, 
                         57, 56, 60, 59, 63, 62, 61, 50, 49, 48, 52, 51, 55,
                         54, 53, 42, 41, 40, 44, 43, 47, 46, 45]   
    
        pi_new = [pi_64[i] for i in flip_indices]
  
        return pi_new
        
#        # One counter clockwise rotation:
#        l = []
#        for i in range(8):
#            l2 = []
#            for k in range(8):
#                l2.append(pi[i*8 + k])
#            l3 = []
#            l3 = [l2[i] for i in [2, 4, 7, 1, 6, 0, 3, 5]]
#            l.append(l3)
#        l_pi = [l[i] for i in [2, 4, 7, 1, 6, 0, 3, 5]]
#        
#    
#        # One flip (mirror) left <-->> right 
#        l_flip
#        for i in range(8):
#            l2_flip = []
#            for k in range(8):
#                l2_flip.append(pi[i*8 + k])
#            l3_flip = []
#            l3_flip = [l2_flip[i] for i in [2, 1, 0, 4, 3, 7, 6, 5]]
#            l_flip.append(l3_flip)
#        l_pi_flip = [l_flip[i] for i in [2, 1, 0, 4, 3, 7, 6, 5]]
#    
        """
        split into 
        0-63, and 64-127. Here the letters a,...,h denote move locations
        
        These are the actions the first 64 values of pi correspond to doing
        
        0  1  2    8  9  10    16 17 18 
        3  a  4    11 b  12    19 c  20
        5  6  7    13 14 15    21 22 23
        
        24 25 26   original    32 33 34
        27 d  28    piece      35 e  36
        29 30 31   location    37 38 39
        
        40 41 42   48 49 50    56 57 58
        43 f  44   51 g  52    59 h  60
        45 46 47   53 54 55    61 62 63
        
        
        
        Initially we have: 
            
            a  b  c
            d     e
            f  g  h
        
        after CCW rotation we have: 
            
            c  e  h
            b     g
            a  d  f
        
        where for each move location a,..,h:
        
                                0  1  2    
        initially a is:         3  a  4    
                                5  6  7 

                                2  4  7
        after CCW rotation:     1  a  6
                                0  3  5
                                
        
                            
        initial values at indices: [0, 1, 2, 3, 4, 5, 6, 7]
                               --->[2, 4, 7, 1, 6, 0, 3, 5] under 1 CCW rotation
        
        
        
        For flips left <---> right:
               
        [0, 1, 2, 3, 4, 5, 6, 7]
    --->[2, 1, 0, 4, 3, 7, 6, 5] under 1 flip
        
        
        """


    def stringRepresentation(self, board):
        return board.tobytes()

    def stringRepresentationReadable(self, board):
        # Do not think this works.
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    def getScore(self, board, player):
        """
        Only used by 'Greedy player'
        """

        b = Board(self.n)
        b.pieces = np.copy(board)

        piece_locations = self.getCharacterLocations(board, player) 
        char0 = piece_locations[0]
        char1 = piece_locations[1]

        opponent_piece_locations = self.getCharacterLocations(board, -player) 
        opp_char0 = opponent_piece_locations[0]
        opp_char1 = opponent_piece_locations[1]
        
        player_score = max(b.pieces[1][char0], b.pieces[1][char1])
        opponent_score = max(b.pieces[1][opp_char0], b.pieces[1][opp_char1])
        if player_score == 3:
            # this is a winning move, set score very high
            player_score = 100
        if opponent_score == 3:
            # this is a winning move, set score very high
            opponent_score = 100
        score = player_score - opponent_score 
        # height of highest piece for player:
        return score

    @staticmethod
    def display(board):
        n = board.shape[1]
        columns = [chr(ord('A') + col) for col in range(n)]
        print("    " + " ".join(columns))
        print("  " + "-" * (2 * n + 3))
        for row in range(n):
            print(row + 1, "|", end="")
            for col in range(n):
                piece = board[0][row][col]
                print(SantoriniGame.square_content[piece], end=" ")
            print("|")

        print("  " + "-" * (2 * n + 3))
        print("    " + " ".join(columns))
        print("  " + "-" * (2 * n + 3))
        for row in range(n):
            print(row + 1, "|", end="")
            for col in range(n):
                height = board[1][row][col]
                print(height, end=" ")
            print("|")

        print("  " + "-" * (2 * n + 3))
