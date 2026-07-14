import unittest

import numpy as np

from MCTS import MCTS
from utils import dotdict


class SamePlayerTerminalGame:
    def getActionSize(self):
        return 1

    def getNextState(self, board, player, action):
        return np.array([1], dtype=np.int8), player

    def getValidMoves(self, board, player):
        return np.array([1], dtype=np.int8)

    def getGameEnded(self, board, player):
        return 1 if int(board[0]) == 1 else 0

    def getCanonicalForm(self, board, player):
        return board

    def stringRepresentation(self, board):
        return board.tobytes()


class ZeroNetwork:
    def predict(self, board):
        return np.array([1.0], dtype=np.float32), 0.0


class TestTurnAwareMCTS(unittest.TestCase):
    def test_same_player_edge_does_not_invert_value(self):
        game = SamePlayerTerminalGame()
        mcts = MCTS(game, ZeroNetwork(), dotdict({'numMCTSSims': 2, 'cpuct': 1.0}))
        root = np.array([0], dtype=np.int8)

        mcts.getActionProb(root, temp=1)

        root_key = game.stringRepresentation(root)
        self.assertEqual(mcts.Qsa[(root_key, 0)], 1.0)


if __name__ == '__main__':
    unittest.main()
