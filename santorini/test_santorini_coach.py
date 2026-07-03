import os
import pickle
import tempfile
import unittest

import numpy as np

from Coach import Coach
from utils import dotdict


class TinyGame:
    def getInitBoard(self):
        return np.array([0], dtype=np.int64)

    def getActionSize(self):
        return 2

    def getNextState(self, board, player, action):
        return np.array([board[0] + 1], dtype=np.int64), -player

    def getValidMoves(self, board, player):
        return np.array([1, 1], dtype=np.int64)

    def getGameEnded(self, board, player):
        if board[0] >= 2:
            return -1
        return 0

    def getCanonicalForm(self, board, player):
        return board

    def getSymmetries(self, board, pi):
        return [(board, pi)]

    def stringRepresentation(self, board):
        return board.tobytes()


class BatchCountingNNet:
    def __init__(self, game):
        self.game = game
        self.batch_sizes = []

    def predict(self, board):
        return np.array([0.5, 0.5], dtype=np.float32), 0.0

    def predict_batch(self, boards):
        self.batch_sizes.append(len(boards))
        policies = np.tile(np.array([0.5, 0.5], dtype=np.float32), (len(boards), 1))
        values = np.zeros(len(boards), dtype=np.float32)
        return policies, values


class FixedOpeningSampler:
    def __init__(self, board):
        self.board = board
        self.self_play_calls = 0

    def sample_self_play_board(self):
        self.self_play_calls += 1
        return self.board.copy()

    def sample_arena_suite(self, count):
        return [self.board.copy() for _ in range(count)]


class TestSantoriniCoachExamples(unittest.TestCase):
    def make_coach_shell(self, load_folder, load_file='best.pth.tar'):
        coach = object.__new__(Coach)
        coach.args = dotdict({
            'load_folder_file': (load_folder, load_file),
        })
        return coach

    def test_examples_candidates_prefer_explicit_then_resume_files(self):
        with tempfile.TemporaryDirectory() as folder:
            open(os.path.join(folder, 'checkpoint_1.pth.tar.examples'), 'wb').close()
            open(os.path.join(folder, 'checkpoint_2.pth.tar.examples'), 'wb').close()
            coach = self.make_coach_shell(folder)

            candidates = coach._examplesCandidates('manual.examples')

            self.assertEqual(candidates[0], os.path.join(folder, 'manual.examples'))
            self.assertEqual(candidates[1], 'manual.examples')
            self.assertEqual(candidates[2], os.path.join(folder, 'best.pth.tar.examples'))
            self.assertEqual(candidates[3], os.path.join(folder, 'latest.examples'))
            self.assertEqual(len(candidates), len({os.path.abspath(path) for path in candidates}))

    def test_load_train_examples_falls_back_to_latest(self):
        with tempfile.TemporaryDirectory() as folder:
            examples = [('board', 'policy', 1)]
            with open(os.path.join(folder, 'latest.examples'), 'wb') as examples_file:
                pickle.dump(examples, examples_file)

            coach = self.make_coach_shell(folder)
            coach.loadTrainExamples(skipFirstSelfPlay=False)

            self.assertEqual(coach.trainExamplesHistory, examples)
            self.assertFalse(coach.skipFirstSelfPlay)

    def test_prune_checkpoint_examples_keeps_resume_anchors(self):
        with tempfile.TemporaryDirectory() as folder:
            coach = object.__new__(Coach)
            coach.args = dotdict({
                'checkpoint': folder,
                'checkpointExamplesToKeep': 2,
            })

            for filename in [
                'checkpoint_0.pth.tar.examples',
                'checkpoint_1.pth.tar.examples',
                'checkpoint_2.pth.tar.examples',
                'latest.examples',
                'best.pth.tar.examples',
            ]:
                with open(os.path.join(folder, filename), 'wb') as examples_file:
                    examples_file.write(b'examples')

            coach._pruneCheckpointExampleFiles()

            self.assertFalse(os.path.exists(os.path.join(folder, 'checkpoint_0.pth.tar.examples')))
            self.assertTrue(os.path.exists(os.path.join(folder, 'checkpoint_1.pth.tar.examples')))
            self.assertTrue(os.path.exists(os.path.join(folder, 'checkpoint_2.pth.tar.examples')))
            self.assertTrue(os.path.exists(os.path.join(folder, 'latest.examples')))
            self.assertTrue(os.path.exists(os.path.join(folder, 'best.pth.tar.examples')))

    def test_prune_checkpoint_examples_reserves_slot_for_pending_save(self):
        with tempfile.TemporaryDirectory() as folder:
            coach = object.__new__(Coach)
            coach.args = dotdict({
                'checkpoint': folder,
                'checkpointExamplesToKeep': 1,
            })

            with open(os.path.join(folder, 'checkpoint_12.pth.tar.examples'), 'wb') as examples_file:
                examples_file.write(b'examples')

            coach._pruneCheckpointExampleFiles(
                pending_filename='checkpoint_13.pth.tar.examples',
            )

            self.assertFalse(os.path.exists(os.path.join(folder, 'checkpoint_12.pth.tar.examples')))

    def test_delete_loaded_examples_preserves_current_resume_files(self):
        with tempfile.TemporaryDirectory() as folder:
            coach = object.__new__(Coach)
            coach.args = dotdict({'checkpoint': folder})

            merged_examples = os.path.join(folder, 'merged_20.examples')
            latest_examples = os.path.join(folder, 'latest.examples')
            with open(merged_examples, 'wb') as examples_file:
                examples_file.write(b'examples')
            with open(latest_examples, 'wb') as examples_file:
                examples_file.write(b'examples')

            coach.loadedTrainExamplesFile = merged_examples
            coach._deleteLoadedTrainExamplesFile()
            self.assertFalse(os.path.exists(merged_examples))

            coach.loadedTrainExamplesFile = latest_examples
            coach._deleteLoadedTrainExamplesFile()
            self.assertTrue(os.path.exists(latest_examples))


class TestSantoriniCoachBatchedSelfPlay(unittest.TestCase):
    def test_batched_self_play_uses_batched_prediction(self):
        np.random.seed(13)
        game = TinyGame()
        nnet = BatchCountingNNet(game)
        args = dotdict({
            'numMCTSSims': 2,
            'cpuct': 1.0,
            'tempThreshold': 10,
            'selfPlayBatchSize': 2,
            'quiet': True,
        })
        coach = Coach(game, nnet, args)

        examples = coach.executeEpisodesBatched(2)

        self.assertEqual(len(examples), 4)
        self.assertGreaterEqual(max(nnet.batch_sizes), 2)
        for board, pi, value in examples:
            self.assertEqual(board.shape, (1,))
            self.assertAlmostEqual(float(sum(pi)), 1.0, places=5)
            self.assertIn(value, [-1, 1])

    def test_batched_self_play_uses_opening_sampler(self):
        np.random.seed(13)
        game = TinyGame()
        nnet = BatchCountingNNet(game)
        args = dotdict({
            'numMCTSSims': 2,
            'cpuct': 1.0,
            'tempThreshold': 10,
            'selfPlayBatchSize': 2,
            'quiet': True,
        })
        sampler = FixedOpeningSampler(np.array([1], dtype=np.int64))
        coach = Coach(game, nnet, args, opening_sampler=sampler)

        examples = coach.executeEpisodesBatched(2)

        self.assertEqual(sampler.self_play_calls, 2)
        self.assertEqual(len(examples), 2)


if __name__ == "__main__":
    unittest.main()
