import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from Arena import Arena
from BatchedArena import BatchedMCTSArena
from MCTS import MCTS

log = logging.getLogger(__name__)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args, opening_sampler=None):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.opening_sampler = opening_sampler
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()
        self.loadedTrainExamplesFile = None

    def _quiet(self):
        return bool(getattr(self.args, 'quiet', False))

    def _self_play_batch_size(self):
        return max(1, int(getattr(self.args, 'selfPlayBatchSize', 1)))

    def _arena_batch_size(self):
        return max(1, int(getattr(self.args, 'arenaBatchSize', 1)))

    def _initial_board(self):
        if self.opening_sampler is not None:
            return self.opening_sampler.sample_self_play_board()
        return self.game.getInitBoard()

    def _arena_opening_suite(self):
        if self.opening_sampler is None:
            return None
        return self.opening_sampler.sample_arena_suite(int(self.args.arenaCompare / 2))

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self._initial_board()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

    def executeEpisodesBatched(self, numEpisodes):
        """
        Executes self-play episodes in a batch of active games. MCTS trees stay
        independent per game, but neural-network leaf evaluations are batched
        across those trees.
        """
        completedExamples = []
        activeEpisodes = []
        launched = 0
        completed = 0
        batch_size = self._self_play_batch_size()
        progress = tqdm(total=numEpisodes, desc="Self Play", disable=self._quiet())

        try:
            while completed < numEpisodes:
                while launched < numEpisodes and len(activeEpisodes) < batch_size:
                    activeEpisodes.append({
                        'board': self._initial_board(),
                        'curPlayer': 1,
                        'episodeStep': 0,
                        'trainExamples': [],
                        'mcts': MCTS(self.game, self.nnet, self.args),
                    })
                    launched += 1

                for episode in activeEpisodes:
                    episode['episodeStep'] += 1
                    episode['canonicalBoard'] = self.game.getCanonicalForm(
                        episode['board'],
                        episode['curPlayer'],
                    )
                    episode['temp'] = int(episode['episodeStep'] < self.args.tempThreshold)

                action_probs = self._getBatchedActionProbs(activeEpisodes)
                still_active = []

                for episode, pi in zip(activeEpisodes, action_probs):
                    sym = self.game.getSymmetries(episode['canonicalBoard'], pi)
                    for b, p in sym:
                        episode['trainExamples'].append([b, episode['curPlayer'], p, None])

                    action = np.random.choice(len(pi), p=pi)
                    episode['board'], episode['curPlayer'] = self.game.getNextState(
                        episode['board'],
                        episode['curPlayer'],
                        action,
                    )

                    r = self.game.getGameEnded(episode['board'], episode['curPlayer'])
                    if r != 0:
                        completedExamples.extend(
                            (x[0], x[2], r * ((-1) ** (x[1] != episode['curPlayer'])))
                            for x in episode['trainExamples']
                        )
                        completed += 1
                        progress.update(1)
                    else:
                        still_active.append(episode)

                activeEpisodes = still_active
        finally:
            progress.close()

        return completedExamples

    def _getBatchedActionProbs(self, episodes):
        for _ in range(self.args.numMCTSSims):
            pending = []

            for episode in episodes:
                leaf = episode['mcts'].select_leaf(episode['canonicalBoard'])
                if leaf['needs_eval']:
                    pending.append((episode['mcts'], leaf))
                else:
                    episode['mcts'].complete_search(leaf)

            if not pending:
                continue

            boards = [leaf['board'] for _, leaf in pending]
            if hasattr(self.nnet, 'predict_batch'):
                policies, values = self.nnet.predict_batch(boards)
            else:
                predictions = [self.nnet.predict(board) for board in boards]
                policies, values = zip(*predictions)

            for (mcts, leaf), policy, value in zip(pending, policies, values):
                mcts.complete_search(leaf, policy, float(value))

        return [
            episode['mcts'].getActionProbFromTree(episode['canonicalBoard'], temp=episode['temp'])
            for episode in episodes
        ]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                if self._self_play_batch_size() > 1:
                    iterationTrainExamples += self.executeEpisodesBatched(self.args.numEps)
                else:
                    for _ in tqdm(range(self.args.numEps), desc="Self Play", disable=self._quiet()):
                        self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                        iterationTrainExamples += self.executeEpisode()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i - 1)
            self.saveTrainExamplesFile('latest.examples')

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

            self.nnet.train(trainExamples)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena_opening_suite = self._arena_opening_suite()
            if self._arena_batch_size() > 1:
                arena = BatchedMCTSArena(
                    self.game,
                    self.pnet,
                    self.nnet,
                    self.args,
                    batch_size=self._arena_batch_size(),
                    quiet=self._quiet(),
                    opening_boards=arena_opening_suite,
                )
                pwins, nwins, draws = arena.playGames(self.args.arenaCompare)
            else:
                pmcts = MCTS(self.game, self.pnet, self.args)
                nmcts = MCTS(self.game, self.nnet, self.args)
                arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                              lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game,
                              opening_boards=arena_opening_suite)
                pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.saveTrainExamplesFile(self.getCheckpointFile(i) + ".examples")
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
                self.saveTrainExamplesFile('best.pth.tar.examples')

            if i == 1 and getattr(self.args, 'deleteLoadedExamplesAfterFirstIteration', False):
                self._deleteLoadedTrainExamplesFile()

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        self.saveTrainExamplesFile(self.getCheckpointFile(iteration) + ".examples")

    def saveTrainExamplesFile(self, examples_filename):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, examples_filename)
        temp_filename = filename + ".tmp"
        self._pruneCheckpointExampleFiles(pending_filename=examples_filename)
        with open(temp_filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        os.replace(temp_filename, filename)
        self._pruneCheckpointExampleFiles()

    def _checkpointExamplesToKeep(self):
        return getattr(self.args, 'checkpointExamplesToKeep', None)

    def _checkpointExampleIteration(self, filename):
        filename = os.path.basename(filename)
        prefix = 'checkpoint_'
        suffix = '.pth.tar.examples'
        if not filename.startswith(prefix) or not filename.endswith(suffix):
            return None

        iteration = filename[len(prefix):-len(suffix)]
        if not iteration.isdigit():
            return None
        return int(iteration)

    def _pruneCheckpointExampleFiles(self, pending_filename=None):
        keep_count = self._checkpointExamplesToKeep()
        if keep_count is None:
            return

        keep_count = int(keep_count)
        if keep_count < 0:
            return

        folder = self.args.checkpoint
        if not os.path.isdir(folder):
            return

        checkpoint_examples = []
        if pending_filename is not None:
            pending_iteration = self._checkpointExampleIteration(pending_filename)
            if pending_iteration is not None:
                pending_path = os.path.join(folder, os.path.basename(pending_filename))
                checkpoint_examples.append((pending_iteration, float('inf'), pending_path))

        for filename in os.listdir(folder):
            iteration = self._checkpointExampleIteration(filename)
            if iteration is None:
                continue
            path = os.path.join(folder, filename)
            checkpoint_examples.append((iteration, os.path.getmtime(path), path))

        checkpoint_examples.sort(reverse=True)
        keep_paths = {path for _, _, path in checkpoint_examples[:keep_count]}
        for _, _, path in checkpoint_examples[keep_count:]:
            if path in keep_paths:
                continue
            try:
                os.remove(path)
            except FileNotFoundError:
                pass

    def _protectedExamplesFiles(self):
        folder = os.path.abspath(self.args.checkpoint)
        return {
            os.path.join(folder, 'latest.examples'),
            os.path.join(folder, 'best.pth.tar.examples'),
        }

    def _deleteLoadedTrainExamplesFile(self):
        loaded_examples_file = getattr(self, 'loadedTrainExamplesFile', None)
        if not loaded_examples_file:
            return

        self.loadedTrainExamplesFile = None
        loaded_examples_file = os.path.abspath(loaded_examples_file)
        if loaded_examples_file in self._protectedExamplesFiles():
            return

        try:
            os.remove(loaded_examples_file)
            log.info('Deleted loaded trainExamples file "%s".', loaded_examples_file)
        except FileNotFoundError:
            pass
        except OSError as exc:
            log.warning('Could not delete loaded trainExamples file "%s": %s', loaded_examples_file, exc)

    def _examplesCandidates(self, examplesFile=None):
        load_folder, load_file = self.args.load_folder_file
        modelFile = os.path.join(load_folder, load_file)
        candidates = []

        if examplesFile:
            if os.path.isabs(examplesFile):
                candidates.append(examplesFile)
            else:
                candidates.append(os.path.join(load_folder, examplesFile))
                candidates.append(examplesFile)

        candidates.extend([
            modelFile + ".examples",
            os.path.join(load_folder, 'latest.examples'),
            os.path.join(load_folder, 'best.pth.tar.examples'),
        ])

        if os.path.isdir(load_folder):
            checkpoint_examples = [
                os.path.join(load_folder, filename)
                for filename in os.listdir(load_folder)
                if filename.startswith('checkpoint_') and filename.endswith('.pth.tar.examples')
            ]
            candidates.extend(sorted(checkpoint_examples, key=os.path.getmtime, reverse=True))

        deduped = []
        seen = set()
        for candidate in candidates:
            normalized = os.path.abspath(candidate)
            if normalized not in seen:
                seen.add(normalized)
                deduped.append(candidate)
        return deduped

    def loadTrainExamples(self, examplesFile=None, skipFirstSelfPlay=True):
        candidates = self._examplesCandidates(examplesFile)
        found_examples = next((path for path in candidates if os.path.isfile(path)), None)

        if not found_examples:
            log.warning('No trainExamples file found. Checked: %s', ', '.join(candidates))
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
            return

        log.info('Loading trainExamples from "%s"...', found_examples)
        with open(found_examples, "rb") as f:
            self.trainExamplesHistory = Unpickler(f).load()
        log.info('Loading done!')
        self.loadedTrainExamplesFile = os.path.abspath(found_examples)

        # Keep the legacy behavior available for callers that intentionally resume
        # from examples that were generated by the loaded model.
        self.skipFirstSelfPlay = skipFirstSelfPlay
