import json
import logging
import os
import sys
import time
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from Arena import Arena
from BatchedArena import BatchedMCTSArena
from MCTS import MCTS

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

try:
    from santorini.ReplayBuffer import load_compact_replay, save_compact_replay
except ImportError:
    load_compact_replay = save_compact_replay = None

try:
    from santorini.SantoriniTelemetry import ReferenceSuite
except ImportError:
    ReferenceSuite = None

log = logging.getLogger(__name__)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args, opening_sampler=None):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.training_mode = self._arg('trainingMode', 'arena')
        self.pnet = self.nnet.__class__(self.game) if self.training_mode == 'arena' else None
        self.opening_sampler = opening_sampler
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()
        self.loadedTrainExamplesFile = None
        self._completed_game_lengths = []
        self._completed_game_results = []
        self._placement_choices = []
        self._completed_openings = []
        self._reference_suite = None
        reference_suite_path = self._arg('referenceSuite', None)
        if reference_suite_path:
            if ReferenceSuite is None:
                raise RuntimeError('Reference-suite telemetry support is unavailable.')
            self._reference_suite = ReferenceSuite(reference_suite_path)
        self._writer = None
        telemetry_dir = self._arg('telemetryDir', None)
        if telemetry_dir and SummaryWriter is not None:
            self._writer = SummaryWriter(telemetry_dir)

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

    def _game_supports_draws(self):
        return bool(getattr(self.game, 'supports_draws', True))

    def _temperature(self, canonical_board, episode_step):
        if hasattr(self.game, 'isPlacementPhase') and self.game.isPlacementPhase(canonical_board):
            return float(self._arg('placementTemperature', 1.0))
        placement_steps = 4 if getattr(self.game, 'sequential_placement', False) else 0
        standard_step = max(1, episode_step - placement_steps)
        return int(standard_step < self.args.tempThreshold)

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
            temp = self._temperature(canonicalBoard, episodeStep)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            if hasattr(self.game, 'isPlacementAction') and self.game.isPlacementAction(action):
                self._recordPlacementChoice(episodeStep, action)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)
            if episodeStep == 4 and getattr(self.game, 'sequential_placement', False):
                self._completed_openings.append(board[0].tobytes())

            r = self.game.getGameEnded(board, self.curPlayer)

            if r != 0:
                self._completed_game_lengths.append(episodeStep)
                self._completed_game_results.append(int(self.curPlayer * r))
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
                        'placementActions': [],
                        'mcts': MCTS(self.game, self.nnet, self.args),
                    })
                    launched += 1

                for episode in activeEpisodes:
                    episode['episodeStep'] += 1
                    episode['canonicalBoard'] = self.game.getCanonicalForm(
                        episode['board'],
                        episode['curPlayer'],
                    )
                    episode['temp'] = self._temperature(
                        episode['canonicalBoard'],
                        episode['episodeStep'],
                    )

                action_probs = self._getBatchedActionProbs(activeEpisodes)
                still_active = []

                for episode, pi in zip(activeEpisodes, action_probs):
                    sym = self.game.getSymmetries(episode['canonicalBoard'], pi)
                    for b, p in sym:
                        episode['trainExamples'].append([b, episode['curPlayer'], p, None])

                    action = np.random.choice(len(pi), p=pi)
                    if hasattr(self.game, 'isPlacementAction') and self.game.isPlacementAction(action):
                        self._recordPlacementChoice(episode['episodeStep'], action)
                        episode['placementActions'].append(action)
                    episode['board'], episode['curPlayer'] = self.game.getNextState(
                        episode['board'],
                        episode['curPlayer'],
                        action,
                    )
                    if episode['episodeStep'] == 4 and getattr(self.game, 'sequential_placement', False):
                        self._completed_openings.append(episode['board'][0].tobytes())

                    r = self.game.getGameEnded(episode['board'], episode['curPlayer'])
                    if r != 0:
                        self._completed_game_lengths.append(episode['episodeStep'])
                        self._completed_game_results.append(int(episode['curPlayer'] * r))
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
        for simulation_index in range(self.args.numMCTSSims):
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

            if simulation_index == 0:
                for episode in episodes:
                    episode['mcts'].add_root_noise(episode['canonicalBoard'])

        return [
            episode['mcts'].getActionProbFromTree(episode['canonicalBoard'], temp=episode['temp'])
            for episode in episodes
        ]

    def learn(self):
        """
        Performs numIters iterations of self-play and training. Arena mode pits
        the new network against the old one; latest mode immediately advances
        the trained network and records non-gating telemetry.
        """

        start_iteration = int(self._arg('startIteration', 0))
        for local_iteration, i in enumerate(
            range(start_iteration + 1, start_iteration + self.args.numIters + 1),
            start=1,
        ):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            iteration_started = time.time()
            self._completed_game_lengths = []
            self._completed_game_results = []
            self._placement_choices = []
            self._completed_openings = []
            # examples of the iteration
            if not self.skipFirstSelfPlay or local_iteration > 1:
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
            self.saveTrainExamplesFile(self._latestExamplesFilename())
            if not self._saveBestTrainExamples():
                self._deleteExamplesFile('best.pth.tar.examples')

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)
            telemetry_sample_size = min(int(self._arg('telemetrySampleSize', 256)), len(trainExamples))
            self._telemetry_boards = [trainExamples[index][0] for index in range(telemetry_sample_size)]

            if self.training_mode == 'latest':
                metrics = self.nnet.train(trainExamples)
                metadata = {
                    'iteration': i,
                    'training_mode': self.training_mode,
                }
                self.nnet.save_checkpoint(
                    folder=self.args.checkpoint,
                    filename='latest-training.pth.tar',
                    include_optimizer=True,
                    metadata=metadata,
                )
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='latest.pth.tar')
                milestone_interval = max(1, int(self._arg('milestoneInterval', 10)))
                if i % milestone_interval == 0:
                    self.nnet.save_checkpoint(
                        folder=self.args.checkpoint,
                        filename=self.getCheckpointFile(i),
                    )
                    metrics.update(self._runMilestoneMatch(i, milestone_interval))
                self._writeTelemetry(i, metrics, time.time() - iteration_started, len(trainExamples))
                if local_iteration == 1 and self._arg('deleteLoadedExamplesAfterFirstIteration', False):
                    self._deleteLoadedTrainExamplesFile()
                continue

            # Arena mode keeps a copy of the old network for promotion testing.
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

            metrics = self.nnet.train(trainExamples)

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

            if self._game_supports_draws():
                log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            else:
                log.info('NEW/PREV WINS : %d / %d' % (nwins, pwins))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.saveTrainExamplesFile(self.getCheckpointFile(i) + ".examples")
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
                if self._saveBestTrainExamples():
                    self.saveTrainExamplesFile('best.pth.tar.examples')
                else:
                    self._deleteExamplesFile('best.pth.tar.examples')

            self._writeTelemetry(i, metrics, time.time() - iteration_started, len(trainExamples))

            if local_iteration == 1 and self._arg('deleteLoadedExamplesAfterFirstIteration', False):
                self._deleteLoadedTrainExamplesFile()

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def _latestExamplesFilename(self):
        return 'latest.examples.npz' if self._arg('compactReplay', False) else 'latest.examples'

    def saveTrainExamples(self, iteration):
        if self._checkpointExamplesToKeep() == 0:
            return
        self.saveTrainExamplesFile(self.getCheckpointFile(iteration) + ".examples")

    def saveTrainExamplesFile(self, examples_filename):
        if self._checkpointExampleIteration(examples_filename) is not None and self._checkpointExamplesToKeep() == 0:
            return

        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, examples_filename)
        temp_filename = filename + ".tmp"
        self._deleteStaleExampleTempFiles()
        self._pruneCheckpointExampleFiles(pending_filename=examples_filename)

        compact = examples_filename.endswith('.npz')
        if compact and save_compact_replay is None:
            raise RuntimeError('Compact replay support is unavailable.')

        if self._atomicExamplesSave():
            if compact:
                save_compact_replay(temp_filename, self.trainExamplesHistory)
            else:
                with open(temp_filename, "wb+") as f:
                    Pickler(f).dump(self.trainExamplesHistory)
            os.replace(temp_filename, filename)
        else:
            self._deleteFile(temp_filename)
            if compact:
                save_compact_replay(filename, self.trainExamplesHistory)
            else:
                with open(filename, "wb+") as f:
                    Pickler(f).dump(self.trainExamplesHistory)

        self._pruneCheckpointExampleFiles()

    def _checkpointExamplesToKeep(self):
        return self._arg('checkpointExamplesToKeep', None)

    def _atomicExamplesSave(self):
        return self._arg('atomicExamplesSave', True)

    def _saveBestTrainExamples(self):
        return self._arg('saveBestTrainExamples', True)

    def _arg(self, name, default=None):
        if hasattr(self.args, 'get'):
            return self.args.get(name, default)
        return getattr(self.args, name, default)

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
            os.path.join(folder, 'latest.examples.npz'),
            os.path.join(folder, 'best.pth.tar.examples'),
        }

    def _deleteFile(self, path):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    def _deleteExamplesFile(self, examples_filename):
        self._deleteFile(os.path.join(self.args.checkpoint, examples_filename))

    def _deleteStaleExampleTempFiles(self):
        folder = self.args.checkpoint
        if not os.path.isdir(folder):
            return

        for filename in os.listdir(folder):
            if filename.endswith('.examples.tmp') or filename.endswith('.examples.npz.tmp'):
                self._deleteFile(os.path.join(folder, filename))

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
            os.path.join(load_folder, 'latest.examples.npz'),
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
        if found_examples.endswith('.npz'):
            if load_compact_replay is None:
                raise RuntimeError('Compact replay support is unavailable.')
            self.trainExamplesHistory = load_compact_replay(found_examples)
        else:
            with open(found_examples, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
        log.info('Loading done!')
        self.loadedTrainExamplesFile = os.path.abspath(found_examples)

        # Keep the legacy behavior available for callers that intentionally resume
        # from examples that were generated by the loaded model.
        self.skipFirstSelfPlay = skipFirstSelfPlay

    def _writeTelemetry(self, iteration, training_metrics, duration_seconds, replay_examples):
        lengths = np.asarray(self._completed_game_lengths, dtype=np.float64)
        standard_lengths = np.maximum(
            0,
            lengths - (4 if getattr(self.game, 'sequential_placement', False) else 0),
        )
        results = np.asarray(self._completed_game_results, dtype=np.float64)
        payload = {
            'iteration': int(iteration),
            'duration_seconds': float(duration_seconds),
            'replay_examples': int(replay_examples),
            'games': int(len(lengths)),
            'average_game_plies': float(lengths.mean()) if len(lengths) else None,
            'average_standard_plies': float(standard_lengths.mean()) if len(lengths) else None,
            'median_standard_plies': float(np.median(standard_lengths)) if len(lengths) else None,
            'standard_plies_p25': float(np.percentile(standard_lengths, 25)) if len(lengths) else None,
            'standard_plies_p75': float(np.percentile(standard_lengths, 75)) if len(lengths) else None,
            'standard_plies_p90': float(np.percentile(standard_lengths, 90)) if len(lengths) else None,
            'first_player_win_rate': float(np.mean(results == 1)) if len(results) else None,
        }
        payload.update(training_metrics or {})
        payload.update(self._placementTelemetry())
        payload.update(self._policyTelemetry())
        if self._reference_suite is not None:
            payload.update(self._reference_suite.evaluate(self.game, self.nnet))

        telemetry_dir = self._arg('telemetryDir', None)
        if telemetry_dir:
            os.makedirs(telemetry_dir, exist_ok=True)
            with open(os.path.join(telemetry_dir, 'telemetry.jsonl'), 'a') as telemetry_file:
                telemetry_file.write(json.dumps(payload, sort_keys=True) + '\n')

        if self._writer is not None:
            for key, value in payload.items():
                if key != 'iteration' and isinstance(value, (int, float)) and value is not None:
                    self._writer.add_scalar(key, value, iteration)
            self._writer.flush()

    def _recordPlacementChoice(self, ply, action):
        square = int(action) // self.game.local_action_size
        self._placement_choices.append((int(ply), square))

    def _placementTelemetry(self):
        if not self._placement_choices:
            return {}
        heatmaps = np.zeros((4, self.game.n * self.game.n), dtype=np.int64)
        for ply, square in self._placement_choices:
            if 1 <= ply <= 4:
                heatmaps[ply - 1, square] += 1
        payload = {'unique_completed_openings': int(len(set(self._completed_openings)))}
        epsilon = 1e-12
        for ply, counts in enumerate(heatmaps, start=1):
            total = counts.sum()
            if total:
                probabilities = counts[counts > 0] / total
                payload['placement_{}_selection_entropy'.format(ply)] = float(
                    -np.sum(probabilities * np.log(probabilities + epsilon)) / np.log(self.game.n * self.game.n)
                )
                payload['placement_{}_max_square_frequency'.format(ply)] = float(counts.max() / total)
                if self._writer is not None:
                    image = counts.reshape(1, self.game.n, self.game.n).astype(np.float32)
                    image /= max(1.0, float(image.max()))
                    self._writer.add_image('placement/ply_{}_heatmap'.format(ply), image, dataformats='CHW')
        return payload

    def _policyTelemetry(self):
        boards = getattr(self, '_telemetry_boards', None)
        if not boards:
            return {}
        policies, _ = self.nnet.predict_batch(boards)
        buckets = {
            'placement': {'mass': [], 'entropy': []},
            'standard': {'mass': [], 'entropy': []},
        }
        epsilon = 1e-12
        for board, policy in zip(boards, policies):
            phase = 'placement' if (
                hasattr(self.game, 'isPlacementPhase') and self.game.isPlacementPhase(board)
            ) else 'standard'
            valids = self.game.getValidMoves(board, 1).astype(bool)
            legal = policy[valids]
            mass = float(legal.sum())
            buckets[phase]['mass'].append(mass)
            if mass > 0 and len(legal) > 1:
                normalized = legal / mass
                buckets[phase]['entropy'].append(
                    -float(np.sum(normalized * np.log(normalized + epsilon))) / np.log(len(legal))
                )
        metrics = {}
        for phase, values in buckets.items():
            if values['mass']:
                metrics['{}_legal_policy_mass'.format(phase)] = float(np.mean(values['mass']))
            if values['entropy']:
                metrics['{}_normalized_legal_entropy'.format(phase)] = float(np.mean(values['entropy']))
        return metrics

    def _runMilestoneMatch(self, iteration, milestone_interval):
        game_count = int(self._arg('telemetryMatchGames', 40))
        previous_iteration = iteration - milestone_interval
        previous_filename = self.getCheckpointFile(previous_iteration)
        previous_path = os.path.join(self.args.checkpoint, previous_filename)
        if game_count <= 0 or previous_iteration <= 0 or not os.path.isfile(previous_path):
            return {}

        log.info(
            'Running non-gating milestone telemetry: checkpoint %s vs %s (%s games)',
            iteration,
            previous_iteration,
            game_count,
        )
        previous = self.nnet.__class__(self.game)
        previous.load_checkpoint(self.args.checkpoint, previous_filename)
        arena = BatchedMCTSArena(
            self.game,
            previous,
            self.nnet,
            self.args,
            batch_size=max(1, int(self._arg('telemetryMatchBatchSize', self._self_play_batch_size()))),
            quiet=self._quiet(),
        )
        previous_wins, current_wins, draws = arena.playGames(game_count)
        decisive = previous_wins + current_wins
        win_rate = float(current_wins / decisive) if decisive else None
        confidence_low = confidence_high = None
        if decisive:
            z = 1.96
            denominator = 1.0 + z * z / decisive
            center = (win_rate + z * z / (2.0 * decisive)) / denominator
            margin = z * np.sqrt(
                win_rate * (1.0 - win_rate) / decisive + z * z / (4.0 * decisive * decisive)
            ) / denominator
            confidence_low = float(center - margin)
            confidence_high = float(center + margin)
            log.info(
                'Milestone result: checkpoint %s vs %s = %s-%s-%s '
                '(current-previous-draws), current win rate %.1f%%, 95%% CI %.1f%%-%.1f%%',
                iteration,
                previous_iteration,
                current_wins,
                previous_wins,
                draws,
                100.0 * win_rate,
                100.0 * confidence_low,
                100.0 * confidence_high,
            )
        else:
            log.info(
                'Milestone result: checkpoint %s vs %s = %s-%s-%s '
                '(current-previous-draws); no decisive games',
                iteration,
                previous_iteration,
                current_wins,
                previous_wins,
                draws,
            )
        return {
            'milestone_opponent_iteration': int(previous_iteration),
            'milestone_previous_wins': int(previous_wins),
            'milestone_current_wins': int(current_wins),
            'milestone_draws': int(draws),
            'milestone_current_win_rate': win_rate,
            'milestone_current_win_rate_95ci_low': confidence_low,
            'milestone_current_win_rate_95ci_high': confidence_high,
        }
