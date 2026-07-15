import json
import hashlib
import logging
import os
import random
import sys
import time
from collections import deque
from contextlib import contextmanager
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
import torch
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

try:
    from santorini.SantoriniOpeningBook import SantoriniRandomOpeningSampler
except ImportError:
    SantoriniRandomOpeningSampler = None

from utils import dotdict

log = logging.getLogger(__name__)


@contextmanager
def preserve_rng_state():
    """Keep evaluation-only work from changing the resumable training trajectory."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args, opening_sampler=None, anchor_nnet=None):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.training_mode = self._arg('trainingMode', 'arena')
        self.pnet = self.nnet.__class__(self.game) if self.training_mode == 'arena' else None
        self.opening_sampler = opening_sampler
        self.anchor_nnet = anchor_nnet
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()
        self.loadedTrainExamplesFile = None
        self._completed_game_lengths = []
        self._completed_game_results = []
        self._placement_choices = []
        self._completed_openings = []
        self._policy_target_stats = self._newPolicyTargetStats()
        self._reference_suite = None
        reference_suite_path = self._arg('referenceSuite', None)
        if reference_suite_path:
            if ReferenceSuite is None:
                raise RuntimeError('Reference-suite telemetry support is unavailable.')
            self._reference_suite = ReferenceSuite(reference_suite_path)
        self._telemetry_opening_boards = self._buildTelemetryOpeningSuite()
        self._telemetry_placement_seeds = self._buildTelemetryPlacementSeeds()
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

    def _uses_on_the_fly_symmetry(self):
        return self._arg('symmetryAugmentation', 'expanded') == 'on-the-fly'

    def _appendTrainingPosition(self, examples, canonical_board, player, policy):
        if hasattr(self, '_policy_target_stats'):
            self._recordPolicyTarget(canonical_board, policy)
        if self._uses_on_the_fly_symmetry():
            examples.append([canonical_board, player, policy, None])
            return
        for sym_board, sym_policy in self.game.getSymmetries(canonical_board, policy):
            examples.append([sym_board, player, sym_policy, None])

    def _isValidationExample(self, example):
        fraction = float(self._arg('validationFraction', 0.0))
        if fraction <= 0:
            return False
        # Hash the network-visible position rather than its incidental storage
        # representation. Same-color worker labels are anonymous to the encoder,
        # and every D4 transform must stay on the same side of the split so
        # on-the-fly augmentation cannot leak validation inputs into training.
        board = np.asarray(example[0])
        network_board = np.stack((
            np.sign(board[0]),
            np.clip(board[1], 0, 4),
        )).astype(np.int8)
        symmetry_encodings = []
        for rotations in range(4):
            rotated = np.rot90(network_board, rotations, axes=(-2, -1))
            symmetry_encodings.append(np.ascontiguousarray(rotated).tobytes())
            symmetry_encodings.append(np.ascontiguousarray(np.flip(rotated, axis=-1)).tobytes())
        digest = hashlib.blake2b(
            min(symmetry_encodings),
            digest_size=8,
            person=b'SantoriniVal',
        ).digest()
        return int.from_bytes(digest, byteorder='big') < int(fraction * (1 << 64))

    def _splitTrainingValidation(self, examples):
        training = []
        validation = []
        for example in examples:
            (validation if self._isValidationExample(example) else training).append(example)
        return training, validation

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

    def _policyTargetTemperature(self, action_temperature):
        target_temperature = self._arg('policyTargetTemperature', None)
        return action_temperature if target_temperature is None else float(target_temperature)

    def _selfPlayPoliciesFromTree(
        self,
        mcts,
        canonical_board,
        action_temperature,
        training_policy=None,
    ):
        target_temperature = self._policyTargetTemperature(action_temperature)
        if training_policy is None:
            training_policy = mcts.getActionProbFromTree(
                canonical_board,
                temp=target_temperature,
            )
        if float(action_temperature) == float(target_temperature):
            action_policy = training_policy
        else:
            action_policy = mcts.getActionProbFromTree(
                canonical_board,
                temp=action_temperature,
            )
        return training_policy, action_policy

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        Action selection uses temp=1 before tempThreshold and temp=0 afterward.
        The stored policy can use an independent policyTargetTemperature.

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
            action_temperature = self._temperature(canonicalBoard, episodeStep)
            target_temperature = self._policyTargetTemperature(action_temperature)
            training_policy = self.mcts.getActionProb(canonicalBoard, temp=target_temperature)
            training_policy, action_policy = self._selfPlayPoliciesFromTree(
                self.mcts,
                canonicalBoard,
                action_temperature,
                training_policy=training_policy,
            )
            self._appendTrainingPosition(
                trainExamples,
                canonicalBoard,
                self.curPlayer,
                training_policy,
            )

            action = np.random.choice(len(action_policy), p=action_policy)
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

                training_policies, action_policies = self._getBatchedSelfPlayPolicies(activeEpisodes)
                still_active = []

                for episode, training_policy, action_policy in zip(
                    activeEpisodes,
                    training_policies,
                    action_policies,
                ):
                    self._appendTrainingPosition(
                        episode['trainExamples'],
                        episode['canonicalBoard'],
                        episode['curPlayer'],
                        training_policy,
                    )

                    action = np.random.choice(len(action_policy), p=action_policy)
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

    def _getBatchedSelfPlayPolicies(self, episodes):
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

        pairs = [
            self._selfPlayPoliciesFromTree(
                episode['mcts'],
                episode['canonicalBoard'],
                episode['temp'],
            )
            for episode in episodes
        ]
        return (
            [training_policy for training_policy, _ in pairs],
            [action_policy for _, action_policy in pairs],
        )

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
            self._policy_target_stats = self._newPolicyTargetStats()
            iterationTrainExamples = None
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
            replay_example_count = len(trainExamples)
            trainExamples, validationExamples = self._splitTrainingValidation(trainExamples)
            freshExamples = (
                list(iterationTrainExamples)
                if iterationTrainExamples is not None
                else (list(self.trainExamplesHistory[-1]) if self.trainExamplesHistory else [])
            )
            freshTrainingExamples, _ = self._splitTrainingValidation(freshExamples)
            shuffle(trainExamples)
            telemetry_sample_size = min(int(self._arg('telemetrySampleSize', 256)), len(trainExamples))
            self._telemetry_boards = [trainExamples[index][0] for index in range(telemetry_sample_size)]

            if self.training_mode == 'latest':
                metrics = self.nnet.train(
                    trainExamples,
                    new_example_count=len(freshTrainingExamples),
                    validation_examples=validationExamples,
                    iteration=i,
                )
                metadata = {
                    'iteration': i,
                    'training_mode': self.training_mode,
                    'num_mcts_sims': int(self.args.numMCTSSims),
                    'policy_target_temperature': self._arg('policyTargetTemperature', None),
                    'max_train_steps': getattr(getattr(self.nnet, 'net_args', None), 'max_train_steps', None),
                    'symmetry_augmentation': self._arg('symmetryAugmentation', 'expanded'),
                    'replay_reuse': self._arg('replayReuse', None),
                    'validation_fraction': self._arg('validationFraction', 0.0),
                    'optimizer': getattr(getattr(self.nnet, 'net_args', None), 'optimizer', None),
                    'learning_rate': metrics.get('learning_rate'),
                    'weight_decay': getattr(getattr(self.nnet, 'net_args', None), 'weight_decay', None),
                    'lr_schedule': getattr(getattr(self.nnet, 'net_args', None), 'lr_schedule', None),
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
                    with preserve_rng_state():
                        metrics.update(self._runMilestoneMatch(i, milestone_interval))
                anchor_interval = max(1, int(self._arg('anchorInterval', milestone_interval)))
                if self.anchor_nnet is not None and i % anchor_interval == 0:
                    with preserve_rng_state():
                        metrics.update(self._runAnchorMatch(i))
                self._writeTelemetry(i, metrics, time.time() - iteration_started, replay_example_count)
                if local_iteration == 1 and self._arg('deleteLoadedExamplesAfterFirstIteration', False):
                    self._deleteLoadedTrainExamplesFile()
                continue

            # Arena mode keeps a copy of the old network for promotion testing.
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

            metrics = self.nnet.train(
                trainExamples,
                new_example_count=len(freshTrainingExamples),
                validation_examples=validationExamples,
                iteration=i,
            )

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

            self._writeTelemetry(i, metrics, time.time() - iteration_started, replay_example_count)

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
            'num_mcts_sims': int(self.args.numMCTSSims),
            'policy_target_temperature': self._arg('policyTargetTemperature', None),
            'standard_action_temperature_threshold': int(self.args.tempThreshold),
            'placement_action_temperature': float(self._arg('placementTemperature', 1.0)),
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
        payload.update(self._policyTargetTelemetry())
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

    @staticmethod
    def _newPolicyTargetStats():
        return {
            'placement': {'entropy': [], 'support': [], 'one_hot': []},
            'standard': {'entropy': [], 'support': [], 'one_hot': []},
        }

    def _recordPolicyTarget(self, canonical_board, policy):
        phase = 'placement' if (
            hasattr(self.game, 'isPlacementPhase') and self.game.isPlacementPhase(canonical_board)
        ) else 'standard'
        probabilities = np.asarray(policy, dtype=np.float64)
        positive = probabilities[probabilities > 0]
        support = int(len(positive))
        entropy = -float(np.sum(positive * np.log(positive))) if support else 0.0
        bucket = self._policy_target_stats[phase]
        bucket['entropy'].append(entropy)
        bucket['support'].append(support)
        bucket['one_hot'].append(support <= 1)

    def _policyTargetTelemetry(self):
        payload = {}
        for phase, values in self._policy_target_stats.items():
            if not values['entropy']:
                continue
            payload['{}_policy_target_entropy'.format(phase)] = float(np.mean(values['entropy']))
            payload['{}_policy_target_support'.format(phase)] = float(np.mean(values['support']))
            payload['{}_policy_target_one_hot_rate'.format(phase)] = float(np.mean(values['one_hot']))
        return payload

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

    def _buildTelemetryOpeningSuite(self):
        if self.training_mode != 'latest' and self.anchor_nnet is None:
            return []
        standard_games = int(self._arg('telemetryMatchGames', 40))
        anchor_games = int(self._arg('anchorGames', 40)) if self.anchor_nnet is not None else 0
        opening_count = max(standard_games, anchor_games) // 2
        if opening_count <= 0:
            return []
        if SantoriniRandomOpeningSampler is None:
            raise RuntimeError('Symmetry-distinct telemetry opening support is unavailable.')
        seed = int(self._arg('telemetryOpeningSeed', 20260715))
        sampler = SantoriniRandomOpeningSampler(
            board_size=self.game.n,
            random_orientation=True,
            rng=np.random.RandomState(seed),
        )
        return sampler.sample_distinct_arena_suite(opening_count)

    def _buildTelemetryPlacementSeeds(self):
        if self.training_mode != 'latest':
            return []
        game_count = int(self._arg('telemetryPlacementGames', self._arg('telemetryMatchGames', 40)))
        seed_count = game_count // 2
        if seed_count <= 0:
            return []
        rng = np.random.RandomState(int(self._arg('telemetryOpeningSeed', 20260715)) + 1)
        seeds = []
        seen = set()
        while len(seeds) < seed_count:
            value = int(rng.randint(0, 2 ** 31 - 1))
            if value not in seen:
                seen.add(value)
                seeds.append(value)
        return seeds

    def _matchArgs(self, simulations=None):
        args = dotdict(dict(self.args))
        if simulations is not None:
            args.numMCTSSims = int(simulations)
        args.addDirichletNoise = False
        return args

    @staticmethod
    def _matchStatistics(opponent_wins, current_wins, draws):
        decisive = opponent_wins + current_wins
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
        return win_rate, confidence_low, confidence_high

    def _runPairedMatch(
        self,
        opponent,
        game_count,
        opening_boards=None,
        placement_temperature=0.0,
        game_seeds=None,
        simulations=None,
    ):
        arena = BatchedMCTSArena(
            self.game,
            opponent,
            self.nnet,
            self._matchArgs(simulations),
            batch_size=max(1, int(self._arg('telemetryMatchBatchSize', self._self_play_batch_size()))),
            quiet=self._quiet(),
            opening_boards=opening_boards,
            placement_temperature=placement_temperature,
            game_seeds=game_seeds,
        )
        opponent_wins, current_wins, draws = arena.playGames(game_count)
        return (
            int(opponent_wins),
            int(current_wins),
            int(draws),
            *self._matchStatistics(opponent_wins, current_wins, draws),
        )

    @staticmethod
    def _matchMetrics(prefix, opponent_wins, current_wins, draws, win_rate, low, high):
        return {
            '{}_opponent_wins'.format(prefix): opponent_wins,
            '{}_current_wins'.format(prefix): current_wins,
            '{}_draws'.format(prefix): draws,
            '{}_current_win_rate'.format(prefix): win_rate,
            '{}_current_win_rate_95ci_low'.format(prefix): low,
            '{}_current_win_rate_95ci_high'.format(prefix): high,
        }

    def _logMatchResult(self, label, iteration, opponent_label, result):
        opponent_wins, current_wins, draws, win_rate, low, high = result
        if win_rate is None:
            log.info(
                '%s result: checkpoint %s vs %s = %s-%s-%s '
                '(current-opponent-draws); no decisive games',
                label,
                iteration,
                opponent_label,
                current_wins,
                opponent_wins,
                draws,
            )
            return
        log.info(
            '%s result: checkpoint %s vs %s = %s-%s-%s '
            '(current-opponent-draws), current win rate %.1f%%, approximate 95%% CI %.1f%%-%.1f%%',
            label,
            iteration,
            opponent_label,
            current_wins,
            opponent_wins,
            draws,
            100.0 * win_rate,
            100.0 * low,
            100.0 * high,
        )

    def _runMilestoneMatch(self, iteration, milestone_interval):
        game_count = int(self._arg('telemetryMatchGames', 40))
        placement_game_count = int(self._arg('telemetryPlacementGames', game_count))
        previous_iteration = iteration - milestone_interval
        previous_filename = self.getCheckpointFile(previous_iteration)
        previous_path = os.path.join(self.args.checkpoint, previous_filename)
        if previous_iteration <= 0 or not os.path.isfile(previous_path):
            return {}
        if game_count <= 0 and placement_game_count <= 0:
            return {}

        previous = self.nnet.__class__(self.game)
        previous.load_checkpoint(self.args.checkpoint, previous_filename)
        metrics = {'milestone_opponent_iteration': int(previous_iteration)}

        if game_count > 0:
            opening_count = game_count // 2
            log.info(
                'Running standard-play milestone telemetry: checkpoint %s vs %s '
                '(%s games, %s fixed symmetry-distinct openings)',
                iteration,
                previous_iteration,
                game_count,
                opening_count,
            )
            standard_result = self._runPairedMatch(
                previous,
                game_count,
                opening_boards=self._telemetry_opening_boards[:opening_count],
            )
            self._logMatchResult('Standard-play milestone', iteration, previous_iteration, standard_result)
            metrics.update(self._matchMetrics('milestone', *standard_result))
            # Preserve the original field name for existing telemetry consumers.
            metrics['milestone_previous_wins'] = metrics['milestone_opponent_wins']
            metrics['milestone_opening_count'] = opening_count

        if placement_game_count > 0 and getattr(self.game, 'sequential_placement', False):
            seed_count = placement_game_count // 2
            placement_temperature = float(self._arg('telemetryPlacementTemperature', 1.0))
            log.info(
                'Running placement-inclusive milestone telemetry: checkpoint %s vs %s '
                '(%s games, %s paired seeds, placement temperature %.2f)',
                iteration,
                previous_iteration,
                placement_game_count,
                seed_count,
                placement_temperature,
            )
            placement_result = self._runPairedMatch(
                previous,
                placement_game_count,
                placement_temperature=placement_temperature,
                game_seeds=self._telemetry_placement_seeds[:seed_count],
            )
            self._logMatchResult(
                'Placement-inclusive milestone',
                iteration,
                previous_iteration,
                placement_result,
            )
            metrics.update(self._matchMetrics('placement_milestone', *placement_result))
            metrics['placement_milestone_seed_count'] = seed_count
            metrics['placement_milestone_temperature'] = placement_temperature
        return metrics

    def _runAnchorMatch(self, iteration):
        game_count = int(self._arg('anchorGames', 40))
        if self.anchor_nnet is None or game_count <= 0:
            return {}
        opening_count = game_count // 2
        architecture = str(self._arg('anchorArchitecture', 'v1'))
        simulations = int(self._arg('anchorMCTSSims', self.args.numMCTSSims))
        log.info(
            'Running fixed %s anchor telemetry at checkpoint %s '
            '(%s games, %s fixed symmetry-distinct openings, %s simulations)',
            architecture,
            iteration,
            game_count,
            opening_count,
            simulations,
        )
        result = self._runPairedMatch(
            self.anchor_nnet,
            game_count,
            opening_boards=self._telemetry_opening_boards[:opening_count],
            simulations=simulations,
        )
        self._logMatchResult('{} anchor'.format(architecture.upper()), iteration, architecture, result)
        metrics = self._matchMetrics('anchor', *result)
        metrics.update({
            'anchor_architecture': architecture,
            'anchor_opening_count': opening_count,
            'anchor_mcts_simulations': simulations,
        })
        return metrics
