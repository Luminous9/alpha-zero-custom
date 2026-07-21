import json
import hashlib
import logging
import os
import random
import sys
import time
from collections import Counter, deque
from contextlib import contextmanager
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
import torch
from tqdm import tqdm

from Arena import Arena
from BatchedArena import BatchedMCTSArena
from MCTS import MCTS
from santorini.SantoriniInference import predict_batch_deduplicated

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
    from santorini.SantoriniSymmetryDiagnostics import (
        aggregate_positions,
        build_diagnostic_suite,
        evaluate_suite,
        flatten_aggregate,
        suite_fingerprint,
    )
except ImportError:
    aggregate_positions = build_diagnostic_suite = evaluate_suite = None
    flatten_aggregate = suite_fingerprint = None

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
        self._completed_opening_symmetries = []
        self._placement_scale_game_counts = {'base': 0, 'exploratory': 0}
        self._placement_geometry_records = []
        self._placement_policy_geometry = self._newPlacementPolicyGeometry()
        self._policy_target_stats = self._newPolicyTargetStats()
        self._playout_cap_stats = self._newPlayoutCapStats()
        self._tactical_stats = self._newTacticalStats()
        self._symmetry_telemetry_suite = None
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

    def _newSelfPlayMCTS(self):
        """Create one episode's search with an independently sampled placement scale."""
        episode_args = dotdict(dict(self.args))
        probability = float(self._arg('placementScaleExplorationProbability', 0.0))
        exploratory = (
            self._arg('searchMode', 'puct') == 'gumbel'
            and probability > 0.0
            and np.random.random() < probability
        )
        if exploratory:
            episode_args.gumbelPlacementScale = float(self._arg(
                'placementExplorationGumbelScale',
                self._arg('gumbelPlacementScale', self._arg('gumbelScale', 1.0)),
            ))
            bucket = 'exploratory'
        else:
            bucket = 'base'
        if hasattr(self, '_placement_scale_game_counts'):
            self._placement_scale_game_counts[bucket] += 1
        return MCTS(self.game, self.nnet, episode_args)

    def _recordCompletedOpening(self, board):
        pieces = np.sign(np.asarray(board)[0]).astype(np.int8)
        exact = np.ascontiguousarray(pieces).tobytes()
        symmetries = []
        for rotations in range(4):
            rotated = np.rot90(pieces, rotations)
            symmetries.append(np.ascontiguousarray(rotated).tobytes())
            symmetries.append(np.ascontiguousarray(np.fliplr(rotated)).tobytes())
        self._completed_openings.append(exact)
        self._completed_opening_symmetries.append(min(symmetries))

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

    def _playoutCapSearch(self, canonical_board=None):
        if not bool(self._arg('playoutCapRandomization', False)):
            return True, int(self.args.numMCTSSims)
        if (
            bool(self._arg('playoutCapFullPlacement', True))
            and canonical_board is not None
            and hasattr(self.game, 'isPlacementPhase')
            and self.game.isPlacementPhase(canonical_board)
        ):
            return True, int(self.args.numMCTSSims)
        full_search = bool(
            np.random.random() < float(self._arg('playoutCapFullProbability', 0.25))
        )
        simulations = (
            int(self.args.numMCTSSims)
            if full_search else int(self._arg('playoutCapFastSims', 32))
        )
        return full_search, simulations

    @staticmethod
    def _newPlayoutCapStats():
        return {
            'full': {'placement': 0, 'standard': 0},
            'fast': {'placement': 0, 'standard': 0},
            'simulations': 0,
        }

    def _recordPlayoutCapSearch(self, canonical_board, full_search, simulations):
        if not bool(self._arg('playoutCapRandomization', False)):
            return
        phase = 'placement' if (
            hasattr(self.game, 'isPlacementPhase') and self.game.isPlacementPhase(canonical_board)
        ) else 'standard'
        kind = 'full' if full_search else 'fast'
        self._playout_cap_stats[kind][phase] += 1
        self._playout_cap_stats['simulations'] += int(simulations)

    def _playoutCapTelemetry(self):
        if not bool(self._arg('playoutCapRandomization', False)):
            return {}
        stats = self._playout_cap_stats
        full_moves = sum(stats['full'].values())
        fast_moves = sum(stats['fast'].values())
        total_moves = full_moves + fast_moves
        payload = {
            'playout_cap_full_search_moves': int(full_moves),
            'playout_cap_fast_search_moves': int(fast_moves),
            'playout_cap_full_search_rate': float(full_moves / total_moves) if total_moves else None,
            'playout_cap_average_simulations': (
                float(stats['simulations'] / total_moves) if total_moves else None
            ),
            'playout_cap_total_simulations': int(stats['simulations']),
        }
        for phase in ('placement', 'standard'):
            phase_full = stats['full'][phase]
            phase_fast = stats['fast'][phase]
            phase_total = phase_full + phase_fast
            payload['{}_full_search_moves'.format(phase)] = int(phase_full)
            payload['{}_fast_search_moves'.format(phase)] = int(phase_fast)
            payload['{}_full_search_rate'.format(phase)] = (
                float(phase_full / phase_total) if phase_total else None
            )
        return payload

    @staticmethod
    def _newTacticalStats():
        return {
            'immediate_win': 0,
            'single_forced_block': 0,
            'forced_block_pruned': 0,
            'proven_loss_in_two': 0,
            'simulations_skipped': 0,
        }

    def _recordTacticalRoot(self, tactical, simulations):
        if tactical is None:
            return
        kind = tactical['kind']
        self._tactical_stats[kind] += 1
        if tactical['policy'] is not None:
            self._tactical_stats['simulations_skipped'] += int(simulations)

    def _tacticalTelemetry(self):
        if not bool(self._arg('tacticalShortcuts', True)):
            return {}
        stats = self._tactical_stats
        return {
            'tactical_immediate_win_roots': int(stats['immediate_win']),
            'tactical_single_forced_block_roots': int(stats['single_forced_block']),
            'tactical_forced_block_pruned_roots': int(stats['forced_block_pruned']),
            'tactical_proven_loss_in_two_roots': int(stats['proven_loss_in_two']),
            'tactical_simulations_skipped': int(stats['simulations_skipped']),
        }

    @staticmethod
    def _newSearchSymmetryStats():
        return {
            'root_evaluations': 0,
            'root_orientations': 0,
            'interior_evaluations': 0,
            'inference_batches': 0,
            'inference_requested': 0,
            'inference_executed': 0,
            'inference_reused': 0,
        }

    def _recordInferenceStats(self, stats):
        if not hasattr(self, '_search_symmetry_stats'):
            self._search_symmetry_stats = self._newSearchSymmetryStats()
        self._search_symmetry_stats['inference_batches'] += 1
        for key in ('requested', 'executed', 'reused'):
            self._search_symmetry_stats['inference_{}'.format(key)] += int(stats[key])

    def _recordSearchSymmetryStats(self, mcts):
        if not hasattr(mcts, 'drainSymmetryEvaluationStats'):
            return
        if not hasattr(self, '_search_symmetry_stats'):
            self._search_symmetry_stats = self._newSearchSymmetryStats()
        for key, value in mcts.drainSymmetryEvaluationStats().items():
            self._search_symmetry_stats[key] += int(value)

    def _searchSymmetryTelemetry(self):
        stats = self._search_symmetry_stats
        metrics = {
            'inference_batches': int(stats['inference_batches']),
            'inference_requested_evaluations': int(stats['inference_requested']),
            'inference_executed_evaluations': int(stats['inference_executed']),
            'inference_reused_evaluations': int(stats['inference_reused']),
            'inference_reuse_rate': (
                float(stats['inference_reused'] / stats['inference_requested'])
                if stats['inference_requested'] else None
            ),
        }
        if not bool(self._arg('searchSymmetryEvaluation', False)):
            return metrics if bool(self._arg('inferenceDeduplication', False)) else {}
        root_evaluations = int(stats['root_evaluations'])
        root_orientations = int(stats['root_orientations'])
        metrics.update({
            'search_symmetry_root_evaluations': root_evaluations,
            'search_symmetry_root_orientations': root_orientations,
            'search_symmetry_average_root_orientations': (
                float(root_orientations / root_evaluations)
                if root_evaluations else None
            ),
            'search_symmetry_interior_evaluations': int(
                stats['interior_evaluations']
            ),
        })
        return metrics

    @staticmethod
    def _prepareTacticalRoot(mcts, canonical_board):
        if not hasattr(mcts, 'prepareTacticalRoot'):
            return None
        return mcts.prepareTacticalRoot(canonical_board)

    @staticmethod
    def _prepareSearchRoot(mcts, canonical_board, simulations):
        if hasattr(mcts, 'prepareSearchRoot'):
            mcts.prepareSearchRoot(canonical_board, simulations)

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
            if hasattr(mcts, 'getTrainingPolicyFromTree'):
                training_policy = mcts.getTrainingPolicyFromTree(
                    canonical_board,
                    temp=target_temperature,
                )
            else:
                training_policy = mcts.getActionProbFromTree(
                    canonical_board,
                    temp=target_temperature,
                )
        gumbel_search = bool(
            hasattr(mcts, 'usesGumbelSearch') and mcts.usesGumbelSearch()
        )
        if float(action_temperature) == float(target_temperature) and not gumbel_search:
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
        placementActions = []
        board = self._initial_board()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            full_search, simulations = self._playoutCapSearch(canonicalBoard)
            configured_temperature = self._temperature(canonicalBoard, episodeStep)
            action_temperature = configured_temperature if full_search else 0
            tactical = self._prepareTacticalRoot(self.mcts, canonicalBoard)
            self._recordTacticalRoot(tactical, simulations)
            exact_tactical_policy = tactical is not None and tactical['policy'] is not None
            if exact_tactical_policy:
                training_policy = action_policy = tactical['policy']
                if full_search:
                    self._appendTrainingPosition(
                        trainExamples,
                        canonicalBoard,
                        self.curPlayer,
                        training_policy,
                    )
            elif full_search:
                target_temperature = self._policyTargetTemperature(action_temperature)
                searched_policy = self.mcts.getActionProb(
                    canonicalBoard,
                    temp=target_temperature,
                    num_simulations=simulations,
                    add_root_noise=True,
                )
                if hasattr(self.mcts, 'getTrainingPolicyFromTree'):
                    training_policy = self.mcts.getTrainingPolicyFromTree(
                        canonicalBoard,
                        temp=target_temperature,
                    )
                else:
                    training_policy = searched_policy
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
            else:
                action_policy = self.mcts.getActionProb(
                    canonicalBoard,
                    temp=0,
                    num_simulations=simulations,
                    add_root_noise=False,
                )
            self._recordPlayoutCapSearch(
                canonicalBoard,
                full_search,
                0 if exact_tactical_policy else simulations,
            )
            self._recordSearchSymmetryStats(self.mcts)

            action = np.random.choice(len(action_policy), p=action_policy)
            if hasattr(self.game, 'isPlacementAction') and self.game.isPlacementAction(action):
                self._recordPlacementChoice(episodeStep, action)
                placementActions.append(action)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)
            if episodeStep == 4 and getattr(self.game, 'sequential_placement', False):
                self._recordCompletedOpening(board)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r != 0:
                player_one_result = int(self.curPlayer * r)
                self._completed_game_lengths.append(episodeStep)
                self._completed_game_results.append(player_one_result)
                self._recordCompletedPlacementGeometry(placementActions, player_one_result)
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
                        'mcts': self._newSelfPlayMCTS(),
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
                    episode['fullSearch'], episode['searchSims'] = self._playoutCapSearch(
                        episode['canonicalBoard']
                    )
                    if not episode['fullSearch']:
                        episode['temp'] = 0
                    episode['tactical'] = self._prepareTacticalRoot(
                        episode['mcts'], episode['canonicalBoard']
                    )
                    if not (
                        episode['tactical'] is not None
                        and episode['tactical']['policy'] is not None
                    ):
                        self._prepareSearchRoot(
                            episode['mcts'],
                            episode['canonicalBoard'],
                            episode['searchSims'],
                        )
                    self._recordTacticalRoot(episode['tactical'], episode['searchSims'])

                training_policies, action_policies = self._getBatchedSelfPlayPolicies(activeEpisodes)
                still_active = []

                for episode, training_policy, action_policy in zip(
                    activeEpisodes,
                    training_policies,
                    action_policies,
                ):
                    self._recordSearchSymmetryStats(episode['mcts'])
                    if episode['fullSearch']:
                        self._appendTrainingPosition(
                            episode['trainExamples'],
                            episode['canonicalBoard'],
                            episode['curPlayer'],
                            training_policy,
                        )
                    self._recordPlayoutCapSearch(
                        episode['canonicalBoard'],
                        episode['fullSearch'],
                        (
                            0
                            if episode['tactical'] is not None
                            and episode['tactical']['policy'] is not None
                            else episode['searchSims']
                        ),
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
                        self._recordCompletedOpening(episode['board'])

                    r = self.game.getGameEnded(episode['board'], episode['curPlayer'])
                    if r != 0:
                        player_one_result = int(episode['curPlayer'] * r)
                        self._completed_game_lengths.append(episode['episodeStep'])
                        self._completed_game_results.append(player_one_result)
                        self._recordCompletedPlacementGeometry(
                            episode['placementActions'],
                            player_one_result,
                        )
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
        max_simulations = max(
            int(episode.get('searchSims', self.args.numMCTSSims)) for episode in episodes
        )
        inference_cache = {}
        for simulation_index in range(max_simulations):
            pending = []

            for episode in episodes:
                tactical = episode.get('tactical')
                if tactical is not None and tactical['policy'] is not None:
                    continue
                if simulation_index >= int(episode.get('searchSims', self.args.numMCTSSims)):
                    continue
                leaf = episode['mcts'].select_leaf(episode['canonicalBoard'])
                if leaf['needs_eval']:
                    pending.append((episode['mcts'], leaf))
                else:
                    episode['mcts'].complete_search(leaf)

            if not pending:
                continue

            boards = []
            evaluation_ranges = []
            for mcts, leaf in pending:
                leaf_boards = (
                    mcts.getLeafEvaluationBoards(leaf)
                    if hasattr(mcts, 'getLeafEvaluationBoards')
                    else [leaf['board']]
                )
                start = len(boards)
                boards.extend(leaf_boards)
                evaluation_ranges.append((start, len(boards)))
            if bool(self._arg('inferenceDeduplication', False)):
                policies, values, inference_stats = predict_batch_deduplicated(
                    self.nnet,
                    boards,
                    cache=inference_cache,
                    max_cache_entries=int(self._arg('inferenceCacheSize', 4096)),
                )
                self._recordInferenceStats(inference_stats)
            elif hasattr(self.nnet, 'predict_batch'):
                policies, values = self.nnet.predict_batch(boards)
            else:
                predictions = [self.nnet.predict(board) for board in boards]
                policies, values = zip(*predictions)

            for (mcts, leaf), (start, end) in zip(pending, evaluation_ranges):
                mcts.complete_search(
                    leaf,
                    np.asarray(policies)[start:end],
                    np.asarray(values)[start:end],
                )

            if simulation_index == 0:
                for episode in episodes:
                    tactical = episode.get('tactical')
                    if (
                        episode.get('fullSearch', True)
                        and not (tactical is not None and tactical['policy'] is not None)
                    ):
                        episode['mcts'].add_root_noise(episode['canonicalBoard'])

        pairs = []
        for episode in episodes:
            tactical = episode.get('tactical')
            if tactical is not None and tactical['policy'] is not None:
                pairs.append((tactical['policy'], tactical['policy']))
            elif episode.get('fullSearch', True):
                pairs.append(self._selfPlayPoliciesFromTree(
                    episode['mcts'],
                    episode['canonicalBoard'],
                    episode['temp'],
                ))
            else:
                pairs.append((None, episode['mcts'].getActionProbFromTree(
                    episode['canonicalBoard'],
                    temp=0,
                )))
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
            self._completed_opening_symmetries = []
            self._placement_scale_game_counts = {'base': 0, 'exploratory': 0}
            self._placement_geometry_records = []
            self._placement_policy_geometry = self._newPlacementPolicyGeometry()
            self._policy_target_stats = self._newPolicyTargetStats()
            self._playout_cap_stats = self._newPlayoutCapStats()
            self._tactical_stats = self._newTacticalStats()
            self._search_symmetry_stats = self._newSearchSymmetryStats()
            iterationTrainExamples = None
            # examples of the iteration
            if not self.skipFirstSelfPlay or local_iteration > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                if self._self_play_batch_size() > 1:
                    iterationTrainExamples += self.executeEpisodesBatched(self.args.numEps)
                else:
                    for _ in tqdm(range(self.args.numEps), desc="Self Play", disable=self._quiet()):
                        self.mcts = self._newSelfPlayMCTS()  # reset search tree
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
            self._prepareSymmetryTelemetrySuite(validationExamples)
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
                    'search_mode': self._arg('searchMode', 'puct'),
                    'gumbel_max_considered_actions': self._arg(
                        'gumbelMaxConsideredActions', 16
                    ),
                    'gumbel_scale': self._arg('gumbelScale', 1.0),
                    'gumbel_placement_scale': self._arg(
                        'gumbelPlacementScale', self._arg('gumbelScale', 1.0)
                    ),
                    'placement_scale_exploration_probability': self._arg(
                        'placementScaleExplorationProbability', 0.0
                    ),
                    'placement_exploration_gumbel_scale': self._arg(
                        'placementExplorationGumbelScale',
                        self._arg('gumbelPlacementScale', self._arg('gumbelScale', 1.0)),
                    ),
                    'evaluation_gumbel_scale': self._arg(
                        'evaluationGumbelScale', self._arg('gumbelScale', 1.0)
                    ),
                    'evaluation_gumbel_placement_scale': self._arg(
                        'evaluationGumbelPlacementScale',
                        self._arg('gumbelPlacementScale', self._arg('gumbelScale', 1.0)),
                    ),
                    'policy_target_temperature': self._arg('policyTargetTemperature', None),
                    'max_train_steps': getattr(getattr(self.nnet, 'net_args', None), 'max_train_steps', None),
                    'symmetry_augmentation': self._arg('symmetryAugmentation', 'expanded'),
                    'symmetry_consistency_fraction': getattr(
                        getattr(self.nnet, 'net_args', None),
                        'symmetry_consistency_fraction',
                        0.0,
                    ),
                    'symmetry_consistency_policy_weight': getattr(
                        getattr(self.nnet, 'net_args', None),
                        'symmetry_consistency_policy_weight',
                        0.0,
                    ),
                    'symmetry_consistency_value_weight': getattr(
                        getattr(self.nnet, 'net_args', None),
                        'symmetry_consistency_value_weight',
                        0.0,
                    ),
                    'symmetry_telemetry_sample_size': int(
                        self._arg('symmetryTelemetrySampleSize', 0)
                    ),
                    'inference_deduplication': bool(
                        self._arg('inferenceDeduplication', False)
                    ),
                    'inference_cache_size': int(self._arg('inferenceCacheSize', 0)),
                    'search_symmetry_evaluation': self._arg(
                        'searchSymmetryEvaluation', False
                    ),
                    'root_symmetry_samples': self._arg('rootSymmetrySamples', 1),
                    'placement_root_symmetry_samples': self._arg(
                        'placementRootSymmetrySamples', 1
                    ),
                    'evaluation_root_symmetry_samples': self._arg(
                        'evaluationRootSymmetrySamples', 1
                    ),
                    'evaluation_placement_root_symmetry_samples': self._arg(
                        'evaluationPlacementRootSymmetrySamples', 1
                    ),
                    'replay_reuse': self._arg('replayReuse', None),
                    'validation_fraction': self._arg('validationFraction', 0.0),
                    'optimizer': getattr(getattr(self.nnet, 'net_args', None), 'optimizer', None),
                    'learning_rate': metrics.get('learning_rate'),
                    'weight_decay': getattr(getattr(self.nnet, 'net_args', None), 'weight_decay', None),
                    'lr_schedule': getattr(getattr(self.nnet, 'net_args', None), 'lr_schedule', None),
                    'playout_cap_randomization': self._arg('playoutCapRandomization', False),
                    'playout_cap_full_probability': self._arg('playoutCapFullProbability', None),
                    'playout_cap_fast_sims': self._arg('playoutCapFastSims', None),
                    'playout_cap_full_placement': self._arg('playoutCapFullPlacement', True),
                    'tactical_shortcuts': self._arg('tacticalShortcuts', True),
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
            'search_mode': self._arg('searchMode', 'puct'),
            'gumbel_max_considered_actions': int(
                self._arg('gumbelMaxConsideredActions', 16)
            ),
            'gumbel_scale': float(self._arg('gumbelScale', 1.0)),
            'gumbel_placement_scale': float(
                self._arg('gumbelPlacementScale', self._arg('gumbelScale', 1.0))
            ),
            'placement_scale_exploration_probability': float(
                self._arg('placementScaleExplorationProbability', 0.0)
            ),
            'placement_exploration_gumbel_scale': float(self._arg(
                'placementExplorationGumbelScale',
                self._arg('gumbelPlacementScale', self._arg('gumbelScale', 1.0)),
            )),
            'evaluation_gumbel_scale': float(
                self._arg('evaluationGumbelScale', self._arg('gumbelScale', 1.0))
            ),
            'evaluation_gumbel_placement_scale': float(self._arg(
                'evaluationGumbelPlacementScale',
                self._arg('gumbelPlacementScale', self._arg('gumbelScale', 1.0)),
            )),
            'policy_target_temperature': self._arg('policyTargetTemperature', None),
            'standard_action_temperature_threshold': int(self.args.tempThreshold),
            'placement_action_temperature': float(self._arg('placementTemperature', 1.0)),
            'playout_cap_randomization': bool(self._arg('playoutCapRandomization', False)),
            'playout_cap_full_probability': (
                float(self._arg('playoutCapFullProbability', 0.25))
                if self._arg('playoutCapRandomization', False) else None
            ),
            'playout_cap_fast_sims': (
                int(self._arg('playoutCapFastSims', 32))
                if self._arg('playoutCapRandomization', False) else None
            ),
            'playout_cap_full_placement': bool(self._arg('playoutCapFullPlacement', True)),
            'tactical_shortcuts': bool(self._arg('tacticalShortcuts', True)),
            'search_symmetry_evaluation': bool(
                self._arg('searchSymmetryEvaluation', False)
            ),
            'inference_deduplication': bool(
                self._arg('inferenceDeduplication', False)
            ),
            'inference_cache_size': int(self._arg('inferenceCacheSize', 0)),
            'root_symmetry_samples': int(self._arg('rootSymmetrySamples', 1)),
            'placement_root_symmetry_samples': int(
                self._arg('placementRootSymmetrySamples', 1)
            ),
            'evaluation_root_symmetry_samples': int(
                self._arg('evaluationRootSymmetrySamples', 1)
            ),
            'evaluation_placement_root_symmetry_samples': int(
                self._arg('evaluationPlacementRootSymmetrySamples', 1)
            ),
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
        payload.update(self._placementScaleTelemetry())
        payload.update(self._placementTelemetry())
        payload.update(self._placementGeometryTelemetry())
        payload.update(self._policyTargetTelemetry())
        payload.update(self._playoutCapTelemetry())
        payload.update(self._tacticalTelemetry())
        payload.update(self._searchSymmetryTelemetry())
        symmetry_metrics = self._symmetryTelemetry()
        payload.update(symmetry_metrics)
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

        placement_games = (
            payload.get('placement_base_scale_games', 0)
            + payload.get('placement_exploratory_scale_games', 0)
        )
        if placement_games and self._quiet():
            log.info(
                'Self-play placement mix: base=%s exploratory=%s (%.1f%%); '
                'completed openings exact=%s symmetry-unique=%s.',
                payload['placement_base_scale_games'],
                payload['placement_exploratory_scale_games'],
                100.0 * payload['placement_exploratory_scale_game_rate'],
                payload.get('unique_completed_openings', 'n/a'),
                payload.get('symmetry_unique_completed_openings', 'n/a'),
            )

        if symmetry_metrics and self._quiet():
            log.info(
                'Held-out D4 value symmetry: placement range=%s sign_disagreement=%s; '
                'standard range=%s sign_disagreement=%s',
                self._formatTelemetryMetric(
                    symmetry_metrics.get('symmetry_placement_value_mean_orbit_range')
                ),
                self._formatTelemetryMetric(
                    symmetry_metrics.get('symmetry_placement_value_sign_disagreement_rate')
                ),
                self._formatTelemetryMetric(
                    symmetry_metrics.get('symmetry_all_standard_value_mean_orbit_range')
                ),
                self._formatTelemetryMetric(
                    symmetry_metrics.get('symmetry_all_standard_value_sign_disagreement_rate')
                ),
            )
        requested_inferences = payload.get('inference_requested_evaluations', 0)
        if requested_inferences and self._quiet():
            log.info(
                'Exact inference reuse: %s / %s evaluation(s) reused (%.1f%%); %s executed.',
                payload['inference_reused_evaluations'],
                requested_inferences,
                100.0 * payload['inference_reuse_rate'],
                payload['inference_executed_evaluations'],
            )

    @staticmethod
    def _formatTelemetryMetric(value):
        return 'n/a' if value is None else '{:.4f}'.format(float(value))

    def _symmetryTelemetrySuitePath(self):
        return os.path.join(self.args.checkpoint, 'symmetry_telemetry_suite.npz')

    def _prepareSymmetryTelemetrySuite(self, validation_examples):
        requested = int(self._arg('symmetryTelemetrySampleSize', 0))
        if requested <= 0 or getattr(self, '_symmetry_telemetry_suite', None) is not None:
            return
        if build_diagnostic_suite is None:
            raise RuntimeError('Symmetry telemetry support is unavailable.')

        path = self._symmetryTelemetrySuitePath()
        if os.path.isfile(path):
            with np.load(path, allow_pickle=False) as payload:
                stored_requested = int(payload.get('requested_sample_size', len(payload['boards'])))
                if stored_requested == requested:
                    self._symmetry_telemetry_suite = {
                        'boards': payload['boards'].astype(int),
                        'targets': payload['targets'].astype(np.float32),
                        'buckets': payload['buckets'].astype(str),
                    }
                    return

        boards, targets, buckets = build_diagnostic_suite(
            self.game,
            validation_examples,
            requested,
        )
        if not boards:
            return
        suite = {
            'boards': np.asarray(boards, dtype=np.int8),
            'targets': np.asarray(targets, dtype=np.float32),
            'buckets': np.asarray(buckets, dtype='<U10'),
        }
        os.makedirs(self.args.checkpoint, exist_ok=True)
        temporary_path = path + '.tmp.npz'
        np.savez_compressed(
            temporary_path,
            requested_sample_size=np.asarray(requested, dtype=np.int32),
            **suite
        )
        os.replace(temporary_path, path)
        self._symmetry_telemetry_suite = suite
        log.info(
            'Created fixed D4 telemetry suite with %s held-out position(s): %s',
            len(boards),
            path,
        )

    def _symmetryTelemetry(self):
        suite = getattr(self, '_symmetry_telemetry_suite', None)
        if not suite:
            return {}
        with preserve_rng_state():
            result = evaluate_suite(
                self.game,
                self.nnet,
                suite['boards'],
                targets=suite['targets'],
                buckets=suite['buckets'],
            )
        aggregate = dict(result['aggregate'])
        standard_positions = [
            position for position in result['positions']
            if position['bucket'] != 'placement'
        ]
        aggregate['all_standard'] = aggregate_positions(standard_positions)
        metrics = flatten_aggregate(aggregate)
        metrics.update({
            'symmetry_telemetry_requested_positions': int(
                self._arg('symmetryTelemetrySampleSize', 0)
            ),
            'symmetry_telemetry_suite_fingerprint': suite_fingerprint(
                suite['boards'],
                suite['targets'],
                suite['buckets'],
            ),
        })
        return metrics

    def _recordPlacementChoice(self, ply, action):
        square = int(action) // self.game.local_action_size
        self._placement_choices.append((int(ply), square))

    def _placementCoordinates(self, action):
        square = int(action) // self.game.local_action_size
        return divmod(square, self.game.n)

    @staticmethod
    def _chebyshevDistance(first, second):
        return max(abs(first[0] - second[0]), abs(first[1] - second[1]))

    def _workerPairGeometry(self, locations):
        center = ((self.game.n - 1) / 2.0, (self.game.n - 1) / 2.0)
        center_distances = [
            max(abs(location[0] - center[0]), abs(location[1] - center[1]))
            for location in locations
        ]
        central_low = (self.game.n - 3) // 2
        central_high = central_low + 2
        return {
            'mean_center_distance': float(np.mean(center_distances)),
            'both_central': bool(all(
                central_low <= row <= central_high and central_low <= column <= central_high
                for row, column in locations
            )),
            'worker_separation': float(self._chebyshevDistance(*locations)),
        }

    def _recordCompletedPlacementGeometry(self, placement_actions, player_one_result):
        if not getattr(self.game, 'sequential_placement', False) or len(placement_actions) < 2:
            return
        player_one = [self._placementCoordinates(action) for action in placement_actions[:2]]
        record = {
            'result': int(player_one_result),
            'p1': self._workerPairGeometry(player_one),
        }
        if len(placement_actions) >= 4:
            player_two = [self._placementCoordinates(action) for action in placement_actions[2:4]]
            record['p2'] = self._workerPairGeometry(player_two)
            opponent_distances = np.asarray([
                [self._chebyshevDistance(p2_location, p1_location) for p1_location in player_one]
                for p2_location in player_two
            ], dtype=np.float64)
            nearest_distances = opponent_distances.min(axis=1)
            center = (self.game.n // 2, self.game.n // 2)
            record.update({
                'center_owner': (
                    1 if center in player_one else (-1 if center in player_two else 0)
                ),
                'minimum_opponent_distance': float(opponent_distances.min()),
                'p2_mean_nearest_p1_distance': float(nearest_distances.mean()),
                'p2_adjacent_to_p1_rate': float(np.mean(nearest_distances == 1)),
            })
        self._placement_geometry_records.append(record)

    @staticmethod
    def _newPlacementPolicyGeometry():
        stats = {
            player: {
                'expected_center_distance': [],
                'center_mass': [],
                'inner_ring_mass': [],
                'outer_ring_mass': [],
                'expected_worker_separation': [],
            }
            for player in ('p1', 'p2')
        }
        stats.update({
            'p2_first_center_available': [],
            'p2_center_mass_when_available': [],
        })
        return stats

    def _recordPlacementPolicyGeometry(self, canonical_board, policy):
        if not getattr(self.game, 'sequential_placement', False):
            return
        occupied = int(np.count_nonzero(canonical_board[0]))
        if occupied not in (0, 1, 2, 3):
            return

        local_action = int(getattr(self.game, 'PLACEMENT_LOCAL_ACTION', 64))
        action_indices = (
            np.arange(self.game.n * self.game.n, dtype=np.int64) * self.game.local_action_size
            + local_action
        )
        probabilities = np.asarray(policy, dtype=np.float64)[action_indices]
        total_mass = float(probabilities.sum())
        if total_mass <= 0:
            return
        probabilities /= total_mass

        center = ((self.game.n - 1) / 2.0, (self.game.n - 1) / 2.0)
        locations = [divmod(square, self.game.n) for square in range(self.game.n * self.game.n)]
        center_distances = np.asarray([
            max(abs(row - center[0]), abs(column - center[1]))
            for row, column in locations
        ], dtype=np.float64)
        player = 'p1' if occupied < 2 else 'p2'
        stats = self._placement_policy_geometry[player]
        stats['expected_center_distance'].append(float(probabilities @ center_distances))
        stats['center_mass'].append(float(probabilities[center_distances == 0].sum()))
        stats['inner_ring_mass'].append(float(probabilities[center_distances == 1].sum()))
        stats['outer_ring_mass'].append(float(probabilities[center_distances >= 2].sum()))

        if player == 'p2':
            center_location = (self.game.n // 2, self.game.n // 2)
            center_available = bool(canonical_board[0][center_location] == 0)
            if occupied == 2:
                self._placement_policy_geometry['p2_first_center_available'].append(center_available)
            if center_available:
                self._placement_policy_geometry['p2_center_mass_when_available'].append(
                    float(probabilities[center_distances == 0].sum())
                )

        if occupied in (1, 3):
            existing = tuple(int(value) for value in np.argwhere(canonical_board[0] > 0)[0])
            separations = np.asarray([
                self._chebyshevDistance(existing, location) for location in locations
            ], dtype=np.float64)
            stats['expected_worker_separation'].append(float(probabilities @ separations))

    def _placementGeometryTelemetry(self):
        records = self._placement_geometry_records
        if not records:
            return {}

        payload = {}
        for player, win_result in (('p1', 1), ('p2', -1)):
            player_records = [record for record in records if player in record]
            if not player_records:
                continue
            winners = [record for record in player_records if record['result'] == win_result]
            losers = [record for record in player_records if record['result'] == -win_result]
            separations = np.asarray([
                record[player]['worker_separation'] for record in player_records
            ], dtype=np.float64)

            def mean(records_for_metric, key):
                return float(np.mean([record[player][key] for record in records_for_metric]))

            payload.update({
                '{}_placement_games'.format(player): int(len(player_records)),
                '{}_placement_mean_center_distance'.format(player): (
                    mean(player_records, 'mean_center_distance')
                ),
                '{}_placement_both_central_rate'.format(player): mean(player_records, 'both_central'),
                '{}_placement_mean_worker_separation'.format(player): float(separations.mean()),
                '{}_placement_adjacent_rate'.format(player): float(np.mean(separations == 1)),
                '{}_placement_moderate_separation_rate'.format(player): float(np.mean(separations == 2)),
                '{}_placement_far_separation_rate'.format(player): float(np.mean(separations >= 3)),
                '{}_winner_placement_games'.format(player): int(len(winners)),
                '{}_loser_placement_games'.format(player): int(len(losers)),
                '{}_winner_mean_center_distance'.format(player): (
                    mean(winners, 'mean_center_distance') if winners else None
                ),
                '{}_loser_mean_center_distance'.format(player): (
                    mean(losers, 'mean_center_distance') if losers else None
                ),
                '{}_winner_mean_worker_separation'.format(player): (
                    mean(winners, 'worker_separation') if winners else None
                ),
                '{}_loser_mean_worker_separation'.format(player): (
                    mean(losers, 'worker_separation') if losers else None
                ),
            })

        interaction_records = [record for record in records if 'p2' in record]
        if interaction_records:
            payload.update({
                'placement_center_owned_by_p1_rate': float(np.mean([
                    record['center_owner'] == 1 for record in interaction_records
                ])),
                'placement_center_owned_by_p2_rate': float(np.mean([
                    record['center_owner'] == -1 for record in interaction_records
                ])),
                'placement_center_unoccupied_rate': float(np.mean([
                    record['center_owner'] == 0 for record in interaction_records
                ])),
                'placement_mean_minimum_opponent_distance': float(np.mean([
                    record['minimum_opponent_distance'] for record in interaction_records
                ])),
                'p2_placement_mean_nearest_p1_distance': float(np.mean([
                    record['p2_mean_nearest_p1_distance'] for record in interaction_records
                ])),
                'p2_placement_adjacent_to_p1_rate': float(np.mean([
                    record['p2_adjacent_to_p1_rate'] for record in interaction_records
                ])),
            })

        policy_stats = self._placement_policy_geometry
        for player in ('p1', 'p2'):
            for key in (
                'expected_center_distance',
                'center_mass',
                'inner_ring_mass',
                'outer_ring_mass',
                'expected_worker_separation',
            ):
                values = policy_stats[player][key]
                if values:
                    payload['{}_policy_{}'.format(player, key)] = float(np.mean(values))
        center_available = policy_stats['p2_first_center_available']
        if center_available:
            payload['p2_first_placement_center_available_rate'] = float(np.mean(center_available))
        available_center_mass = policy_stats['p2_center_mass_when_available']
        if available_center_mass:
            payload['p2_policy_center_mass_when_available'] = float(
                np.mean(available_center_mass)
            )
        return payload

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
        if phase == 'placement':
            self._recordPlacementPolicyGeometry(canonical_board, policy)

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
        exact_counts = Counter(self._completed_openings)
        symmetry_counts = Counter(self._completed_opening_symmetries)
        completed = len(self._completed_openings)
        payload = {
            'unique_completed_openings': int(len(exact_counts)),
            'symmetry_unique_completed_openings': int(len(symmetry_counts)),
            'most_frequent_completed_opening_rate': (
                float(max(exact_counts.values()) / completed) if completed else None
            ),
            'most_frequent_completed_opening_symmetry_rate': (
                float(max(symmetry_counts.values()) / completed) if completed else None
            ),
        }
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

    def _placementScaleTelemetry(self):
        counts = getattr(
            self,
            '_placement_scale_game_counts',
            {'base': 0, 'exploratory': 0},
        )
        total = int(counts['base'] + counts['exploratory'])
        return {
            'placement_base_scale_games': int(counts['base']),
            'placement_exploratory_scale_games': int(counts['exploratory']),
            'placement_exploratory_scale_game_rate': (
                float(counts['exploratory'] / total) if total else None
            ),
        }

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
        args.gumbelScale = float(self._arg(
            'evaluationGumbelScale',
            self._arg('gumbelScale', 1.0),
        ))
        args.gumbelPlacementScale = float(self._arg(
            'evaluationGumbelPlacementScale',
            self._arg('gumbelPlacementScale', self._arg('gumbelScale', 1.0)),
        ))
        args.rootSymmetrySamples = int(self._arg(
            'evaluationRootSymmetrySamples',
            self._arg('rootSymmetrySamples', 1),
        ))
        args.placementRootSymmetrySamples = int(self._arg(
            'evaluationPlacementRootSymmetrySamples',
            self._arg('placementRootSymmetrySamples', args.rootSymmetrySamples),
        ))
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
        include_placement_diagnostics=False,
        opponent_search_symmetry_evaluation=None,
    ):
        current_args = self._matchArgs(simulations)
        opponent_args = dotdict(dict(current_args))
        if opponent_search_symmetry_evaluation is not None:
            opponent_args.searchSymmetryEvaluation = bool(
                opponent_search_symmetry_evaluation
            )
        arena = BatchedMCTSArena(
            self.game,
            opponent,
            self.nnet,
            current_args,
            batch_size=max(1, int(self._arg('telemetryMatchBatchSize', self._self_play_batch_size()))),
            quiet=self._quiet(),
            opening_boards=opening_boards,
            placement_temperature=placement_temperature,
            game_seeds=game_seeds,
            player_args={1: opponent_args, -1: current_args},
            record_placement_diagnostics=include_placement_diagnostics,
        )
        opponent_wins, current_wins, draws = arena.playGames(game_count)
        result = (
            int(opponent_wins),
            int(current_wins),
            int(draws),
            *self._matchStatistics(opponent_wins, current_wins, draws),
        )
        if include_placement_diagnostics:
            return result, arena.placementDiagnostics()
        return result

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

    @staticmethod
    def _placementDiagnosticMetrics(prefix, diagnostics):
        if not diagnostics:
            return {}
        games = int(diagnostics['games_recorded'])
        duplicate_games = int(diagnostics['duplicate_game_count'])
        repeated_groups = int(diagnostics['repeated_exact_labeled_opening_groups'])
        identical_groups = int(
            diagnostics['repeated_groups_with_identical_standard_trajectory']
        )
        return {
            '{}_distinct_exact_openings'.format(prefix): int(
                diagnostics['distinct_exact_openings']
            ),
            '{}_distinct_symmetry_unique_openings'.format(prefix): int(
                diagnostics['distinct_symmetry_unique_openings']
            ),
            '{}_duplicate_games'.format(prefix): duplicate_games,
            '{}_duplicate_rate'.format(prefix): (
                float(duplicate_games / games) if games else None
            ),
            '{}_most_frequent_opening_count'.format(prefix): int(
                diagnostics['most_frequent_opening_count']
            ),
            '{}_repeated_trajectory_groups'.format(prefix): repeated_groups,
            '{}_identical_trajectory_groups'.format(prefix): identical_groups,
            '{}_divergent_trajectory_groups'.format(prefix): int(
                diagnostics['repeated_groups_with_divergent_standard_trajectories']
            ),
            '{}_trajectory_consistency_rate'.format(prefix): (
                float(identical_groups / repeated_groups) if repeated_groups else None
            ),
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
            placement_result, placement_diagnostics = self._runPairedMatch(
                previous,
                placement_game_count,
                placement_temperature=placement_temperature,
                game_seeds=self._telemetry_placement_seeds[:seed_count],
                include_placement_diagnostics=True,
            )
            self._logMatchResult(
                'Placement-inclusive milestone',
                iteration,
                previous_iteration,
                placement_result,
            )
            metrics.update(self._matchMetrics('placement_milestone', *placement_result))
            metrics.update(self._placementDiagnosticMetrics(
                'placement_milestone',
                placement_diagnostics,
            ))
            metrics['placement_milestone_seed_count'] = seed_count
            metrics['placement_milestone_temperature'] = placement_temperature
            if placement_diagnostics:
                log.info(
                    'Placement-inclusive diversity: %s exact / %s symmetry-unique openings; '
                    '%s/%s duplicate games (%.1f%%); most frequent opening %s games; '
                    'repeated trajectory groups %s identical / %s divergent.',
                    placement_diagnostics['distinct_exact_openings'],
                    placement_diagnostics['distinct_symmetry_unique_openings'],
                    placement_diagnostics['duplicate_game_count'],
                    placement_diagnostics['games_recorded'],
                    100.0 * metrics['placement_milestone_duplicate_rate'],
                    placement_diagnostics['most_frequent_opening_count'],
                    placement_diagnostics[
                        'repeated_groups_with_identical_standard_trajectory'
                    ],
                    placement_diagnostics[
                        'repeated_groups_with_divergent_standard_trajectories'
                    ],
                )
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
            opponent_search_symmetry_evaluation=(architecture == 'v3'),
        )
        self._logMatchResult('{} anchor'.format(architecture.upper()), iteration, architecture, result)
        metrics = self._matchMetrics('anchor', *result)
        metrics.update({
            'anchor_architecture': architecture,
            'anchor_opening_count': opening_count,
            'anchor_mcts_simulations': simulations,
        })
        return metrics
