import logging
import math

import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores policy values compacted to each state's legal actions
        self.Qs = {}  # stores Q values compacted to each state's legal actions
        self.Nsas = {}  # stores visit counts compacted to each state's legal actions
        self.As = {}  # maps compact edge slots to global action indices

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # temporarily caches dense valid masks until a leaf is expanded
        self.noised_roots = set()
        self.root_action_overrides = {}
        self.tactical_roots = {}
        self.raw_values = {}
        self.gumbel_roots = {}
        self.symmetry_evaluated_roots = set()
        self.pending_symmetry_roots = {}
        self.symmetry_rng = None
        self.symmetry_evaluation_stats = {
            'root_evaluations': 0,
            'root_orientations': 0,
            'interior_evaluations': 0,
        }

    def _arg(self, key, default=None):
        if hasattr(self.args, 'get'):
            return self.args.get(key, default)
        return getattr(self.args, key, default)

    def usesGumbelSearch(self):
        return str(self._arg('searchMode', 'puct')).lower() == 'gumbel'

    def usesSymmetryEvaluation(self):
        """Whether neural leaf evaluations should be randomized over D4."""
        return bool(self._arg('searchSymmetryEvaluation', False)) and hasattr(
            self.game,
            'getPolicySymmetryPermutation',
        )

    def _symmetry_generator(self):
        return self.symmetry_rng if self.symmetry_rng is not None else np.random

    def _root_symmetry_count(self, canonicalBoard):
        count = int(self._arg('rootSymmetrySamples', 1))
        if (
            hasattr(self.game, 'isPlacementPhase')
            and self.game.isPlacementPhase(canonicalBoard)
        ):
            count = int(self._arg('placementRootSymmetrySamples', count))
        return max(1, min(8, count))

    def _sample_symmetry_ids(self, count):
        count = max(1, min(8, int(count)))
        if count == 8:
            return tuple(range(8))
        generator = self._symmetry_generator()
        return tuple(int(value) for value in generator.choice(8, size=count, replace=False))

    @staticmethod
    def _transform_board_symmetry(board, symmetry_id):
        rotations = int(symmetry_id) // 2
        flip = bool(int(symmetry_id) % 2)
        transformed = np.rot90(np.asarray(board), rotations, axes=(-2, -1))
        if flip:
            transformed = np.flip(transformed, axis=-1)
        return np.ascontiguousarray(transformed)

    def _restore_policy_symmetry(self, policy, symmetry_id):
        rotations = int(symmetry_id) // 2
        flip = bool(int(symmetry_id) % 2)
        old_indices, new_indices = self.game.getPolicySymmetryPermutation(
            rotations,
            flip,
        )
        restored = np.empty(self.game.getActionSize(), dtype=np.asarray(policy).dtype)
        restored[old_indices] = np.asarray(policy)[new_indices]
        return restored

    def getLeafEvaluationBoards(self, leaf):
        """Return the oriented network inputs needed to evaluate one selected leaf."""
        symmetry_ids = leaf.get('eval_symmetry_ids')
        if symmetry_ids is None:
            return [leaf['board']]
        return [
            self._transform_board_symmetry(leaf['board'], symmetry_id)
            for symmetry_id in symmetry_ids
        ]

    def drainSymmetryEvaluationStats(self):
        """Return and reset search-symmetry counters for training telemetry."""
        stats = dict(self.symmetry_evaluation_stats)
        for key in self.symmetry_evaluation_stats:
            self.symmetry_evaluation_stats[key] = 0
        return stats

    @staticmethod
    def _gumbel_considered_visits(max_num_considered_actions, num_simulations):
        """Port of mctx's Sequential Halving visit schedule."""
        if num_simulations <= 0:
            return ()
        if max_num_considered_actions <= 1:
            return tuple(range(num_simulations))
        log2max = int(math.ceil(math.log2(max_num_considered_actions)))
        sequence = []
        visits = [0] * max_num_considered_actions
        num_considered = max_num_considered_actions
        while len(sequence) < num_simulations:
            extra_visits = max(
                1,
                int(num_simulations / (log2max * num_considered)),
            )
            for _ in range(extra_visits):
                sequence.extend(visits[:num_considered])
                for index in range(num_considered):
                    visits[index] += 1
            num_considered = max(2, num_considered // 2)
        return tuple(sequence[:num_simulations])

    def prepareSearchRoot(self, canonicalBoard, num_simulations, rng=None):
        """Register a root and its simulation budget for the selected search mode."""
        state_key = self.game.stringRepresentation(canonicalBoard)
        if rng is not None:
            self.symmetry_rng = rng
        if (
            self.usesSymmetryEvaluation()
            and state_key not in self.symmetry_evaluated_roots
            and state_key not in self.pending_symmetry_roots
        ):
            self.pending_symmetry_roots[state_key] = self._sample_symmetry_ids(
                self._root_symmetry_count(canonicalBoard)
            )

        if not self.usesGumbelSearch():
            return
        if state_key in self.gumbel_roots:
            return
        already_expanded = state_key in self.Ps
        is_placement = bool(
            hasattr(self.game, 'isPlacementPhase')
            and self.game.isPlacementPhase(canonicalBoard)
        )
        gumbel_scale = float(self._arg('gumbelScale', 1.0))
        if is_placement:
            placement_scale = self._arg('gumbelPlacementScale', None)
            if placement_scale is not None:
                gumbel_scale = float(placement_scale)
        self.gumbel_roots[state_key] = {
            # This MCTS implementation spends its first simulation expanding
            # a new root. A child retained from the prior move is already
            # expanded and can spend its entire budget on root edges.
            'edge_budget': max(
                0,
                int(num_simulations) - (
                    1
                    if state_key in self.pending_symmetry_roots
                    else (0 if already_expanded else 1)
                ),
            ),
            'gumbel': None,
            'considered_visits': None,
            'baseline_visits': None,
            'rng': rng,
            'gumbel_scale': gumbel_scale,
        }

    @staticmethod
    def _uniform_policy(action_size, actions):
        policy = np.zeros(action_size, dtype=np.float32)
        if len(actions):
            policy[np.asarray(actions, dtype=np.int64)] = 1.0 / len(actions)
        return policy

    def prepareTacticalRoot(self, canonicalBoard):
        """
        Detect exact level-three wins and one-ply level-three defenses.

        Exact tactical policies bypass search. If several safe defenses exist,
        the root is restricted to them and normal search chooses among them.
        """
        if not bool(self._arg('tacticalShortcuts', True)):
            return None
        if not hasattr(self.game, 'getImmediateLevelThreeMoves'):
            return None
        if hasattr(self.game, 'isPlacementPhase') and self.game.isPlacementPhase(canonicalBoard):
            return None

        state_key = self.game.stringRepresentation(canonicalBoard)
        if state_key in self.tactical_roots:
            return self.tactical_roots[state_key]

        ended, valids = self._get_game_ended_and_valids(canonicalBoard)
        if ended != 0:
            self.Es[state_key] = ended
            self.tactical_roots[state_key] = None
            return None
        if valids is None:
            valids = self.game.getValidMoves(canonicalBoard, 1)
        self.Es[state_key] = 0
        self.Vs[state_key] = valids
        legal_actions = np.flatnonzero(valids).astype(np.int32)
        winning_actions = np.flatnonzero(
            self.game.getImmediateLevelThreeMoves(canonicalBoard, 1, valids=valids)
        ).astype(np.int32)
        if len(winning_actions):
            result = {
                'kind': 'immediate_win',
                'policy': self._uniform_policy(self.game.getActionSize(), winning_actions),
                'actions': winning_actions,
            }
            self.tactical_roots[state_key] = result
            return result

        opponent_board = self.game.getCanonicalForm(canonicalBoard, -1)
        opponent_valids = self.game.getValidMoves(opponent_board, 1)
        opponent_wins = self.game.getImmediateLevelThreeMoves(
            opponent_board,
            1,
            valids=opponent_valids,
        )
        if not np.any(opponent_wins):
            self.tactical_roots[state_key] = None
            return None

        safe_actions = []
        counter_wins = []
        for action in legal_actions:
            next_board, next_player = self.game.getNextState(canonicalBoard, 1, int(action))
            next_canonical = self.game.getCanonicalForm(next_board, next_player)
            ended = self.game.getGameEnded(next_canonical, 1)
            if ended == -1:
                counter_wins.append(int(action))
                continue
            next_valids = self.game.getValidMoves(next_canonical, 1)
            if not np.any(self.game.getImmediateLevelThreeMoves(
                next_canonical,
                1,
                valids=next_valids,
            )):
                safe_actions.append(int(action))

        if counter_wins:
            result = {
                'kind': 'immediate_win',
                'policy': self._uniform_policy(self.game.getActionSize(), counter_wins),
                'actions': np.asarray(counter_wins, dtype=np.int32),
            }
        elif not safe_actions:
            result = {
                'kind': 'proven_loss_in_two',
                'policy': self._uniform_policy(self.game.getActionSize(), legal_actions),
                'actions': legal_actions,
            }
        elif len(safe_actions) == 1:
            result = {
                'kind': 'single_forced_block',
                'policy': self._uniform_policy(self.game.getActionSize(), safe_actions),
                'actions': np.asarray(safe_actions, dtype=np.int32),
            }
        else:
            safe_actions = np.asarray(safe_actions, dtype=np.int32)
            self.root_action_overrides[state_key] = safe_actions
            result = {
                'kind': 'forced_block_pruned',
                'policy': None,
                'actions': safe_actions,
            }
        self.tactical_roots[state_key] = result
        return result

    def getActionProb(
        self,
        canonicalBoard,
        temp=1,
        num_simulations=None,
        add_root_noise=True,
    ):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: PUCT visit-count probabilities, or the one-hot action
                   selected by Gumbel Sequential Halving.
        """
        simulations = (
            int(self.args.numMCTSSims)
            if num_simulations is None else int(num_simulations)
        )
        if simulations < 1:
            raise ValueError('MCTS requires at least one simulation.')
        tactical = self.prepareTacticalRoot(canonicalBoard)
        if tactical is not None and tactical['policy'] is not None:
            return list(tactical['policy'])
        self.prepareSearchRoot(canonicalBoard, simulations)
        for i in range(simulations):
            self.search(canonicalBoard)
            if i == 0 and add_root_noise:
                self.add_root_noise(canonicalBoard)

        return self.getActionProbFromTree(canonicalBoard, temp=temp)

    def getTrainingPolicyFromTree(self, canonicalBoard, temp=1):
        """Return the replay policy target produced by the selected search."""
        if self.usesGumbelSearch():
            return list(self._gumbel_improved_policy(
                self.game.stringRepresentation(canonicalBoard)
            ))
        return self.getActionProbFromTree(canonicalBoard, temp=temp)

    def getActionProbFromTree(self, canonicalBoard, temp=1):
        """
        Returns the MCTS visit-count policy for canonicalBoard without running
        additional simulations.
        """
        s = self.game.stringRepresentation(canonicalBoard)
        if self.usesGumbelSearch() and s in self.Ps:
            if s not in self.gumbel_roots:
                self.prepareSearchRoot(canonicalBoard, 1)
            return list(self._gumbel_action_policy(s))
        if s in self.Nsas:
            actions = self.As[s]
            counts = self.Nsas[s].astype(np.float64)
        else:
            actions = np.arange(self.game.getActionSize(), dtype=np.int32)
            counts = np.array(
                [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())],
                dtype=np.float64,
            )

        counts_sum = float(np.sum(counts))
        if counts_sum == 0:
            if s in self.Ps:
                probs = np.zeros(self.game.getActionSize(), dtype=np.float32)
                probs[actions] = self.Ps[s]
            else:
                valids = self.game.getValidMoves(canonicalBoard, 1)
                probs = valids / np.sum(valids)
            return list(probs)

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = int(actions[int(np.random.choice(bestAs))])
            probs = [0] * self.game.getActionSize()
            probs[bestA] = 1
            return probs

        counts = counts ** (1. / temp)
        counts_sum = float(np.sum(counts))
        probs = np.zeros(self.game.getActionSize(), dtype=np.float64)
        probs[actions] = counts / counts_sum
        return list(probs)

    def getDenseActionCounts(self, state_key):
        """Return dense visit counts for callers that export MCTS statistics."""
        counts = np.zeros(self.game.getActionSize(), dtype=np.int32)
        if state_key in self.Nsas:
            counts[self.As[state_key]] = self.Nsas[state_key]
            return counts
        for action in range(self.game.getActionSize()):
            counts[action] = self.Nsa.get((state_key, action), 0)
        return counts

    def getDenseActionValues(self, state_key):
        """Return dense Q values for callers that export MCTS statistics."""
        values = np.zeros(self.game.getActionSize(), dtype=np.float32)
        if state_key in self.Qs:
            values[self.As[state_key]] = self.Qs[state_key]
            return values
        for action in range(self.game.getActionSize()):
            values[action] = self.Qsa.get((state_key, action), 0.0)
        return values

    def search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        Values are backed up from the acting player's perspective. The sign is
        inverted only across edges where control changes; games may legally
        return the same player for consecutive actions.

        Returns:
            v: the value of the current canonicalBoard's player
        """

        leaf = self.select_leaf(canonicalBoard)
        if leaf['needs_eval']:
            boards = self.getLeafEvaluationBoards(leaf)
            if len(boards) == 1:
                policy, value = self.nnet.predict(boards[0])
            elif hasattr(self.nnet, 'predict_batch'):
                policy, value = self.nnet.predict_batch(boards)
            else:
                predictions = [self.nnet.predict(board) for board in boards]
                policy, value = zip(*predictions)
            return self.complete_search(leaf, policy, value)
        return self.complete_search(leaf)

    def select_leaf(self, canonicalBoard):
        """
        Walks the current tree until it reaches either a terminal node or an
        unexpanded leaf. The returned object can be passed to complete_search()
        after neural network evaluation, which lets callers batch those
        evaluations across multiple MCTS instances.
        """
        path = []
        board = canonicalBoard
        root_key = self.game.stringRepresentation(canonicalBoard)

        if root_key in self.pending_symmetry_roots:
            symmetry_ids = self.pending_symmetry_roots.pop(root_key)
            self.symmetry_evaluation_stats['root_evaluations'] += 1
            self.symmetry_evaluation_stats['root_orientations'] += len(symmetry_ids)
            return {
                'needs_eval': True,
                'path': path,
                'board': board,
                'state_key': root_key,
                'eval_symmetry_ids': symmetry_ids,
                'root_symmetry_refresh': True,
            }

        while True:
            s = self.game.stringRepresentation(board)

            if s not in self.Es:
                self.Es[s], valids = self._get_game_ended_and_valids(board)
                if self.Es[s] == 0 and valids is not None:
                    self.Vs[s] = valids
            if self.Es[s] != 0:
                return {
                    'needs_eval': False,
                    'path': path,
                    'value': self.Es[s],
                }

            if s not in self.Ps:
                leaf = {
                    'needs_eval': True,
                    'path': path,
                    'board': board,
                    'state_key': s,
                }
                if self.usesSymmetryEvaluation():
                    symmetry_id = self._sample_symmetry_ids(1)
                    leaf['eval_symmetry_ids'] = symmetry_id
                    self.symmetry_evaluation_stats['interior_evaluations'] += 1
                return leaf

            a, action_index = self._best_action(s)
            next_s, next_player = self.game.getNextState(board, 1, a)
            path.append((s, a, action_index, next_player != 1))
            board = self.game.getCanonicalForm(next_s, next_player)

    def complete_search(self, leaf, policy=None, value=None):
        """
        Expands an evaluated leaf, then backs its value up along the selection
        path. For terminal leaves, policy/value are omitted.
        """
        if leaf['needs_eval']:
            policy, value = self._combine_symmetry_evaluations(leaf, policy, value)
            if leaf.get('root_symmetry_refresh') and leaf['state_key'] in self.Ps:
                self._refresh_leaf_policy(leaf['state_key'], leaf['board'], policy)
            else:
                self._expand_leaf(leaf['state_key'], leaf['board'], policy)
            propagated_value = float(value)
            self.raw_values[leaf['state_key']] = propagated_value
            if leaf.get('root_symmetry_refresh'):
                self.symmetry_evaluated_roots.add(leaf['state_key'])
        else:
            propagated_value = leaf['value']

        for s, a, action_index, player_changed in reversed(leaf['path']):
            parent_value = -propagated_value if player_changed else propagated_value
            self._update_edge(s, a, parent_value, action_index=action_index)
            self.Ns[s] += 1
            propagated_value = parent_value

        return propagated_value

    def _combine_symmetry_evaluations(self, leaf, policies, values):
        symmetry_ids = leaf.get('eval_symmetry_ids')
        if symmetry_ids is None:
            policies = np.asarray(policies)
            values = np.asarray(values).reshape(-1)
            if policies.ndim > 1:
                if len(policies) != 1:
                    raise ValueError('A non-symmetry leaf requires exactly one policy.')
                policies = policies[0]
            if len(values) != 1:
                raise ValueError('A non-symmetry leaf requires exactly one value.')
            return policies, float(values[0])

        policies = np.asarray(policies)
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if policies.ndim == 1:
            policies = policies.reshape(1, -1)
        if len(policies) != len(symmetry_ids) or len(values) != len(symmetry_ids):
            raise ValueError(
                'Expected {} symmetry evaluation(s), received {} policies and {} values.'.format(
                    len(symmetry_ids),
                    len(policies),
                    len(values),
                )
            )
        restored = [
            self._restore_policy_symmetry(policy, symmetry_id)
            for policy, symmetry_id in zip(policies, symmetry_ids)
        ]
        return np.mean(restored, axis=0), float(np.mean(values))

    def _refresh_leaf_policy(self, s, canonicalBoard, policy):
        """Replace an existing root prior/value estimate without resetting visits."""
        valids = self.game.getValidMoves(canonicalBoard, 1)
        actions = np.flatnonzero(valids).astype(np.int32)
        root_actions = self.root_action_overrides.get(s)
        if root_actions is not None:
            actions = np.intersect1d(actions, root_actions, assume_unique=True).astype(np.int32)
        if not np.array_equal(actions, self.As[s]):
            raise ValueError('Root symmetry refresh changed the legal MCTS action set.')
        legal_policy = np.asarray(policy, dtype=np.float32)[actions].copy()
        policy_sum = float(np.sum(legal_policy))
        if policy_sum > 0:
            legal_policy /= policy_sum
        else:
            log.error('All valid moves were masked during root symmetry averaging.')
            legal_policy.fill(1.0 / len(actions))
        self.Ps[s] = legal_policy

    def _expand_leaf(self, s, canonicalBoard, policy):
        valids = self.Vs.get(s)
        if valids is None:
            valids = self.game.getValidMoves(canonicalBoard, 1)
        actions = np.flatnonzero(valids).astype(np.int32)
        root_actions = self.root_action_overrides.get(s)
        if root_actions is not None:
            actions = np.intersect1d(actions, root_actions, assume_unique=True).astype(np.int32)
        legal_policy = np.asarray(policy, dtype=np.float32)[actions].copy()
        sum_Ps_s = np.sum(legal_policy)
        if sum_Ps_s > 0:
            legal_policy /= sum_Ps_s
        else:
            # if all valid moves were masked make all valid moves equally probable

            # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
            # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
            log.error("All valid moves were masked, doing a workaround.")
            legal_policy = np.full(len(actions), 1.0 / len(actions), dtype=np.float32)

        self.Ps[s] = legal_policy
        self.As[s] = actions
        self.Qs[s] = np.zeros(len(actions), dtype=np.float32)
        self.Nsas[s] = np.zeros(len(actions), dtype=np.int32)
        self.Ns[s] = 0
        self.Vs.pop(s, None)

    def add_root_noise(self, canonicalBoard):
        if self.usesGumbelSearch():
            return
        get_arg = self.args.get if hasattr(self.args, 'get') else lambda key, default: getattr(self.args, key, default)
        if not bool(get_arg('addDirichletNoise', False)):
            return
        s = self.game.stringRepresentation(canonicalBoard)
        if s in self.noised_roots or s not in self.Ps:
            return
        if len(self.As[s]) == 0:
            return
        epsilon = float(get_arg('dirichletEpsilon', 0.25))
        alpha = float(get_arg('dirichletAlpha', 0.30))
        noise = np.random.dirichlet([alpha] * len(self.As[s]))
        self.Ps[s] = np.asarray(
            (1.0 - epsilon) * self.Ps[s] + epsilon * noise,
            dtype=np.float32,
        )
        self.noised_roots.add(s)

    def _get_game_ended_and_valids(self, canonicalBoard):
        if hasattr(self.game, 'getGameEndedAndValidMoves'):
            return self.game.getGameEndedAndValidMoves(canonicalBoard, 1)
        return self.game.getGameEnded(canonicalBoard, 1), None

    def _best_action(self, s):
        if self.usesGumbelSearch():
            if s in self.gumbel_roots:
                return self._best_gumbel_root_action(s)
            return self._best_gumbel_interior_action(s)

        actions = self.As[s]
        edge_counts = self.Nsas[s]
        visited = edge_counts > 0
        u = np.empty(len(actions), dtype=np.float32)

        if np.any(visited):
            u[visited] = (
                self.Qs[s][visited]
                + self.args.cpuct
                * self.Ps[s][visited]
                * math.sqrt(self.Ns[s])
                / (1 + edge_counts[visited])
            )

        if np.any(~visited):
            u[~visited] = (
                self.args.cpuct
                * self.Ps[s][~visited]
                * math.sqrt(self.Ns[s] + EPS)
            )

        action_index = int(np.argmax(u))
        return int(actions[action_index]), action_index

    def _completed_qvalues(self, s):
        """Published completed-by-mixed-value Q transform used by mctx."""
        priors = np.asarray(self.Ps[s], dtype=np.float64)
        qvalues = np.asarray(self.Qs[s], dtype=np.float64)
        visits = np.asarray(self.Nsas[s], dtype=np.float64)
        raw_value = float(self.raw_values.get(s, 0.0))
        visited = visits > 0
        total_visits = float(np.sum(visits))
        if np.any(visited):
            visited_prior_mass = float(np.sum(priors[visited]))
            if visited_prior_mass > 0:
                weighted_q = float(np.sum(
                    priors[visited] * qvalues[visited] / visited_prior_mass
                ))
            else:
                weighted_q = float(np.mean(qvalues[visited]))
            mixed_value = (raw_value + total_visits * weighted_q) / (total_visits + 1.0)
        else:
            mixed_value = raw_value
        completed = np.where(visited, qvalues, mixed_value)
        value_range = float(np.max(completed) - np.min(completed))
        if value_range > EPS:
            completed = (completed - np.min(completed)) / value_range
        else:
            completed = np.zeros_like(completed)
        max_visits = float(np.max(visits)) if len(visits) else 0.0
        return (50.0 + max_visits) * 0.1 * completed

    def _policy_logits(self, s):
        return np.log(np.maximum(np.asarray(self.Ps[s], dtype=np.float64), 1e-30))

    @staticmethod
    def _softmax(logits):
        shifted = np.asarray(logits, dtype=np.float64) - float(np.max(logits))
        weights = np.exp(shifted)
        return weights / np.sum(weights)

    def _initialize_gumbel_root(self, s):
        root = self.gumbel_roots[s]
        if root['gumbel'] is not None:
            return root
        action_count = len(self.As[s])
        max_considered = min(
            int(self._arg('gumbelMaxConsideredActions', 16)),
            action_count,
        )
        generator = root['rng'] if root['rng'] is not None else np.random
        root['gumbel'] = (
            root['gumbel_scale']
            * generator.gumbel(size=action_count)
        )
        root['considered_visits'] = self._gumbel_considered_visits(
            max_considered,
            root['edge_budget'],
        )
        root['baseline_visits'] = self.Nsas[s].copy()
        return root

    def _best_gumbel_root_action(self, s):
        root = self._initialize_gumbel_root(s)
        visits = self.Nsas[s]
        root_visits = visits - root['baseline_visits']
        simulation_index = int(np.sum(root_visits))
        schedule = root['considered_visits']
        if simulation_index >= len(schedule):
            # Defensive fallback for callers that search beyond the registered
            # root budget: continue allocating according to improved policy.
            return self._best_gumbel_interior_action(s)
        considered_visit = schedule[simulation_index]
        score = root['gumbel'] + self._policy_logits(s) + self._completed_qvalues(s)
        score = np.where(root_visits == considered_visit, score, -np.inf)
        action_index = int(np.argmax(score))
        return int(self.As[s][action_index]), action_index

    def _best_gumbel_interior_action(self, s):
        improved = self._softmax(self._policy_logits(s) + self._completed_qvalues(s))
        visits = np.asarray(self.Nsas[s], dtype=np.float64)
        score = improved - visits / (1.0 + float(np.sum(visits)))
        action_index = int(np.argmax(score))
        return int(self.As[s][action_index]), action_index

    def _gumbel_improved_policy(self, s):
        dense = np.zeros(self.game.getActionSize(), dtype=np.float64)
        if s not in self.Ps:
            return dense
        dense[self.As[s]] = self._softmax(
            self._policy_logits(s) + self._completed_qvalues(s)
        )
        return dense

    def _gumbel_action_policy(self, s):
        dense = np.zeros(self.game.getActionSize(), dtype=np.float64)
        root = self._initialize_gumbel_root(s)
        visits = self.Nsas[s]
        root_visits = visits - root['baseline_visits']
        considered_visit = int(np.max(root_visits)) if len(root_visits) else 0
        score = root['gumbel'] + self._policy_logits(s) + self._completed_qvalues(s)
        score = np.where(root_visits == considered_visit, score, -np.inf)
        dense[int(self.As[s][int(np.argmax(score))])] = 1.0
        return dense

    def _update_edge(self, s, a, v, action_index=None):
        if s in self.Qs:
            if action_index is None:
                action_index = int(np.searchsorted(self.As[s], a))
                if action_index >= len(self.As[s]) or int(self.As[s][action_index]) != int(a):
                    raise ValueError('Action {} is not legal for the selected MCTS state.'.format(a))
            visits = self.Nsas[s][action_index]
            self.Qs[s][action_index] = (visits * self.Qs[s][action_index] + v) / (visits + 1)
            self.Nsas[s][action_index] = visits + 1
            self.Qsa[(s, a)] = float(self.Qs[s][action_index])
            self.Nsa[(s, a)] = int(self.Nsas[s][action_index])
        elif (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1
