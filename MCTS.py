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

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard)
            if i == 0:
                self.add_root_noise(canonicalBoard)

        return self.getActionProbFromTree(canonicalBoard, temp=temp)

    def getActionProbFromTree(self, canonicalBoard, temp=1):
        """
        Returns the MCTS visit-count policy for canonicalBoard without running
        additional simulations.
        """
        s = self.game.stringRepresentation(canonicalBoard)
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
            policy, value = self.nnet.predict(leaf['board'])
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
                return {
                    'needs_eval': True,
                    'path': path,
                    'board': board,
                    'state_key': s,
                }

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
            self._expand_leaf(leaf['state_key'], leaf['board'], policy)
            propagated_value = value
        else:
            propagated_value = leaf['value']

        for s, a, action_index, player_changed in reversed(leaf['path']):
            parent_value = -propagated_value if player_changed else propagated_value
            self._update_edge(s, a, parent_value, action_index=action_index)
            self.Ns[s] += 1
            propagated_value = parent_value

        return propagated_value

    def _expand_leaf(self, s, canonicalBoard, policy):
        valids = self.Vs.get(s)
        if valids is None:
            valids = self.game.getValidMoves(canonicalBoard, 1)
        actions = np.flatnonzero(valids).astype(np.int32)
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
