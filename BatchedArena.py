import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm

from MCTS import MCTS
from santorini.SantoriniInference import predict_batch_deduplicated


class BatchedMCTSArena:
    """
    Arena evaluator for neural-MCTS players. It keeps independent MCTS trees per
    active game/player while batching neural-network leaf evaluations.
    """

    def __init__(
        self,
        game,
        player1_nnet,
        player2_nnet,
        args,
        batch_size=1,
        quiet=False,
        opening_boards=None,
        progress_file=None,
        placement_temperature=0.0,
        game_seeds=None,
        player_args=None,
        standard_controller_nnet=None,
        standard_controller_args=None,
        record_placement_diagnostics=False,
    ):
        self.game = game
        self.nnets = {
            1: player1_nnet,
            -1: player2_nnet,
        }
        self.args = args
        self.player_args = player_args or {1: args, -1: args}
        self.batch_size = max(1, int(batch_size))
        self.quiet = quiet
        self.opening_boards = opening_boards
        self.progress_file = progress_file
        self.placement_temperature = float(placement_temperature)
        self.game_seeds = game_seeds
        if (standard_controller_nnet is None) != (standard_controller_args is None):
            raise ValueError(
                'standard_controller_nnet and standard_controller_args must be provided together.'
            )
        self.standard_controller_nnet = standard_controller_nnet
        self.standard_controller_args = standard_controller_args
        self.record_placement_diagnostics = bool(record_placement_diagnostics)
        self.placement_records = []
        self.inference_stats = {
            'batches': 0,
            'requested': 0,
            'executed': 0,
            'reused': 0,
        }

    def playGames(self, num):
        self.placement_records = []
        for key in self.inference_stats:
            self.inference_stats[key] = 0
        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0

        first_results = self._playGamesForSides(
            num,
            {1: 1, -1: -1},
            "BatchedArena.playGames (1)",
            'contestant1_first',
        )
        second_results = self._playGamesForSides(
            num,
            {1: -1, -1: 1},
            "BatchedArena.playGames (2)",
            'contestant2_first',
        )

        for results in (first_results, second_results):
            oneWon += results[0]
            twoWon += results[1]
            draws += results[2]

        return oneWon, twoWon, draws

    def _playGamesForSides(self, num, side_to_player, desc, seat_order=None):
        oneWon = 0
        twoWon = 0
        draws = 0
        launched = 0
        completed = 0
        active = []
        progress = tqdm(
            total=num,
            desc=desc,
            disable=self.quiet,
            file=self.progress_file,
            dynamic_ncols=True,
        )

        try:
            while completed < num:
                while launched < num and len(active) < self.batch_size:
                    opening_board = self.opening_boards[launched] if self.opening_boards is not None else None
                    game_seed = self.game_seeds[launched] if self.game_seeds is not None else None
                    active.append(self._newGame(
                        side_to_player,
                        opening_board=opening_board,
                        game_seed=game_seed,
                        game_index=launched,
                        seat_order=seat_order,
                    ))
                    launched += 1

                for game_state in active:
                    game_state['canonicalBoard'] = self.game.getCanonicalForm(
                        game_state['board'],
                        game_state['curPlayer'],
                    )

                actions = self._getBatchedActions(active)
                still_active = []

                for game_state, action in zip(active, actions):
                    was_placement = self._isPlacement(game_state['canonicalBoard'])
                    if self.record_placement_diagnostics and not was_placement:
                        game_state['standard_actions'].append(int(action))
                    game_state['board'], game_state['curPlayer'] = self.game.getNextState(
                        game_state['board'],
                        game_state['curPlayer'],
                        action,
                    )
                    if (
                        self.record_placement_diagnostics
                        and was_placement
                        and game_state['placement_board'] is None
                    ):
                        next_canonical = self.game.getCanonicalForm(
                            game_state['board'],
                            game_state['curPlayer'],
                        )
                        if not self._isPlacement(next_canonical):
                            game_state['placement_board'] = game_state['board'].copy()
                    ended = self.game.getGameEnded(game_state['board'], game_state['curPlayer'])

                    if ended == 0:
                        still_active.append(game_state)
                        continue

                    game_result = game_state['curPlayer'] * ended
                    if game_result == 1:
                        winner = game_state['side_to_player'][1]
                    elif game_result == -1:
                        winner = game_state['side_to_player'][-1]
                    else:
                        winner = 0

                    if self.record_placement_diagnostics:
                        self._recordPlacementGame(game_state, winner, game_result)

                    if winner == 1:
                        oneWon += 1
                    elif winner == -1:
                        twoWon += 1
                    else:
                        draws += 1

                    completed += 1
                    progress.update(1)

                active = still_active
        finally:
            progress.close()

        return oneWon, twoWon, draws

    def _newGame(
        self,
        side_to_player,
        opening_board=None,
        game_seed=None,
        game_index=None,
        seat_order=None,
    ):
        game_state = {
            'board': opening_board.copy() if opening_board is not None else self.game.getInitBoard(),
            'curPlayer': 1,
            'side_to_player': side_to_player,
            'mcts_by_player': {
                1: MCTS(self.game, self.nnets[1], self.player_args[1]),
                -1: MCTS(self.game, self.nnets[-1], self.player_args[-1]),
            },
            'rng': np.random.RandomState(game_seed) if game_seed is not None else np.random,
            'game_seed': game_seed,
            'game_index': game_index,
            'seat_order': seat_order,
            'placement_board': None,
            'standard_actions': [],
        }
        if self.standard_controller_nnet is not None:
            # The shared standard-play network still gets an independent tree for
            # each physical side. Contestant identity continues to determine who
            # receives the result, because the contestants chose the placements.
            game_state['standard_mcts_by_side'] = {
                1: MCTS(
                    self.game,
                    self.standard_controller_nnet,
                    self.standard_controller_args,
                ),
                -1: MCTS(
                    self.game,
                    self.standard_controller_nnet,
                    self.standard_controller_args,
                ),
            }
        return game_state

    @staticmethod
    def _placementLocations(board):
        pieces = np.asarray(board)[0]
        p1 = tuple(sorted(tuple(int(v) for v in location) for location in np.argwhere(pieces > 0)))
        p2 = tuple(sorted(tuple(int(v) for v in location) for location in np.argwhere(pieces < 0)))
        return p1, p2

    @staticmethod
    def _locationSignature(p1, p2):
        def side_signature(locations):
            return ','.join('{}:{}'.format(row, col) for row, col in locations)

        return 'p1={}|p2={}'.format(side_signature(p1), side_signature(p2))

    @classmethod
    def _symmetryPlacementSignature(cls, board):
        board = np.asarray(board)
        signatures = []
        for rotations in range(4):
            transformed = np.rot90(board, rotations, axes=(-2, -1))
            for flip in (False, True):
                oriented = np.flip(transformed, axis=-1) if flip else transformed
                signatures.append(cls._locationSignature(*cls._placementLocations(oriented)))
        return min(signatures)

    @staticmethod
    def _labeledPlacementKey(board):
        pieces = np.ascontiguousarray(np.asarray(board)[0], dtype=np.int8)
        return pieces.tobytes().hex()

    def _recordPlacementGame(self, game_state, winner, winner_side):
        board = game_state.get('placement_board')
        if board is None:
            return
        p1, p2 = self._placementLocations(board)
        self.placement_records.append({
            'pair_index': game_state.get('game_index'),
            'seat_order': game_state.get('seat_order'),
            'game_seed': game_state.get('game_seed'),
            'p1_contestant': int(game_state['side_to_player'][1]),
            'p2_contestant': int(game_state['side_to_player'][-1]),
            'winner': int(winner),
            'winner_side': int(winner_side),
            'opening': self._locationSignature(p1, p2),
            'labeled_opening_key': self._labeledPlacementKey(board),
            'symmetry_opening': self._symmetryPlacementSignature(board),
            'standard_trajectory': tuple(game_state['standard_actions']),
        })

    def placementDiagnostics(self):
        records = self.placement_records
        if not records:
            return None

        exact_groups = defaultdict(list)
        labeled_groups = defaultdict(list)
        symmetry_counts = Counter()
        for record in records:
            exact_groups[record['opening']].append(record)
            labeled_groups[record['labeled_opening_key']].append(record)
            symmetry_counts[record['symmetry_opening']] += 1

        opening_results = []
        for opening, group in exact_groups.items():
            wins = Counter(record['winner'] for record in group)
            side_wins = Counter(record['winner_side'] for record in group)
            contestant1_as_p1 = sum(record['p1_contestant'] == 1 for record in group)
            contestant1_as_p2 = sum(record['p2_contestant'] == 1 for record in group)
            labeled_variants = {record['labeled_opening_key'] for record in group}
            trajectories = {record['standard_trajectory'] for record in group}
            opening_results.append({
                'opening': opening,
                'games': len(group),
                'contestant1_wins': wins[1],
                'contestant2_wins': wins[-1],
                'draws': wins[0],
                'player1_wins': side_wins[1],
                'player2_wins': side_wins[-1],
                'physical_draws': side_wins[0],
                'contestant1_as_player1_games': contestant1_as_p1,
                'contestant1_as_player2_games': contestant1_as_p2,
                'contestant2_as_player1_games': len(group) - contestant1_as_p1,
                'contestant2_as_player2_games': len(group) - contestant1_as_p2,
                'labeled_variants': len(labeled_variants),
                'distinct_standard_trajectories': len(trajectories),
            })
        opening_results.sort(key=lambda item: (-item['games'], item['opening']))

        repeated_labeled_groups = [group for group in labeled_groups.values() if len(group) > 1]
        identical_trajectory_groups = sum(
            len({record['standard_trajectory'] for record in group}) == 1
            for group in repeated_labeled_groups
        )
        duplicate_games_matching_trajectory = 0
        for group in repeated_labeled_groups:
            trajectory_counts = Counter(record['standard_trajectory'] for record in group)
            duplicate_games_matching_trajectory += sum(
                count - 1 for count in trajectory_counts.values()
            )
        frequency_histogram = Counter(len(group) for group in exact_groups.values())
        most_frequent_count = opening_results[0]['games'] if opening_results else 0

        return {
            'games_recorded': len(records),
            'player1_wins': sum(record['winner_side'] == 1 for record in records),
            'player2_wins': sum(record['winner_side'] == -1 for record in records),
            'physical_draws': sum(record['winner_side'] == 0 for record in records),
            'distinct_exact_openings': len(exact_groups),
            'distinct_exact_labeled_openings': len(labeled_groups),
            'distinct_symmetry_unique_openings': len(symmetry_counts),
            'duplicate_game_count': len(records) - len(exact_groups),
            'opening_frequency_histogram': {
                str(frequency): count
                for frequency, count in sorted(frequency_histogram.items())
            },
            'most_frequent_opening_count': most_frequent_count,
            'most_frequent_openings': [
                item['opening']
                for item in opening_results
                if item['games'] == most_frequent_count
            ],
            'repeated_exact_labeled_opening_groups': len(repeated_labeled_groups),
            'repeated_exact_labeled_games': sum(len(group) for group in repeated_labeled_groups),
            'duplicate_exact_labeled_game_count': len(records) - len(labeled_groups),
            'duplicate_games_matching_an_existing_standard_trajectory': (
                duplicate_games_matching_trajectory
            ),
            'repeated_groups_with_identical_standard_trajectory': identical_trajectory_groups,
            'repeated_groups_with_divergent_standard_trajectories': (
                len(repeated_labeled_groups) - identical_trajectory_groups
            ),
            'opening_results': opening_results,
        }

    def _isPlacement(self, canonical_board):
        return bool(
            hasattr(self.game, 'isPlacementPhase')
            and self.game.isPlacementPhase(canonical_board)
        )

    def _controller(self, game_state):
        if (
            self.standard_controller_nnet is not None
            and not self._isPlacement(game_state['canonicalBoard'])
        ):
            return (
                'standard',
                game_state['standard_mcts_by_side'][game_state['curPlayer']],
                self.standard_controller_nnet,
                self.standard_controller_args,
            )

        player = game_state['side_to_player'][game_state['curPlayer']]
        return (
            player,
            game_state['mcts_by_player'][player],
            self.nnets[player],
            self.player_args[player],
        )

    def _getBatchedActions(self, active):
        for game_state in active:
            _, mcts, _, controller_args = self._controller(game_state)
            game_state['tactical'] = mcts.prepareTacticalRoot(
                game_state['canonicalBoard']
            )
            if not (
                game_state['tactical'] is not None
                and game_state['tactical']['policy'] is not None
            ):
                if hasattr(mcts, 'prepareSearchRoot'):
                    mcts.prepareSearchRoot(
                        game_state['canonicalBoard'],
                        controller_args.numMCTSSims,
                        rng=game_state['rng'],
                    )

        max_simulations = max(
            int(self._controller(game_state)[3].numMCTSSims)
            for game_state in active
        )
        inference_caches = defaultdict(dict)
        for simulation_index in range(max_simulations):
            pending_by_controller = {}

            for game_state in active:
                tactical = game_state.get('tactical')
                if tactical is not None and tactical['policy'] is not None:
                    continue
                controller, mcts, nnet, controller_args = self._controller(game_state)
                if simulation_index >= int(controller_args.numMCTSSims):
                    continue
                leaf = mcts.select_leaf(game_state['canonicalBoard'])
                if leaf['needs_eval']:
                    pending = pending_by_controller.setdefault(
                        controller,
                        {'nnet': nnet, 'args': controller_args, 'leaves': []},
                    )
                    pending['leaves'].append((mcts, leaf))
                else:
                    mcts.complete_search(leaf)

            for controller, pending in pending_by_controller.items():
                leaves = pending['leaves']
                if not leaves:
                    continue

                boards = []
                evaluation_ranges = []
                for mcts, leaf in leaves:
                    leaf_boards = (
                        mcts.getLeafEvaluationBoards(leaf)
                        if hasattr(mcts, 'getLeafEvaluationBoards')
                        else [leaf['board']]
                    )
                    start = len(boards)
                    boards.extend(leaf_boards)
                    evaluation_ranges.append((start, len(boards)))
                nnet = pending['nnet']
                controller_args = pending['args']
                inference_deduplication = bool(
                    controller_args.get('inferenceDeduplication', False)
                    if hasattr(controller_args, 'get')
                    else getattr(controller_args, 'inferenceDeduplication', False)
                )
                inference_cache_size = int(
                    controller_args.get('inferenceCacheSize', 4096)
                    if hasattr(controller_args, 'get')
                    else getattr(controller_args, 'inferenceCacheSize', 4096)
                )
                if inference_deduplication:
                    policies, values, stats = predict_batch_deduplicated(
                        nnet,
                        boards,
                        cache=inference_caches[controller],
                        max_cache_entries=inference_cache_size,
                    )
                    self.inference_stats['batches'] += 1
                    for key in ('requested', 'executed', 'reused'):
                        self.inference_stats[key] += int(stats[key])
                elif hasattr(nnet, 'predict_batch'):
                    policies, values = nnet.predict_batch(boards)
                else:
                    predictions = [nnet.predict(board) for board in boards]
                    policies, values = zip(*predictions)

                for (mcts, leaf), (start, end) in zip(leaves, evaluation_ranges):
                    mcts.complete_search(
                        leaf,
                        np.asarray(policies)[start:end],
                        np.asarray(values)[start:end],
                    )

        actions = []
        for game_state in active:
            board = game_state['canonicalBoard']
            tactical = game_state.get('tactical')
            if tactical is not None and tactical['policy'] is not None:
                actions.append(self._selectLegalAction(
                    board,
                    tactical['policy'],
                    sample=False,
                    rng=game_state['rng'],
                ))
                continue
            is_placement = self._isPlacement(board)
            sample = is_placement and self.placement_temperature > 0
            temperature = self.placement_temperature if sample else 1.0
            _, mcts, _, _ = self._controller(game_state)
            probs = mcts.getActionProbFromTree(board, temp=temperature)
            actions.append(self._selectLegalAction(
                board,
                probs,
                sample=sample,
                rng=game_state['rng'],
            ))
        return actions

    def inferenceDiagnostics(self):
        stats = dict(self.inference_stats)
        stats['reuse_rate'] = (
            float(stats['reused'] / stats['requested'])
            if stats['requested'] else None
        )
        return stats

    def _selectLegalAction(self, canonicalBoard, probs, sample=False, rng=None):
        probs = np.array(probs)
        valids = self.game.getValidMoves(canonicalBoard, 1)
        masked_probs = probs * valids
        if masked_probs.sum() > 0:
            if sample:
                masked_probs = masked_probs / masked_probs.sum()
                generator = rng if rng is not None else np.random
                return int(generator.choice(len(masked_probs), p=masked_probs))
            return int(np.argmax(masked_probs))
        return int(np.flatnonzero(valids)[0])
