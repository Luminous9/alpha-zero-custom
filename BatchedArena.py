import numpy as np
from tqdm import tqdm

from MCTS import MCTS


class BatchedMCTSArena:
    """
    Arena evaluator for neural-MCTS players. It keeps independent MCTS trees per
    active game/player while batching neural-network leaf evaluations.
    """

    def __init__(self, game, player1_nnet, player2_nnet, args, batch_size=1, quiet=False, opening_boards=None):
        self.game = game
        self.nnets = {
            1: player1_nnet,
            -1: player2_nnet,
        }
        self.args = args
        self.batch_size = max(1, int(batch_size))
        self.quiet = quiet
        self.opening_boards = opening_boards

    def playGames(self, num):
        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0

        first_results = self._playGamesForSides(num, {1: 1, -1: -1}, "BatchedArena.playGames (1)")
        second_results = self._playGamesForSides(num, {1: -1, -1: 1}, "BatchedArena.playGames (2)")

        for results in (first_results, second_results):
            oneWon += results[0]
            twoWon += results[1]
            draws += results[2]

        return oneWon, twoWon, draws

    def _playGamesForSides(self, num, side_to_player, desc):
        oneWon = 0
        twoWon = 0
        draws = 0
        launched = 0
        completed = 0
        active = []
        progress = tqdm(total=num, desc=desc, disable=self.quiet)

        try:
            while completed < num:
                while launched < num and len(active) < self.batch_size:
                    opening_board = self.opening_boards[launched] if self.opening_boards is not None else None
                    active.append(self._newGame(side_to_player, opening_board=opening_board))
                    launched += 1

                for game_state in active:
                    game_state['canonicalBoard'] = self.game.getCanonicalForm(
                        game_state['board'],
                        game_state['curPlayer'],
                    )

                actions = self._getBatchedActions(active)
                still_active = []

                for game_state, action in zip(active, actions):
                    game_state['board'], game_state['curPlayer'] = self.game.getNextState(
                        game_state['board'],
                        game_state['curPlayer'],
                        action,
                    )
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

    def _newGame(self, side_to_player, opening_board=None):
        return {
            'board': opening_board.copy() if opening_board is not None else self.game.getInitBoard(),
            'curPlayer': 1,
            'side_to_player': side_to_player,
            'mcts_by_player': {
                1: MCTS(self.game, self.nnets[1], self.args),
                -1: MCTS(self.game, self.nnets[-1], self.args),
            },
        }

    def _getBatchedActions(self, active):
        for _ in range(self.args.numMCTSSims):
            pending_by_player = {1: [], -1: []}

            for game_state in active:
                player = game_state['side_to_player'][game_state['curPlayer']]
                mcts = game_state['mcts_by_player'][player]
                leaf = mcts.select_leaf(game_state['canonicalBoard'])
                if leaf['needs_eval']:
                    pending_by_player[player].append((mcts, leaf))
                else:
                    mcts.complete_search(leaf)

            for player, pending in pending_by_player.items():
                if not pending:
                    continue

                boards = [leaf['board'] for _, leaf in pending]
                nnet = self.nnets[player]
                if hasattr(nnet, 'predict_batch'):
                    policies, values = nnet.predict_batch(boards)
                else:
                    predictions = [nnet.predict(board) for board in boards]
                    policies, values = zip(*predictions)

                for (mcts, leaf), policy, value in zip(pending, policies, values):
                    mcts.complete_search(leaf, policy, float(value))

        return [
            self._selectLegalAction(
                game_state['canonicalBoard'],
                game_state['mcts_by_player'][
                    game_state['side_to_player'][game_state['curPlayer']]
                ].getActionProbFromTree(game_state['canonicalBoard'], temp=0),
            )
            for game_state in active
        ]

    def _selectLegalAction(self, canonicalBoard, probs):
        probs = np.array(probs)
        valids = self.game.getValidMoves(canonicalBoard, 1)
        masked_probs = probs * valids
        if masked_probs.sum() > 0:
            return int(np.argmax(masked_probs))
        return int(np.flatnonzero(valids)[0])
