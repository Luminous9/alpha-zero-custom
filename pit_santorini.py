import argparse
import json
import os

import numpy as np

import Arena
from MCTS import MCTS
from santorini.SantoriniGame import SantoriniGame
from santorini.SantoriniPlayers import GreedySantoriniPlayer, RandomPlayer
from santorini.pytorch.LegacyNNet import LegacyNNetWrapper
from santorini.pytorch.NNet import NNetWrapper
from utils import dotdict


def build_nnet(game, architecture):
    if architecture == 'v2':
        return NNetWrapper(game)
    if architecture == 'v1':
        return LegacyNNetWrapper(game)
    raise ValueError("Unknown architecture: {}".format(architecture))


class NeuralMCTSPlayer:
    def __init__(self, game, checkpoint_folder, checkpoint_file, sims, architecture='v2'):
        self.game = game
        self.nnet = build_nnet(game, architecture)
        self.nnet.load_checkpoint(checkpoint_folder, checkpoint_file)
        self.mcts_args = dotdict({'numMCTSSims': sims, 'cpuct': 1.0})
        self.mcts = None

    def startGame(self):
        self.mcts = MCTS(self.game, self.nnet, self.mcts_args)

    def play(self, board):
        if self.mcts is None:
            self.startGame()
        return select_legal_action(self.game, board, self.mcts.getActionProb(board, temp=0))

    __call__ = play


def select_legal_action(game, board, probs):
    probs = np.array(probs)
    valids = game.getValidMoves(board, 1)
    masked_probs = probs * valids
    if masked_probs.sum() > 0:
        return int(np.argmax(masked_probs))
    return int(np.flatnonzero(valids)[0])


def build_baseline(game, name):
    if name == 'random':
        return RandomPlayer(game).play
    if name == 'greedy':
        return GreedySantoriniPlayer(game).play
    raise ValueError("Unknown baseline: {}".format(name))


def main():
    parser = argparse.ArgumentParser(description='Pit a Santorini neural MCTS player against a baseline.')
    parser.add_argument('--baseline', choices=['random', 'greedy'], default='random')
    parser.add_argument('--games', type=int, default=4)
    parser.add_argument('--sims', type=int, default=25)
    parser.add_argument('--checkpoint-folder', default='./temp/santorini_quick/')
    parser.add_argument('--checkpoint-file', default='best.pth.tar')
    parser.add_argument('--architecture', choices=['v1', 'v2'], default='v2')
    parser.add_argument('--opponent-checkpoint-folder')
    parser.add_argument('--opponent-checkpoint-file', default='best.pth.tar')
    parser.add_argument('--opponent-architecture', choices=['v1', 'v2'], default='v2')
    parser.add_argument('--opponent-sims', type=int)
    parser.add_argument('--fresh', action='store_true', help='Use an untrained network even if a checkpoint exists.')
    parser.add_argument('--json-out', help='Optional path to write evaluation results as JSON.')
    args = parser.parse_args()

    game = SantoriniGame(5, true_random_placement=True)

    checkpoint_path = os.path.join(args.checkpoint_folder, args.checkpoint_file)
    if args.opponent_checkpoint_folder:
        opponent_checkpoint_path = os.path.join(args.opponent_checkpoint_folder, args.opponent_checkpoint_file)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError("No model in path {}".format(checkpoint_path))
        if not os.path.exists(opponent_checkpoint_path):
            raise FileNotFoundError("No opponent model in path {}".format(opponent_checkpoint_path))

        print("Loaded checkpoint: {}".format(checkpoint_path))
        print("Loaded opponent checkpoint: {}".format(opponent_checkpoint_path))
        print("Architecture: {}; opponent architecture: {}".format(
            args.architecture,
            args.opponent_architecture,
        ))
        nnet_player_obj = NeuralMCTSPlayer(
            game,
            args.checkpoint_folder,
            args.checkpoint_file,
            args.sims,
            architecture=args.architecture,
        )
        opponent_player_obj = NeuralMCTSPlayer(
            game,
            args.opponent_checkpoint_folder,
            args.opponent_checkpoint_file,
            args.opponent_sims or args.sims,
            architecture=args.opponent_architecture,
        )
        player1 = nnet_player_obj
        player2 = opponent_player_obj
        opponent_name = 'opponent'
    else:
        nnet = build_nnet(game, args.architecture)
        if not args.fresh and os.path.exists(checkpoint_path):
            nnet.load_checkpoint(args.checkpoint_folder, args.checkpoint_file)
            print("Loaded checkpoint: {} ({})".format(checkpoint_path, args.architecture))
        else:
            print("Using fresh untrained network.")

        mcts_args = dotdict({'numMCTSSims': args.sims, 'cpuct': 1.0})
        mcts = MCTS(game, nnet, mcts_args)
        player1 = lambda x: select_legal_action(game, x, mcts.getActionProb(x, temp=0))
        player2 = build_baseline(game, args.baseline)
        opponent_name = args.baseline

    arena = Arena.Arena(player1, player2, game, display=SantoriniGame.display)
    nnet_wins, opponent_wins, draws = arena.playGames(args.games, verbose=False)

    print("Neural MCTS wins: {}".format(nnet_wins))
    print("{} wins: {}".format(opponent_name.title(), opponent_wins))
    print("Draws: {}".format(draws))

    if args.json_out:
        result = {
            'baseline': args.baseline,
            'games': args.games,
            'sims': args.sims,
            'checkpoint_folder': args.checkpoint_folder,
            'checkpoint_file': args.checkpoint_file,
            'architecture': args.architecture,
            'opponent_checkpoint_folder': args.opponent_checkpoint_folder,
            'opponent_checkpoint_file': args.opponent_checkpoint_file,
            'opponent_architecture': args.opponent_architecture,
            'opponent_sims': args.opponent_sims or args.sims,
            'fresh': args.fresh,
            'neural_mcts_wins': int(nnet_wins),
            'baseline_wins': int(opponent_wins),
            'draws': int(draws),
        }
        json_dir = os.path.dirname(args.json_out)
        if json_dir:
            os.makedirs(json_dir, exist_ok=True)
        with open(args.json_out, 'w') as f:
            json.dump(result, f, indent=2, sort_keys=True)
        print("Wrote JSON results: {}".format(args.json_out))


if __name__ == "__main__":
    main()
