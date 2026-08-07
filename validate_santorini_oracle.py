import argparse
import json

import numpy as np

from santorini.SantoriniGame import SantoriniGame
from santorini.SantoriniOpeningBook import SantoriniRandomOpeningSampler
from santorini.SantoriniOracle import SantoriniOracleProcess, compare_legal_successors


def parse_args():
    parser = argparse.ArgumentParser(
        description="Differentially validate Santorini legal successors against santorini-ai."
    )
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--max-positions", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--oracle-binary")
    parser.add_argument("--json-out")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.games < 1 or args.max_positions < 1:
        raise ValueError("--games and --max-positions must be positive.")

    game = SantoriniGame(5, sequential_placement=True)
    rng = np.random.RandomState(args.seed)
    sampler = SantoriniRandomOpeningSampler(random_orientation=True, rng=rng)
    compared = 0
    completed_games = 0

    with SantoriniOracleProcess(args.oracle_binary) as oracle:
        for game_index in range(args.games):
            board = sampler.sample_self_play_board()
            cur_player = 1
            ply = 0
            while game.getGameEnded(board, cur_player) == 0 and compared < args.max_positions:
                canonical = game.getCanonicalForm(board, cur_player)
                result = compare_legal_successors(game, canonical, oracle)
                if not result["matches"]:
                    raise AssertionError(
                        "Rule mismatch in game {} ply {}: ours={} theirs={}, "
                        "only_ours={} only_theirs={}".format(
                            game_index,
                            ply,
                            result["ours_count"],
                            result["theirs_count"],
                            len(result["only_ours"]),
                            len(result["only_theirs"]),
                        )
                    )
                valids = np.flatnonzero(game.getValidMoves(canonical, 1))
                action = int(rng.choice(valids))
                board, cur_player = game.getNextState(board, cur_player, action)
                compared += 1
                ply += 1
            completed_games += 1
            if compared >= args.max_positions:
                break

    summary = {
        "games": completed_games,
        "positions": compared,
        "seed": args.seed,
        "successor_sets_match": True,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_out:
        with open(args.json_out, "w") as output_file:
            json.dump(summary, output_file, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
