import argparse
import concurrent.futures
import os
import pickle
import random
import time

import numpy as np
import torch

from MCTS import MCTS
from santorini.pytorch.NNet import args as nnet_args
from santorini.pytorch.NNet import build_nnet
from santorini.SantoriniGame import SantoriniGame
from utils import dotdict

DEFAULT_CHECKPOINT_FOLDER = './temp/santorini_bootstrap_result'
DEFAULT_EXAMPLES = './temp/santorini_kaggle_training6/merged_20.examples'

_worker_game = None
_worker_nnet = None
_worker_search_args = None
_worker_seed = None


def log(message):
    print(message, flush=True)


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return '{}h {:02d}m'.format(hours, minutes)
    if minutes:
        return '{}m {:02d}s'.format(minutes, seconds)
    return '{}s'.format(seconds)


def flatten_history(path):
    with open(path, 'rb') as examples_file:
        history = pickle.Unpickler(examples_file).load()
    return [example for window in history for example in window]


def stage_for_board(board):
    standard_ply = int(np.sum(board[1]))
    if standard_ply <= 12:
        return 0
    if standard_ply <= 25:
        return 1
    return 2


def select_candidates(examples, quotas, seed):
    rng = random.Random(seed)
    buckets = {0: [], 1: [], 2: []}
    seen = set()
    for board, _, _ in examples:
        key = np.asarray(board).tobytes()
        if key in seen:
            continue
        seen.add(key)
        buckets[stage_for_board(board)].append(np.asarray(board, dtype=int))
    selected = []
    for stage, quota in enumerate(quotas):
        rng.shuffle(buckets[stage])
        if len(buckets[stage]) < quota:
            raise ValueError('Stage {} contains only {} distinct positions; need {}.'.format(stage, len(buckets[stage]), quota))
        selected.extend((board, stage) for board in buckets[stage][:quota])
    rng.shuffle(selected)
    return selected


def initialize_worker(checkpoint_folder, checkpoint_file, mcts_sims, seed, torch_threads):
    global _worker_game, _worker_nnet, _worker_search_args, _worker_seed
    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _worker_seed = seed
    nnet_args.cuda = False
    _worker_game = SantoriniGame(5)
    _worker_nnet = build_nnet(_worker_game, 'v2')
    _worker_nnet.load_checkpoint(checkpoint_folder, checkpoint_file)
    _worker_search_args = dotdict({'numMCTSSims': mcts_sims, 'cpuct': 1.0})


def label_candidate(task):
    index, board, stage = task
    task_seed = _worker_seed + index
    random.seed(task_seed)
    np.random.seed(task_seed % (2 ** 32))
    torch.manual_seed(task_seed)
    mcts = MCTS(_worker_game, _worker_nnet, _worker_search_args)
    policy = np.asarray(mcts.getActionProb(board, temp=1), dtype=np.float32)
    state_key = _worker_game.stringRepresentation(board)
    counts = mcts.Nsas.get(state_key, np.zeros(_worker_game.getActionSize(), dtype=np.int32))
    q_values = mcts.Qs.get(state_key, np.zeros(_worker_game.getActionSize(), dtype=np.float32))
    root_value = float(np.sum(counts * q_values) / max(1, np.sum(counts)))
    return index, board, policy, root_value, stage


def report_progress(completed, total, started_at):
    elapsed = time.monotonic() - started_at
    rate = completed / elapsed if elapsed else 0.0
    remaining = (total - completed) / rate if rate else 0.0
    log('Labeled {}/{} positions | elapsed {} | ETA {} | {:.2f} positions/min'.format(
        completed,
        total,
        format_duration(elapsed),
        format_duration(remaining),
        rate * 60.0,
    ))


def main():
    parser = argparse.ArgumentParser(description='Build the fixed V2 high-budget MCTS reference suite.')
    parser.add_argument('--checkpoint-folder', default=DEFAULT_CHECKPOINT_FOLDER)
    parser.add_argument('--checkpoint-file', default='best.pth.tar')
    parser.add_argument('--examples-file', default=DEFAULT_EXAMPLES)
    parser.add_argument('--output', default='./santorini/reference_suites/v2_reference_500.npz')
    parser.add_argument('--early', type=int, default=150)
    parser.add_argument('--mid', type=int, default=200)
    parser.add_argument('--late', type=int, default=150)
    parser.add_argument('--mcts-sims', type=int, default=1600)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--workers', type=int, default=min(4, os.cpu_count() or 1),
                        help='Number of CPU worker processes (default: up to 4).')
    parser.add_argument('--threads-per-worker', type=int, default=1,
                        help='PyTorch CPU threads in each worker (default: 1).')
    parser.add_argument('--cpu', action='store_true',
                        help='Retained for compatibility; reference labeling always uses CPU workers.')
    args = parser.parse_args()

    if args.workers < 1:
        parser.error('--workers must be at least 1')
    if args.threads_per_worker < 1:
        parser.error('--threads-per-worker must be at least 1')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    nnet_args.cuda = False

    total = args.early + args.mid + args.late
    log('Loading replay history from {} ...'.format(args.examples_file))
    load_started = time.monotonic()
    examples = flatten_history(args.examples_file)
    log('Loaded {:,} examples in {}. Selecting {} distinct positions ...'.format(
        len(examples), format_duration(time.monotonic() - load_started), total))
    candidates = select_candidates(examples, (args.early, args.mid, args.late), args.seed)
    del examples
    log('Selected {} positions (early {}, mid {}, late {}).'.format(
        len(candidates), args.early, args.mid, args.late))
    worker_label = 'worker' if args.workers == 1 else 'workers'
    log('Labeling with {} MCTS simulations using {} CPU {} × {} PyTorch thread(s) ...'.format(
        args.mcts_sims, args.workers, worker_label, args.threads_per_worker))

    tasks = [(index, board, stage) for index, (board, stage) in enumerate(candidates)]
    results = [None] * len(tasks)
    label_started = time.monotonic()
    if args.workers == 1:
        initialize_worker(args.checkpoint_folder, args.checkpoint_file, args.mcts_sims,
                          args.seed, args.threads_per_worker)
        for completed, task in enumerate(tasks, start=1):
            result = label_candidate(task)
            results[result[0]] = result
            report_progress(completed, len(tasks), label_started)
    else:
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=initialize_worker,
                initargs=(args.checkpoint_folder, args.checkpoint_file, args.mcts_sims,
                          args.seed, args.threads_per_worker)) as executor:
            futures = [executor.submit(label_candidate, task) for task in tasks]
            for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                result = future.result()
                results[result[0]] = result
                report_progress(completed, len(tasks), label_started)

    _, boards, policies, values, stages = zip(*results)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, 'wb') as output_file:
        np.savez_compressed(
            output_file,
            boards=np.asarray(boards, dtype=np.int8),
            policies=np.asarray(policies, dtype=np.float32),
            values=np.asarray(values, dtype=np.float32),
            stages=np.asarray(stages, dtype=np.int8),
            checkpoint=np.asarray([os.path.join(args.checkpoint_folder, args.checkpoint_file)]),
            mcts_sims=np.asarray([args.mcts_sims], dtype=np.int32),
        )
    log('Saved {} reference positions to {}'.format(len(boards), args.output))


if __name__ == '__main__':
    main()
