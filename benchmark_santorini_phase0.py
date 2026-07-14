import argparse
import json
import random
import time

import numpy as np
import torch
import torch.optim as optim

from MCTS import MCTS
from santorini.SantoriniGame import SantoriniGame
from santorini.pytorch.SantoriniNNet import SantoriniNNet
from utils import dotdict


DEFAULT_ARCHITECTURES = {
    "v2": (5, 64),
    "v3": (8, 96),
}


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_architectures(value):
    architectures = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item in DEFAULT_ARCHITECTURES:
            blocks, channels = DEFAULT_ARCHITECTURES[item]
            architectures.append((item, blocks, channels))
            continue
        try:
            blocks_text, channels_text = item.lower().split("x", 1)
            blocks = int(blocks_text)
            channels = int(channels_text)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Architecture must be a known name or BLOCKSxCHANNELS, got {}".format(item)
            )
        architectures.append(("{}x{}".format(blocks, channels), blocks, channels))
    return architectures


def sync_cuda(use_cuda):
    if use_cuda:
        torch.cuda.synchronize()


def memory_mb(use_cuda):
    if not use_cuda:
        return None
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def reset_peak_memory(use_cuda):
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()


def make_nnet_args(blocks, channels, use_cuda, policy_channels=64, dropout=0.2):
    return dotdict({
        "lr": 0.001,
        "dropout": dropout,
        "epochs": 1,
        "batch_size": 64,
        "cuda": use_cuda,
        "input_channels": 6,
        "num_channels": channels,
        "num_residual_blocks": blocks,
        "value_hidden_size": 128,
        "quiet": True,
        "policy_channels": policy_channels,
    })


def encode_board(board):
    pieces = board[0]
    heights = board[1]
    encoded = np.zeros((6, pieces.shape[0], pieces.shape[1]), dtype=np.float32)

    encoded[0] = pieces > 0
    encoded[1] = pieces < 0
    encoded[2] = heights == 1
    encoded[3] = heights == 2
    encoded[4] = heights == 3
    encoded[5] = heights >= 4

    return encoded


def encode_boards(boards):
    return np.array([encode_board(board) for board in boards], dtype=np.float32)


class BenchmarkNNet:
    def __init__(self, game, name, blocks, channels, use_cuda):
        self.game = game
        self.name = name
        self.blocks = blocks
        self.channels = channels
        self.use_cuda = use_cuda
        self.args = make_nnet_args(
            blocks,
            channels,
            use_cuda,
            policy_channels=65 if name == 'v3' else 64,
        )
        self.nnet = SantoriniNNet(game, self.args)
        if use_cuda:
            self.nnet.cuda()

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.nnet.parameters())

    def predict(self, board):
        policies, values = self.predict_batch([board])
        return policies[0], float(values[0])

    def predict_batch(self, boards):
        encoded = torch.from_numpy(encode_boards(boards))
        if self.use_cuda:
            encoded = encoded.contiguous().cuda()

        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(encoded)

        return torch.exp(pi).data.cpu().numpy(), v.view(-1).data.cpu().numpy()


def random_legal_action(game, board, player):
    valids = game.getValidMoves(board, player)
    legal_actions = np.flatnonzero(valids)
    if len(legal_actions) == 0:
        return None
    return int(np.random.choice(legal_actions))


def sample_canonical_boards(game, count, max_random_plies):
    boards = []
    attempts = 0
    while len(boards) < count:
        attempts += 1
        board = game.getInitBoard()
        player = 1
        plies = np.random.randint(0, max_random_plies + 1)

        for _ in range(plies):
            if game.getGameEnded(board, player) != 0:
                break
            action = random_legal_action(game, board, player)
            if action is None:
                break
            board, player = game.getNextState(board, player, action)

        if game.getGameEnded(board, player) == 0:
            boards.append(game.getCanonicalForm(board, player))

        if attempts > count * 20:
            raise RuntimeError("Could not sample enough non-terminal Santorini boards.")

    return boards


def benchmark_inference(model, boards, batch_sizes, warmup_batches, timed_batches):
    results = []
    max_batch = max(batch_sizes)
    if len(boards) < max_batch:
        raise ValueError("Need at least {} boards for inference benchmark.".format(max_batch))

    for batch_size in batch_sizes:
        batch = boards[:batch_size]
        for _ in range(warmup_batches):
            model.predict_batch(batch)
        sync_cuda(model.use_cuda)

        reset_peak_memory(model.use_cuda)
        start = time.perf_counter()
        for _ in range(timed_batches):
            model.predict_batch(batch)
        sync_cuda(model.use_cuda)
        elapsed = time.perf_counter() - start

        results.append({
            "mode": "inference",
            "architecture": model.name,
            "blocks": model.blocks,
            "channels": model.channels,
            "batch_size": batch_size,
            "batches": timed_batches,
            "seconds": elapsed,
            "ms_per_batch": elapsed * 1000 / timed_batches,
            "positions_per_second": batch_size * timed_batches / elapsed,
            "peak_cuda_memory_mb": memory_mb(model.use_cuda),
        })

    return results


def benchmark_mcts(model, game, boards, batch_sizes, sims, repeats, cpuct):
    results = []
    if len(boards) < max(batch_sizes):
        raise ValueError("Need at least {} boards for MCTS benchmark.".format(max(batch_sizes)))

    for batch_size in batch_sizes:
        roots = boards[:batch_size]
        mcts_args = dotdict({"numMCTSSims": sims, "cpuct": cpuct})

        reset_peak_memory(model.use_cuda)
        sync_cuda(model.use_cuda)
        start = time.perf_counter()

        for _ in range(repeats):
            mcts_by_root = [MCTS(game, model, mcts_args) for _ in roots]
            for _ in range(sims):
                pending = []
                for mcts, board in zip(mcts_by_root, roots):
                    leaf = mcts.select_leaf(board)
                    if leaf["needs_eval"]:
                        pending.append((mcts, leaf))
                    else:
                        mcts.complete_search(leaf)

                if pending:
                    leaf_boards = [leaf["board"] for _, leaf in pending]
                    policies, values = model.predict_batch(leaf_boards)
                    for (mcts, leaf), policy, value in zip(pending, policies, values):
                        mcts.complete_search(leaf, policy, float(value))

        sync_cuda(model.use_cuda)
        elapsed = time.perf_counter() - start
        total_sims = batch_size * sims * repeats

        results.append({
            "mode": "mcts",
            "architecture": model.name,
            "blocks": model.blocks,
            "channels": model.channels,
            "batch_size": batch_size,
            "sims_per_root": sims,
            "repeats": repeats,
            "total_simulations": total_sims,
            "seconds": elapsed,
            "simulations_per_second": total_sims / elapsed,
            "ms_per_simulation": elapsed * 1000 / total_sims,
            "peak_cuda_memory_mb": memory_mb(model.use_cuda),
        })

    return results


def build_training_tensors(game, boards, use_cuda):
    encoded = torch.from_numpy(encode_boards(boards))
    policies = []
    for board in boards:
        valids = game.getValidMoves(board, 1).astype(np.float32)
        policy_sum = float(np.sum(valids))
        if policy_sum == 0:
            raise ValueError("Training benchmark sampled a board with no legal moves.")
        policies.append(valids / policy_sum)
    target_pis = torch.from_numpy(np.array(policies, dtype=np.float32))
    target_vs = torch.from_numpy(np.random.uniform(-1, 1, size=len(boards)).astype(np.float32))

    if use_cuda:
        encoded = encoded.contiguous().cuda()
        target_pis = target_pis.contiguous().cuda()
        target_vs = target_vs.contiguous().cuda()

    return encoded, target_pis, target_vs


def loss_pi(targets, outputs):
    return -torch.sum(targets * outputs) / targets.size()[0]


def loss_v(targets, outputs):
    return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]


def benchmark_training(model, game, boards, batch_sizes, warmup_steps, timed_steps):
    results = []
    if len(boards) < max(batch_sizes):
        raise ValueError("Need at least {} boards for training benchmark.".format(max(batch_sizes)))

    encoded, target_pis, target_vs = build_training_tensors(game, boards, model.use_cuda)

    for batch_size in batch_sizes:
        optimizer = optim.Adam(model.nnet.parameters(), lr=model.args.lr)

        def training_step(offset):
            sample_ids = (torch.arange(batch_size) + offset) % encoded.size(0)
            if model.use_cuda:
                sample_ids = sample_ids.cuda()
            batch_boards = encoded.index_select(0, sample_ids)
            batch_target_pis = target_pis.index_select(0, sample_ids)
            batch_target_vs = target_vs.index_select(0, sample_ids)

            out_pi, out_v = model.nnet(batch_boards)
            total_loss = loss_pi(batch_target_pis, out_pi) + loss_v(batch_target_vs, out_v)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            return float(total_loss.item())

        model.nnet.train()
        for step in range(warmup_steps):
            training_step(step * batch_size)
        sync_cuda(model.use_cuda)

        reset_peak_memory(model.use_cuda)
        start = time.perf_counter()
        last_loss = None
        for step in range(timed_steps):
            last_loss = training_step(step * batch_size)
        sync_cuda(model.use_cuda)
        elapsed = time.perf_counter() - start

        results.append({
            "mode": "training",
            "architecture": model.name,
            "blocks": model.blocks,
            "channels": model.channels,
            "batch_size": batch_size,
            "steps": timed_steps,
            "seconds": elapsed,
            "ms_per_step": elapsed * 1000 / timed_steps,
            "examples_per_second": batch_size * timed_steps / elapsed,
            "last_loss": last_loss,
            "peak_cuda_memory_mb": memory_mb(model.use_cuda),
        })

    return results


def print_result(result):
    memory = result["peak_cuda_memory_mb"]
    memory_text = " n/a" if memory is None else "{:.1f}".format(memory)
    if result["mode"] == "inference":
        print(
            "{architecture:>4} inference batch={batch_size:<4} "
            "{positions_per_second:>10.1f} pos/s {ms_per_batch:>8.3f} ms/batch "
            "cuda_mb={memory}".format(memory=memory_text, **result)
        )
    elif result["mode"] == "mcts":
        print(
            "{architecture:>4} mcts      batch={batch_size:<4} "
            "{simulations_per_second:>10.1f} sims/s {ms_per_simulation:>8.3f} ms/sim "
            "cuda_mb={memory}".format(memory=memory_text, **result)
        )
    elif result["mode"] == "training":
        print(
            "{architecture:>4} training  batch={batch_size:<4} "
            "{examples_per_second:>10.1f} ex/s {ms_per_step:>8.3f} ms/step "
            "cuda_mb={memory}".format(memory=memory_text, **result)
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 0 throughput benchmarks for Santorini V2/V3 architecture experiments."
    )
    parser.add_argument("--architectures", default="v2,v3", type=parse_architectures)
    parser.add_argument("--modes", default="inference,mcts,training")
    parser.add_argument("--inference-batch-sizes", default="1,8,16,32,64,128,256", type=parse_int_list)
    parser.add_argument("--mcts-batch-sizes", default="1,8,16,32,64", type=parse_int_list)
    parser.add_argument("--training-batch-sizes", default="64,128,256,512", type=parse_int_list)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--timed-batches", type=int, default=30)
    parser.add_argument("--mcts-sims", type=int, default=32)
    parser.add_argument("--mcts-repeats", type=int, default=3)
    parser.add_argument("--cpuct", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--timed-steps", type=int, default=20)
    parser.add_argument("--sample-boards", type=int, default=1024)
    parser.add_argument("--max-random-plies", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--json-out", help="Optional path for machine-readable benchmark results.")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())

    modes = {mode.strip() for mode in args.modes.split(",") if mode.strip()}
    unknown_modes = modes - {"inference", "mcts", "training"}
    if unknown_modes:
        raise ValueError("Unknown benchmark modes: {}".format(", ".join(sorted(unknown_modes))))

    required_batch_sizes = []
    if "inference" in modes:
        required_batch_sizes.extend(args.inference_batch_sizes)
    if "mcts" in modes:
        required_batch_sizes.extend(args.mcts_batch_sizes)
    if "training" in modes:
        required_batch_sizes.extend(args.training_batch_sizes)
    board_count = max(args.sample_boards, max(required_batch_sizes or [1]))

    print("Device: {}".format("cuda" if use_cuda else "cpu"))
    if use_cuda:
        print("GPU: {}".format(torch.cuda.get_device_name(0)))
    print("Sampling {} canonical boards per architecture...".format(board_count))

    all_results = []
    for name, blocks, channels in args.architectures:
        game = SantoriniGame(
            5,
            true_random_placement=name != 'v3',
            sequential_placement=name == 'v3',
        )
        boards = sample_canonical_boards(game, board_count, args.max_random_plies)
        model = BenchmarkNNet(game, name, blocks, channels, use_cuda)
        print(
            "\nArchitecture {}: {} residual blocks, {} channels, {:,} parameters".format(
                name,
                blocks,
                channels,
                model.parameter_count(),
            )
        )

        if "inference" in modes:
            for result in benchmark_inference(
                model,
                boards,
                args.inference_batch_sizes,
                args.warmup_batches,
                args.timed_batches,
            ):
                print_result(result)
                all_results.append(result)

        if "mcts" in modes:
            for result in benchmark_mcts(
                model,
                game,
                boards,
                args.mcts_batch_sizes,
                args.mcts_sims,
                args.mcts_repeats,
                args.cpuct,
            ):
                print_result(result)
                all_results.append(result)

        if "training" in modes:
            for result in benchmark_training(
                model,
                game,
                boards,
                args.training_batch_sizes,
                args.warmup_steps,
                args.timed_steps,
            ):
                print_result(result)
                all_results.append(result)

    if args.json_out:
        payload = {
            "device": "cuda" if use_cuda else "cpu",
            "gpu": torch.cuda.get_device_name(0) if use_cuda else None,
            "seed": args.seed,
            "sample_boards": board_count,
            "max_random_plies": args.max_random_plies,
            "results": all_results,
        }
        with open(args.json_out, "w") as output_file:
            json.dump(payload, output_file, indent=2, sort_keys=True)
        print("\nWrote benchmark JSON: {}".format(args.json_out))


if __name__ == "__main__":
    main()
