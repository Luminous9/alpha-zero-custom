import argparse
import json
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.optim as optim

from santorini.SantoriniGame import SantoriniGame
from santorini.pytorch.NNet import args as nnet_args
from santorini.pytorch.NNet import build_nnet


DEFAULT_EXAMPLES_FILE = "temp/santorini_kaggle_training6_v2/latest.examples"
DEFAULT_OUTPUT_FOLDER = "temp/santorini_v3_bootstrap"


def flatten_examples_history(examples_history):
    examples = []
    for history in examples_history:
        examples.extend(history)
    return examples


def load_examples(path, max_examples=None):
    with open(path, "rb") as examples_file:
        examples_history = pickle.Unpickler(examples_file).load()
    examples = flatten_examples_history(examples_history)
    if max_examples is not None:
        examples = examples[:max_examples]
    return examples, [len(history) for history in examples_history]


def split_indices(example_count, validation_fraction, seed):
    if not 0 < validation_fraction < 1:
        raise ValueError("--validation-fraction must be between 0 and 1.")

    rng = np.random.RandomState(seed)
    indices = rng.permutation(example_count)
    validation_count = max(1, int(round(example_count * validation_fraction)))
    validation_count = min(validation_count, example_count - 1)

    validation_indices = indices[:validation_count]
    train_indices = indices[validation_count:]
    return train_indices, validation_indices


def encode_examples(nnet, examples):
    boards, pis, vs = list(zip(*examples))
    return (
        nnet.encode_boards(boards),
        np.array(pis, dtype=np.float32),
        np.array(vs, dtype=np.float32),
    )


def to_tensor(array, use_cuda):
    tensor = torch.from_numpy(array)
    if use_cuda:
        tensor = tensor.contiguous().cuda()
    return tensor


def policy_loss(targets, outputs):
    return -torch.sum(targets * outputs) / targets.size()[0]


def value_loss(targets, outputs):
    return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]


def evaluate(nnet, encoded_boards, target_pis, target_vs, indices, batch_size, use_cuda):
    nnet.nnet.eval()
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            boards = to_tensor(encoded_boards[batch_indices], use_cuda)
            batch_target_pis = to_tensor(target_pis[batch_indices], use_cuda)
            batch_target_vs = to_tensor(target_vs[batch_indices], use_cuda)

            out_pi, out_v = nnet.nnet(boards)
            batch_policy_loss = policy_loss(batch_target_pis, out_pi)
            batch_value_loss = value_loss(batch_target_vs, out_v)
            batch_count = len(batch_indices)

            total_policy_loss += batch_policy_loss.item() * batch_count
            total_value_loss += batch_value_loss.item() * batch_count
            total_count += batch_count

    average_policy_loss = total_policy_loss / total_count
    average_value_loss = total_value_loss / total_count
    return {
        "policy_loss": average_policy_loss,
        "value_loss": average_value_loss,
        "total_loss": average_policy_loss + average_value_loss,
    }


def train_epoch(nnet, optimizer, encoded_boards, target_pis, target_vs, train_indices, batch_size, use_cuda, rng):
    nnet.nnet.train()
    shuffled = np.array(train_indices, copy=True)
    rng.shuffle(shuffled)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_count = 0

    for start in range(0, len(shuffled), batch_size):
        batch_indices = shuffled[start:start + batch_size]
        boards = to_tensor(encoded_boards[batch_indices], use_cuda)
        batch_target_pis = to_tensor(target_pis[batch_indices], use_cuda)
        batch_target_vs = to_tensor(target_vs[batch_indices], use_cuda)

        out_pi, out_v = nnet.nnet(boards)
        batch_policy_loss = policy_loss(batch_target_pis, out_pi)
        batch_value_loss = value_loss(batch_target_vs, out_v)
        total_loss = batch_policy_loss + batch_value_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        batch_count = len(batch_indices)
        total_policy_loss += batch_policy_loss.item() * batch_count
        total_value_loss += batch_value_loss.item() * batch_count
        total_count += batch_count

    average_policy_loss = total_policy_loss / total_count
    average_value_loss = total_value_loss / total_count
    return {
        "policy_loss": average_policy_loss,
        "value_loss": average_value_loss,
        "total_loss": average_policy_loss + average_value_loss,
    }


def save_metadata(path, payload):
    metadata_dir = os.path.dirname(path)
    if metadata_dir:
        os.makedirs(metadata_dir, exist_ok=True)
    with open(path, "w") as metadata_file:
        json.dump(payload, metadata_file, indent=2, sort_keys=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Supervised bootstrap for the Santorini V3 architecture."
    )
    parser.add_argument("--examples-file", default=DEFAULT_EXAMPLES_FILE)
    parser.add_argument("--output-folder", default=DEFAULT_OUTPUT_FOLDER)
    parser.add_argument("--checkpoint-file", default="best.pth.tar")
    parser.add_argument("--final-checkpoint-file", default="final.pth.tar")
    parser.add_argument("--metadata-file", default="bootstrap_metadata.json")
    parser.add_argument("--architecture", choices=["v2", "v3"], default="v3")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    parsed_args = parse_args()
    random.seed(parsed_args.seed)
    np.random.seed(parsed_args.seed)
    torch.manual_seed(parsed_args.seed)

    use_cuda = bool(torch.cuda.is_available() and not parsed_args.cpu)
    nnet_args.cuda = use_cuda
    nnet_args.lr = parsed_args.lr
    nnet_args.dropout = parsed_args.dropout
    nnet_args.batch_size = parsed_args.batch_size
    nnet_args.quiet = parsed_args.quiet

    os.makedirs(parsed_args.output_folder, exist_ok=True)
    metadata_path = os.path.join(parsed_args.output_folder, parsed_args.metadata_file)

    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print("Loading examples: {}".format(parsed_args.examples_file))
    examples, history_lengths = load_examples(parsed_args.examples_file, max_examples=parsed_args.max_examples)
    if len(examples) < 2:
        raise ValueError("Need at least two examples for train/validation split.")
    print("Loaded {} examples across {} history windows.".format(len(examples), len(history_lengths)))

    train_indices, validation_indices = split_indices(
        len(examples),
        parsed_args.validation_fraction,
        parsed_args.seed,
    )
    print("Train examples: {}; validation examples: {}".format(len(train_indices), len(validation_indices)))

    game = SantoriniGame(5, true_random_placement=True)
    nnet = build_nnet(game, parsed_args.architecture)
    nnet.net_args.quiet = parsed_args.quiet
    print(
        "Initialized {} network: {} blocks, {} channels.".format(
            parsed_args.architecture,
            nnet.net_args.num_residual_blocks,
            nnet.net_args.num_channels,
        )
    )
    print("CUDA: {}".format(use_cuda))

    print("Encoding examples...")
    encoded_boards, target_pis, target_vs = encode_examples(nnet, examples)
    del examples

    rng = np.random.RandomState(parsed_args.seed + 1)
    optimizer = optim.Adam(nnet.nnet.parameters(), lr=nnet.net_args.lr)
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    epoch_metrics = []
    checkpoint_path = os.path.join(parsed_args.output_folder, parsed_args.checkpoint_file)
    final_checkpoint_path = os.path.join(parsed_args.output_folder, parsed_args.final_checkpoint_file)

    metadata = {
        "architecture": parsed_args.architecture,
        "batch_size": parsed_args.batch_size,
        "checkpoint_file": parsed_args.checkpoint_file,
        "cuda": use_cuda,
        "dropout": parsed_args.dropout,
        "epochs_requested": parsed_args.epochs,
        "example_count": int(len(encoded_boards)),
        "examples_file": parsed_args.examples_file,
        "history_lengths": history_lengths,
        "lr": parsed_args.lr,
        "max_examples": parsed_args.max_examples,
        "metadata_file": parsed_args.metadata_file,
        "min_delta": parsed_args.min_delta,
        "num_channels": nnet.net_args.num_channels,
        "num_residual_blocks": nnet.net_args.num_residual_blocks,
        "output_folder": parsed_args.output_folder,
        "patience": parsed_args.patience,
        "seed": parsed_args.seed,
        "started_at": started_at,
        "train_count": int(len(train_indices)),
        "validation_count": int(len(validation_indices)),
        "validation_fraction": parsed_args.validation_fraction,
        "epoch_metrics": epoch_metrics,
    }

    save_metadata(metadata_path, metadata)

    for epoch in range(1, parsed_args.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = train_epoch(
            nnet,
            optimizer,
            encoded_boards,
            target_pis,
            target_vs,
            train_indices,
            parsed_args.batch_size,
            use_cuda,
            rng,
        )
        validation_metrics = evaluate(
            nnet,
            encoded_boards,
            target_pis,
            target_vs,
            validation_indices,
            parsed_args.batch_size,
            use_cuda,
        )
        elapsed = time.perf_counter() - epoch_start

        improved = validation_metrics["total_loss"] < best_validation_loss - parsed_args.min_delta
        if improved:
            best_validation_loss = validation_metrics["total_loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            nnet.save_checkpoint(parsed_args.output_folder, parsed_args.checkpoint_file)
        else:
            epochs_without_improvement += 1

        epoch_record = {
            "epoch": epoch,
            "seconds": elapsed,
            "train": train_metrics,
            "validation": validation_metrics,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "improved": improved,
        }
        epoch_metrics.append(epoch_record)
        metadata.update({
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "completed_epochs": epoch,
            "epoch_metrics": epoch_metrics,
            "last_epoch_seconds": elapsed,
        })
        save_metadata(metadata_path, metadata)

        print(
            "Epoch {}/{}: train total={:.4f} pi={:.4f} v={:.4f}; "
            "val total={:.4f} pi={:.4f} v={:.4f}; best={} ({:.4f}); {:.1f}s".format(
                epoch,
                parsed_args.epochs,
                train_metrics["total_loss"],
                train_metrics["policy_loss"],
                train_metrics["value_loss"],
                validation_metrics["total_loss"],
                validation_metrics["policy_loss"],
                validation_metrics["value_loss"],
                best_epoch,
                best_validation_loss,
                elapsed,
            )
        )

        if epochs_without_improvement >= parsed_args.patience:
            print(
                "Early stopping after {} epochs without validation improvement.".format(
                    epochs_without_improvement
                )
            )
            break

    nnet.save_checkpoint(parsed_args.output_folder, parsed_args.final_checkpoint_file)
    metadata.update({
        "best_checkpoint_path": checkpoint_path,
        "final_checkpoint_path": final_checkpoint_path,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    save_metadata(metadata_path, metadata)

    print("Best checkpoint: {} (epoch {}, val total {:.4f})".format(
        checkpoint_path,
        best_epoch,
        best_validation_loss,
    ))
    print("Final checkpoint: {}".format(final_checkpoint_path))
    print("Metadata: {}".format(metadata_path))


if __name__ == "__main__":
    main()
