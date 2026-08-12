"""Inference and AlphaZero-training adapters for V4 checkpoints."""

from collections import OrderedDict
import logging
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.optim as optim

from santorini.D4Canonical import (
    canonicalize_board_policies,
    canonicalize_boards,
)
from santorini.V4Encoder import encode_v4_boards
from utils import dotdict
from .NNet import NNetWrapper, args as runtime_args
from .SantoriniNNet import SantoriniNNet
from .V4Prototype import D4RegularNetwork


log = logging.getLogger(__name__)

V4_SELECTED_CONFIG = {
    "name": "ordinary_6x192_13_global_blend",
    "architecture": "ordinary",
    "planes": 13,
    "target": "global_blend",
    "candidate": "O-6x192",
    "channels": 192,
    "residual_blocks": 6,
}


def build_v4_model(game, config):
    """Construct the train-time model declared by a V4 checkpoint config."""
    architecture = config["architecture"]
    if architecture == "ordinary":
        args = SimpleNamespace(
            input_channels=int(config["planes"]),
            num_channels=int(config.get("channels", 96)),
            num_residual_blocks=int(config.get("residual_blocks", 8)),
            policy_channels=65,
            value_hidden_size=128,
            dropout=0.0,
        )
        return SantoriniNNet(game, args)
    if architecture != "equivariant":
        raise ValueError("Unknown V4 checkpoint architecture: {}".format(architecture))
    return D4RegularNetwork(
        game,
        input_channels=int(config["planes"]),
        effective_channels=int(config.get("effective_channels", 96)),
        residual_blocks=int(config.get("residual_blocks", 8)),
        value_hidden_size=128,
        dropout=0.0,
    )


def load_v4_checkpoint(path, game):
    # V4 resumable checkpoints deliberately include Python/NumPy RNG state and
    # therefore are not weights-only archives. PyTorch 2.6 changed the default
    # to weights_only=True, so make the trusted local-checkpoint contract
    # explicit while retaining compatibility with older PyTorch releases.
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if "config" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError("Not a V4 supervised checkpoint: {}".format(path))
    config = dict(checkpoint["config"])
    model = build_v4_model(game, config).eval()
    model.load_state_dict(checkpoint["state_dict"])
    return model, config, checkpoint


def export_v4_model(model, config):
    if config["architecture"] == "equivariant":
        return model.export_inference().eval()
    return model.eval()


class V4InferenceWrapper:
    """Expose a V4 checkpoint through the AlphaZero `predict[_batch]` API."""

    def __init__(
        self,
        game,
        checkpoint_path,
        device="auto",
        autocast_fp16=False,
        freeze_torchscript=True,
        canonicalize_d4=False,
        canonical_cache_size=0,
    ):
        self.game = game
        self.action_size = game.getActionSize()
        self.device = self._resolve_device(device)
        self.autocast_fp16 = bool(autocast_fp16)
        if self.autocast_fp16 and self.device.type != "cuda":
            raise ValueError("FP16 autocast requires a CUDA device.")
        model, self.config, self.checkpoint = load_v4_checkpoint(
            checkpoint_path, game
        )
        self.canonicalize_d4 = bool(canonicalize_d4)
        self.canonical_cache_size = int(canonical_cache_size)
        if self.canonical_cache_size < 0:
            raise ValueError("Canonical cache size cannot be negative.")
        if self.canonical_cache_size and not self.canonicalize_d4:
            raise ValueError("Canonical caching requires D4 canonicalization.")
        self._canonical_cache = OrderedDict()
        self._canonical_cache_hits = 0
        self._canonical_cache_misses = 0
        if self.canonicalize_d4 and self.config["architecture"] != "ordinary":
            raise ValueError(
                "D4 canonicalization is only needed for ordinary V4 checkpoints."
            )
        self._canonical_policy_permutations = None
        if self.canonicalize_d4:
            permutations = np.stack([
                game.getPolicySymmetryPermutation(rotations, flip)[1]
                for rotations in range(4)
                for flip in (False, True)
            ])
            self._canonical_policy_permutations = torch.as_tensor(
                permutations, dtype=torch.long, device=self.device
            )
        self.input_channels = int(self.config["planes"])
        model = export_v4_model(model, self.config).to(self.device).eval()
        if freeze_torchscript:
            example = torch.zeros(
                (2, self.input_channels, 5, 5),
                dtype=torch.float32,
                device=self.device,
            )
            with torch.inference_mode():
                model = torch.jit.freeze(torch.jit.trace(model, example))
        self.nnet = model

    def canonical_cache_info(self):
        """Return lightweight diagnostics for the optional exact-frame cache."""
        return {
            "size": len(self._canonical_cache),
            "capacity": self.canonical_cache_size,
            "hits": self._canonical_cache_hits,
            "misses": self._canonical_cache_misses,
        }

    @staticmethod
    def _canonical_cache_key(board):
        return np.ascontiguousarray(board, dtype=np.int8).tobytes()

    def _canonicalize_batch(self, boards):
        if not self.canonical_cache_size:
            canonical, matching_masks, _ = canonicalize_boards(
                boards, return_keys=False
            )
            return canonical, matching_masks

        canonical = np.empty((len(boards), 2, 5, 5), dtype=np.int8)
        matching_masks = np.empty((len(boards), 8), dtype=bool)
        missing_rows = []
        missing_boards = []
        missing_keys = []
        for row, board in enumerate(boards):
            key = self._canonical_cache_key(board)
            cached = self._canonical_cache.get(key)
            if cached is None:
                self._canonical_cache_misses += 1
                missing_rows.append(row)
                missing_boards.append(board)
                missing_keys.append(key)
                continue
            self._canonical_cache_hits += 1
            self._canonical_cache.move_to_end(key)
            canonical[row], matching_masks[row] = cached

        if missing_rows:
            missing_canonical, missing_masks, _ = canonicalize_boards(
                missing_boards, return_keys=False
            )
            for row, key, board, mask in zip(
                missing_rows, missing_keys, missing_canonical, missing_masks
            ):
                canonical[row] = board
                matching_masks[row] = mask
                self._canonical_cache[key] = (board.copy(), mask.copy())
                self._canonical_cache.move_to_end(key)
                while len(self._canonical_cache) > self.canonical_cache_size:
                    self._canonical_cache.popitem(last=False)
        return canonical, matching_masks

    def _restore_canonical_policies(self, canonical_policies, matching_masks):
        """Restore policy frames on-device before the unavoidable CPU copy."""
        matching_masks = np.asarray(matching_masks, dtype=bool)
        counts = matching_masks.sum(axis=1)
        if np.any(counts == 0):
            raise ValueError("Every canonical policy requires a matching transform.")
        restored = torch.empty_like(canonical_policies)
        single_rows = np.flatnonzero(counts == 1)
        if len(single_rows):
            row_indices = torch.as_tensor(
                single_rows, dtype=torch.long, device=self.device
            )
            transform_indices = torch.as_tensor(
                np.argmax(matching_masks[single_rows], axis=1),
                dtype=torch.long,
                device=self.device,
            )
            permutations = self._canonical_policy_permutations.index_select(
                0, transform_indices
            )
            restored.index_copy_(
                0,
                row_indices,
                torch.gather(
                    canonical_policies.index_select(0, row_indices),
                    1,
                    permutations,
                ),
            )

        symmetric_rows = np.flatnonzero(counts > 1)
        if len(symmetric_rows):
            symmetric_indices = torch.as_tensor(
                symmetric_rows, dtype=torch.long, device=self.device
            )
            restored.index_fill_(0, symmetric_indices, 0.0)
            for transform_index in range(8):
                rows = symmetric_rows[
                    matching_masks[symmetric_rows, transform_index]
                ]
                if not len(rows):
                    continue
                row_indices = torch.as_tensor(
                    rows, dtype=torch.long, device=self.device
                )
                permutation = self._canonical_policy_permutations[
                    transform_index
                ].expand(len(rows), -1)
                restored.index_add_(
                    0,
                    row_indices,
                    torch.gather(
                        canonical_policies.index_select(0, row_indices),
                        1,
                        permutation,
                    ),
                )
            restored.index_copy_(
                0,
                symmetric_indices,
                restored.index_select(0, symmetric_indices)
                / torch.as_tensor(
                    counts[symmetric_rows, None],
                    dtype=restored.dtype,
                    device=self.device,
                ),
            )
        return restored

    @staticmethod
    def _resolve_device(name):
        if isinstance(name, torch.device):
            return name
        if name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device(name)

    def predict(self, board):
        policies, values = self.predict_batch([board])
        return policies[0], float(values[0])

    def predict_batch(self, boards):
        boards = np.asarray(list(boards))
        if not len(boards):
            return (
                np.empty((0, self.action_size), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        matching_masks = None
        if self.canonicalize_d4:
            boards, matching_masks = self._canonicalize_batch(boards)
        encoded = encode_v4_boards(boards)[:, :self.input_channels]
        inputs = torch.from_numpy(np.ascontiguousarray(encoded)).to(self.device)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.autocast_fp16,
        ):
            log_policy, value = self.nnet(inputs)
        policies = torch.exp(log_policy).float()
        if matching_masks is not None:
            policies = self._restore_canonical_policies(
                policies, matching_masks
            )
        policies = policies.cpu().numpy()
        values = value[:, 0].float().cpu().numpy()
        if policies.shape != (len(boards), self.action_size):
            raise ValueError("V4 checkpoint returned the wrong policy shape.")
        return policies, values


class V4TrainableNNet(NNetWrapper):
    """Selected canonical 6x192 V4 model inside the normal AlphaZero API.

    The trainable module stays eager so optimizer state remains conventional.
    Prediction uses a refreshed frozen TorchScript snapshot by default.  Both
    replay inputs and inference inputs are mapped to the same exact D4 frame;
    returned inference policies are restored to the caller's frame.
    """

    architecture = "v4"

    def __init__(self, game):
        self.game = game
        self.config = dict(V4_SELECTED_CONFIG)
        self.net_args = dotdict({key: getattr(runtime_args, key) for key in runtime_args})
        self.net_args.input_channels = int(self.config["planes"])
        self.net_args.num_channels = int(self.config["channels"])
        self.net_args.num_residual_blocks = int(self.config["residual_blocks"])
        self.net_args.policy_channels = 65
        self.net_args.action_encoding = "spatial65-placement"
        # P1c used no dropout; changing it at handoff would change the training
        # function even before the first optimizer update.
        self.net_args.dropout = 0.0
        _, self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.native_action_size = self.action_size
        self.nnet = build_v4_model(game, self.config)
        if self.net_args.cuda:
            self.nnet.cuda()
        optimizer_class = (
            optim.AdamW if self.net_args.optimizer == "adamw" else optim.Adam
        )
        self.optimizer = optimizer_class(
            self.nnet.parameters(),
            lr=self.net_args.lr,
            weight_decay=self.net_args.weight_decay,
        )
        self.device = next(self.nnet.parameters()).device
        permutations = np.stack([
            game.getPolicySymmetryPermutation(rotations, flip)[1]
            for rotations in range(4)
            for flip in (False, True)
        ])
        self._canonical_policy_permutations = torch.as_tensor(
            permutations, dtype=torch.long, device=self.device
        )
        self._inference_nnet = None
        self._refresh_inference_model()

    @staticmethod
    def encode_board(board):
        return encode_v4_boards(np.asarray(board)[None])[0]

    @classmethod
    def encode_boards(cls, boards):
        return encode_v4_boards(np.asarray(boards))

    def _refresh_inference_model(self):
        model = self.nnet.eval()
        if bool(self.net_args.v4_freeze_torchscript):
            example = torch.zeros(
                (2, int(self.config["planes"]), 5, 5),
                dtype=torch.float32,
                device=self.device,
            )
            with torch.inference_mode():
                model = torch.jit.freeze(torch.jit.trace(model, example))
        self._inference_nnet = model

    def _restore_canonical_policies(self, canonical_policies, matching_masks):
        """Restore policy frames on-device before the single CPU transfer."""
        matching_masks = np.asarray(matching_masks, dtype=bool)
        counts = matching_masks.sum(axis=1)
        if np.any(counts == 0):
            raise ValueError("Every canonical policy requires a matching transform.")
        restored = torch.empty_like(canonical_policies)
        single_rows = np.flatnonzero(counts == 1)
        if len(single_rows):
            row_indices = torch.as_tensor(
                single_rows, dtype=torch.long, device=self.device
            )
            transform_indices = torch.as_tensor(
                np.argmax(matching_masks[single_rows], axis=1),
                dtype=torch.long,
                device=self.device,
            )
            permutations = self._canonical_policy_permutations.index_select(
                0, transform_indices
            )
            restored.index_copy_(
                0,
                row_indices,
                torch.gather(
                    canonical_policies.index_select(0, row_indices),
                    1,
                    permutations,
                ),
            )

        symmetric_rows = np.flatnonzero(counts > 1)
        if len(symmetric_rows):
            symmetric_indices = torch.as_tensor(
                symmetric_rows, dtype=torch.long, device=self.device
            )
            restored.index_fill_(0, symmetric_indices, 0.0)
            for transform_index in range(8):
                rows = symmetric_rows[
                    matching_masks[symmetric_rows, transform_index]
                ]
                if not len(rows):
                    continue
                row_indices = torch.as_tensor(
                    rows, dtype=torch.long, device=self.device
                )
                permutation = self._canonical_policy_permutations[
                    transform_index
                ].expand(len(rows), -1)
                restored.index_add_(
                    0,
                    row_indices,
                    torch.gather(
                        canonical_policies.index_select(0, row_indices),
                        1,
                        permutation,
                    ),
                )
            restored.index_copy_(
                0,
                symmetric_indices,
                restored.index_select(0, symmetric_indices)
                / torch.as_tensor(
                    counts[symmetric_rows, None],
                    dtype=restored.dtype,
                    device=self.device,
                ),
            )
        return restored

    def _encode_training_examples(self, examples):
        boards = [example[0] for example in examples]
        policies = [example[1] for example in examples]
        values = [example[2] for example in examples]
        canonical_boards, canonical_policies, _ = canonicalize_board_policies(
            self.game,
            np.asarray(boards),
            np.asarray(policies, dtype=np.float32),
        )
        return (
            encode_v4_boards(canonical_boards),
            canonical_policies.astype(np.float32, copy=False),
            np.asarray(values, dtype=np.float32),
        )

    def predict_batch(self, boards):
        boards = np.asarray(list(boards))
        if not len(boards):
            return (
                np.empty((0, self.action_size), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        canonical, matching_masks, _ = canonicalize_boards(
            boards, return_keys=False
        )
        encoded = encode_v4_boards(canonical)
        inputs = torch.from_numpy(np.ascontiguousarray(encoded)).to(self.device)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=bool(self.net_args.v4_autocast_fp16),
        ):
            log_policy, value = self._inference_nnet(inputs)
        policies = self._restore_canonical_policies(
            torch.exp(log_policy).float(), matching_masks
        )
        return policies.cpu().numpy(), value[:, 0].float().cpu().numpy()

    def train(self, examples, **kwargs):
        metrics = super().train(examples, **kwargs)
        self._refresh_inference_model()
        return metrics

    def _validation_metrics(self, examples):
        # Parent training validates before returning. Refresh here so those
        # metrics use the just-updated model, not the previous self-play
        # snapshot.
        self._refresh_inference_model()
        return super()._validation_metrics(examples)

    def save_checkpoint(
        self,
        folder="checkpoint",
        filename="checkpoint.pth.tar",
        include_optimizer=False,
        metadata=None,
    ):
        filepath = os.path.join(folder, filename)
        os.makedirs(folder, exist_ok=True)
        payload = {
            "schema_version": 3,
            "config": dict(self.config),
            "state_dict": self.nnet.state_dict(),
            "architecture": self.architecture,
            "action_encoding": self.net_args.action_encoding,
            "optimizer": self.net_args.optimizer,
        }
        if include_optimizer:
            payload.update({
                "optimizer_state_dict": self.optimizer.state_dict(),
                "numpy_rng_state": np.random.get_state(),
                "python_rng_state": random.getstate(),
                "torch_rng_state": torch.get_rng_state(),
            })
            if torch.cuda.is_available():
                payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        if metadata:
            payload["training_metadata"] = dict(metadata)
        torch.save(payload, filepath)

    def load_checkpoint(
        self,
        folder="checkpoint",
        filename="checkpoint.pth.tar",
        load_optimizer=False,
    ):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError("No model in path {}".format(filepath))
        try:
            checkpoint = torch.load(
                filepath,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(filepath, map_location=self.device)
        config = checkpoint.get("config")
        if config is None:
            raise ValueError("Checkpoint is missing its V4 architecture config.")
        for key in ("architecture", "planes", "channels", "residual_blocks"):
            if config.get(key) != self.config.get(key):
                raise ValueError(
                    "Checkpoint V4 config {}={} does not match selected {}.".format(
                        key, config.get(key), self.config.get(key)
                    )
                )
        self.nnet.load_state_dict(checkpoint["state_dict"])
        if load_optimizer:
            optimizer_state = checkpoint.get("optimizer_state_dict")
            if optimizer_state is None:
                log.warning(
                    "Checkpoint %s has no optimizer state; AdamW starts fresh.",
                    filepath,
                )
            else:
                self.optimizer.load_state_dict(optimizer_state)
            if "numpy_rng_state" in checkpoint:
                np.random.set_state(checkpoint["numpy_rng_state"])
            if "python_rng_state" in checkpoint:
                random.setstate(checkpoint["python_rng_state"])
            if "torch_rng_state" in checkpoint:
                torch.set_rng_state(
                    self._cpuByteRNGState(checkpoint["torch_rng_state"])
                )
            if self.net_args.cuda and "cuda_rng_state_all" in checkpoint:
                self._restore_cuda_rng_states(checkpoint["cuda_rng_state_all"])
        self._refresh_inference_model()
        return checkpoint.get("training_metadata", {})
