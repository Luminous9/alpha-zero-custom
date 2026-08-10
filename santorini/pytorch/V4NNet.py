"""Inference-only checkpoint adapter for V4 supervised candidates."""

from types import SimpleNamespace

import numpy as np
import torch

from santorini.V4Encoder import encode_v4_boards
from .SantoriniNNet import SantoriniNNet
from .V4Prototype import D4RegularNetwork


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
        boards = list(boards)
        if not boards:
            return (
                np.empty((0, self.action_size), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        encoded = encode_v4_boards(boards)[:, :self.input_channels]
        inputs = torch.from_numpy(np.ascontiguousarray(encoded)).to(self.device)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.autocast_fp16,
        ):
            log_policy, value = self.nnet(inputs)
        policies = torch.exp(log_policy).float().cpu().numpy()
        values = value[:, 0].float().cpu().numpy()
        if policies.shape != (len(boards), self.action_size):
            raise ValueError("V4 checkpoint returned the wrong policy shape.")
        return policies, values
