"""Exact D4 reference prototype used to gate the optimized V4 G-CNN."""

import copy
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .SantoriniNNet import SantoriniNNet


D4_ELEMENTS = tuple((rotations, flip) for rotations in range(4) for flip in (False, True))


def transform_spatial(tensor, rotations, flip):
    transformed = torch.rot90(tensor, int(rotations), dims=(-2, -1))
    return torch.flip(transformed, dims=(-1,)) if flip else transformed


def _d4_composition_table():
    marker = np.arange(25).reshape(5, 5)

    def transform(array, element):
        rotations, flip = element
        result = np.rot90(array, rotations)
        return np.fliplr(result) if flip else result

    transformed = [transform(marker, element) for element in D4_ELEMENTS]
    table = np.zeros((8, 8), dtype=np.int64)
    for first in range(8):
        for second in range(8):
            composed = transform(transformed[second], D4_ELEMENTS[first])
            matches = [index for index, candidate in enumerate(transformed) if np.array_equal(composed, candidate)]
            if len(matches) != 1:
                raise AssertionError("D4 composition table is not closed.")
            table[first, second] = matches[0]
    return table


D4_COMPOSE = _d4_composition_table()


class D4RegularConv2d(nn.Module):
    """A block-circulant convolution for right-regular D4 feature fibers."""

    def __init__(self, in_multiplicity, out_multiplicity, kernel_size=3, padding=1):
        super().__init__()
        self.in_multiplicity = int(in_multiplicity)
        self.out_multiplicity = int(out_multiplicity)
        self.kernel_size = int(kernel_size)
        self.padding = int(padding)
        self.weight = nn.Parameter(torch.empty(
            8,
            self.out_multiplicity,
            self.in_multiplicity,
            self.kernel_size,
            self.kernel_size,
        ))
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        relative = np.zeros((8, 8), dtype=np.int64)
        for output_group in range(8):
            for input_group in range(8):
                matches = [
                    relative_group
                    for relative_group in range(8)
                    if D4_COMPOSE[relative_group, output_group] == input_group
                ]
                if len(matches) != 1:
                    raise AssertionError("D4 relative group lookup is ambiguous.")
                relative[output_group, input_group] = matches[0]
        self.register_buffer("relative_groups", torch.as_tensor(relative, dtype=torch.long))

    def expanded_weight(self):
        rows = []
        for output_group in range(8):
            rows.append(torch.cat([
                self.weight[self.relative_groups[output_group, input_group]]
                for input_group in range(8)
            ], dim=1))
        return torch.cat(rows, dim=0)

    def forward(self, features):
        batch, groups, multiplicity, height, width = features.shape
        if groups != 8 or multiplicity != self.in_multiplicity:
            raise ValueError("D4 regular feature shape does not match this convolution.")
        flattened = features.reshape(batch, groups * multiplicity, height, width)
        output = F.conv2d(flattened, self.expanded_weight(), padding=self.padding)
        return output.reshape(batch, 8, self.out_multiplicity, height, width)

    def export(self):
        return FrozenExpandedRegularConv2d(
            self.expanded_weight().detach(), self.padding, self.out_multiplicity
        )


class FrozenExpandedRegularConv2d(nn.Module):
    """Inference-only ordinary Conv2d expansion of a tied regular convolution."""

    def __init__(self, weight, padding, out_multiplicity):
        super().__init__()
        self.out_multiplicity = int(out_multiplicity)
        self.conv = nn.Conv2d(
            int(weight.shape[1]), int(weight.shape[0]),
            kernel_size=int(weight.shape[-1]), padding=int(padding), bias=False,
        )
        self.conv.weight.data.copy_(weight)
        self.conv.weight.requires_grad_(False)

    def forward(self, features):
        batch, groups, multiplicity, height, width = features.shape
        flattened = features.reshape(batch, groups * multiplicity, height, width)
        output = self.conv(flattened)
        return output.reshape(batch, 8, self.out_multiplicity, height, width)


class D4RegularResidualBlock(nn.Module):
    def __init__(self, multiplicity):
        super().__init__()
        self.conv1 = D4RegularConv2d(multiplicity, multiplicity)
        self.bn1 = nn.BatchNorm2d(multiplicity)
        self.conv2 = D4RegularConv2d(multiplicity, multiplicity)
        self.bn2 = nn.BatchNorm2d(multiplicity)

    @staticmethod
    def _batch_norm(features, batch_norm):
        batch, groups, channels, height, width = features.shape
        normalized = batch_norm(features.reshape(batch * groups, channels, height, width))
        return normalized.reshape(batch, groups, channels, height, width)

    def forward(self, features):
        residual = features
        features = F.relu(self._batch_norm(self.conv1(features), self.bn1))
        features = self._batch_norm(self.conv2(features), self.bn2)
        return F.relu(features + residual)


class D4RegularNetwork(nn.Module):
    """Optimized hand-rolled regular-representation V4 feasibility tower."""

    def __init__(
        self,
        game,
        input_channels=13,
        effective_channels=96,
        residual_blocks=8,
        value_hidden_size=128,
        dropout=0.0,
    ):
        super().__init__()
        if int(effective_channels) % 8:
            raise ValueError("D4 effective channels must be divisible by eight.")
        self.multiplicity = int(effective_channels) // 8
        self.dropout = float(dropout)
        self.lift = nn.Conv2d(
            int(input_channels), self.multiplicity, kernel_size=3, padding=1, bias=False
        )
        self.lift_bn = nn.BatchNorm2d(self.multiplicity)
        self.blocks = nn.ModuleList([
            D4RegularResidualBlock(self.multiplicity)
            for _ in range(int(residual_blocks))
        ])
        self.policy_conv = nn.Conv2d(self.multiplicity, 65, kernel_size=1)
        self.value_conv = nn.Conv2d(self.multiplicity, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(25, int(value_hidden_size))
        self.value_fc2 = nn.Linear(int(value_hidden_size), 1)
        self.oracle_value_conv = nn.Conv2d(self.multiplicity, 1, kernel_size=1)
        self.oracle_value_bn = nn.BatchNorm2d(1)
        self.oracle_value_fc1 = nn.Linear(25, int(value_hidden_size))
        self.oracle_value_fc2 = nn.Linear(int(value_hidden_size), 1)
        permutations = []
        for rotations, flip in D4_ELEMENTS:
            _, new_indices = game.getPolicySymmetryPermutation(rotations, flip)
            permutations.append(np.asarray(new_indices, dtype=np.int64))
        self.register_buffer(
            "forward_policy_indices",
            torch.as_tensor(np.asarray(permutations), dtype=torch.long),
        )

    def _lift(self, inputs):
        batch = inputs.size(0)
        transformed = torch.cat([
            transform_spatial(inputs, rotations, flip)
            for rotations, flip in D4_ELEMENTS
        ], dim=0)
        features = F.relu(self.lift_bn(self.lift(transformed)))
        return features.reshape(8, batch, self.multiplicity, 5, 5).permute(1, 0, 2, 3, 4)

    def _trunk(self, inputs):
        features = self._lift(inputs)
        for block in self.blocks:
            features = block(features)
        return features

    def _policy(self, features):
        batch = features.size(0)
        local = features.reshape(batch * 8, self.multiplicity, 5, 5)
        local_policy = self.policy_conv(local).permute(0, 2, 3, 1).reshape(batch, 8, -1)
        canonical_logits = []
        for group_index in range(8):
            canonical_logits.append(
                local_policy[:, group_index, self.forward_policy_indices[group_index]]
            )
        return F.log_softmax(torch.stack(canonical_logits, dim=0).mean(dim=0), dim=1)

    def _value_head(self, features, conv, batch_norm, fc1, fc2):
        batch = features.size(0)
        pooled = features.mean(dim=1)
        value = F.relu(batch_norm(conv(pooled))).reshape(batch, -1)
        value = F.dropout(F.relu(fc1(value)), p=self.dropout, training=self.training)
        return torch.tanh(fc2(value))

    def forward(self, inputs):
        features = self._trunk(inputs)
        value = self._value_head(
            features, self.value_conv, self.value_bn, self.value_fc1, self.value_fc2
        )
        return self._policy(features), value

    def forward_with_auxiliary(self, inputs):
        features = self._trunk(inputs)
        value = self._value_head(
            features, self.value_conv, self.value_bn, self.value_fc1, self.value_fc2
        )
        oracle_value = self._value_head(
            features,
            self.oracle_value_conv,
            self.oracle_value_bn,
            self.oracle_value_fc1,
            self.oracle_value_fc2,
        )
        return self._policy(features), value, oracle_value

    def export_inference(self):
        exported = copy.deepcopy(self).eval()
        for block in exported.blocks:
            block.conv1 = block.conv1.export()
            block.conv2 = block.conv2.export()
        return exported


class D4SymmetrizedReference(nn.Module):
    """Eight tied orientation branches with exact policy/value symmetrization.

    This is deliberately a correctness/export reference, not the throughput
    candidate: it evaluates the shared tower eight times. The optimized regular-
    representation implementation must match its transformation contract while
    avoiding this 8x branch cost.
    """

    def __init__(
        self,
        game,
        input_channels=13,
        channels=16,
        residual_blocks=1,
        value_hidden_size=32,
        dropout=0.0,
    ):
        super().__init__()
        args = SimpleNamespace(
            input_channels=int(input_channels),
            num_channels=int(channels),
            num_residual_blocks=int(residual_blocks),
            policy_channels=65,
            value_hidden_size=int(value_hidden_size),
            dropout=float(dropout),
        )
        self.base = SantoriniNNet(game, args)
        permutations = []
        for rotations, flip in D4_ELEMENTS:
            _, new_indices = game.getPolicySymmetryPermutation(rotations, flip)
            permutations.append(np.asarray(new_indices, dtype=np.int64))
        self.register_buffer(
            "forward_policy_indices",
            torch.as_tensor(np.asarray(permutations), dtype=torch.long),
        )

    def forward(self, inputs):
        batch_size = inputs.size(0)
        transformed = torch.cat([
            transform_spatial(inputs, rotations, flip)
            for rotations, flip in D4_ELEMENTS
        ], dim=0)
        branch_log_policy, branch_value = self.base(transformed)
        branch_policy = branch_log_policy.exp().view(len(D4_ELEMENTS), batch_size, -1)
        branch_value = branch_value.view(len(D4_ELEMENTS), batch_size, 1)
        canonical_policies = []
        for group_index in range(len(D4_ELEMENTS)):
            # Forward permutation satisfies transformed[new] = original[old].
            # Reading transformed[new] therefore maps a local branch back to the
            # canonical old-index order.
            canonical_policies.append(
                branch_policy[group_index][:, self.forward_policy_indices[group_index]]
            )
        policy = torch.stack(canonical_policies, dim=0).mean(dim=0)
        value = branch_value.mean(dim=0)
        return torch.log(policy.clamp_min(1e-30)), value
