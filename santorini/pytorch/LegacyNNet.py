import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .NNet import args as v2_args
from .SantoriniNNet import ResidualBlock


class LegacySantoriniNNet(nn.Module):
    def __init__(self, game):
        super(LegacySantoriniNNet, self).__init__()
        _, self.board_x, self.board_y = game.getBoardSize()

        self.stem = nn.Sequential(
            nn.Conv2d(8, v2_args.num_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(v2_args.num_channels),
            nn.ReLU(inplace=True),
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(v2_args.num_channels) for _ in range(v2_args.num_residual_blocks)]
        )

        self.policy_conv = nn.Conv2d(v2_args.num_channels, 2, kernel_size=1, stride=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * self.board_x * self.board_y, 128)

        self.value_conv = nn.Conv2d(v2_args.num_channels, 1, kernel_size=1, stride=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(self.board_x * self.board_y, v2_args.value_hidden_size)
        self.value_fc2 = nn.Linear(v2_args.value_hidden_size, 1)

    def forward(self, s):
        s = self.stem(s)
        s = self.residual_blocks(s)

        pi = F.relu(self.policy_bn(self.policy_conv(s)))
        pi = pi.view(pi.size(0), -1)
        pi = self.policy_fc(pi)

        v = F.relu(self.value_bn(self.value_conv(s)))
        v = v.view(v.size(0), -1)
        v = F.dropout(F.relu(self.value_fc1(v)), p=v2_args.dropout, training=self.training)
        v = self.value_fc2(v)

        return F.log_softmax(pi, dim=1), torch.tanh(v)


class LegacyNNetWrapper:
    """
    Loads a V1 Santorini checkpoint and adapts its 128 worker-slot policy to
    the V2 physical-origin action space used by the current game.
    """

    def __init__(self, game):
        self.game = game
        self.nnet = LegacySantoriniNNet(game)
        if v2_args.cuda:
            self.nnet.cuda()

    @staticmethod
    def encode_board(board):
        pieces = board[0]
        heights = board[1]
        encoded = np.zeros((8, pieces.shape[0], pieces.shape[1]), dtype=np.float32)

        encoded[0] = pieces == 1
        encoded[1] = pieces == 2
        encoded[2] = pieces == -1
        encoded[3] = pieces == -2
        encoded[4] = heights == 1
        encoded[5] = heights == 2
        encoded[6] = heights == 3
        encoded[7] = heights >= 4

        return encoded

    @classmethod
    def encode_boards(cls, boards):
        return np.array([cls.encode_board(board) for board in boards], dtype=np.float32)

    def translate_policy(self, board, legacy_policy):
        translated = np.zeros(self.game.getActionSize(), dtype=np.float32)
        for worker_idx, origin in enumerate(self.game.getCharacterLocations(board, 1)):
            legacy_offset = worker_idx * 64
            v2_offset = self.game.getActionFromOrigin(origin, 0, 0)
            translated[v2_offset:v2_offset + 64] = legacy_policy[legacy_offset:legacy_offset + 64]
        return translated

    def predict(self, board):
        policies, values = self.predict_batch([board])
        return policies[0], float(values[0])

    def predict_batch(self, boards):
        encoded = torch.FloatTensor(self.encode_boards(boards))
        if v2_args.cuda:
            encoded = encoded.contiguous().cuda()

        self.nnet.eval()
        with torch.no_grad():
            legacy_pi, v = self.nnet(encoded)

        legacy_policies = torch.exp(legacy_pi).data.cpu().numpy()
        translated_policies = np.array(
            [
                self.translate_policy(board, legacy_policy)
                for board, legacy_policy in zip(boards, legacy_policies)
            ],
            dtype=np.float32,
        )
        return translated_policies, v.view(-1).data.cpu().numpy()

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError("No model in path {}".format(filepath))
        map_location = None if v2_args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
