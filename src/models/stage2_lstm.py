from __future__ import annotations

import torch
import torch.nn as nn


class Stage2LSTM(nn.Module):
    """FloPNet second-stage BiLSTM with left, right, and double-eye branches."""

    def __init__(self, input_size: int = 4, hidden_size: int = 64, num_layers: int = 2, bidirectional: bool = True):
        super().__init__()
        self.position_len = 15
        fc_size = hidden_size * 2 if bidirectional else hidden_size
        self.left_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.left_fc = nn.Linear(fc_size, 2)
        self.right_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.right_fc = nn.Linear(fc_size, 2)
        self.double_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.double_fc = nn.Linear(fc_size, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, dir_input, position_input):
        dirs = torch.max(dir_input, dim=1)[1]
        mask_left = ((dirs == 0) | (dirs == 5)).unsqueeze(-1).float()
        mask_right = ((dirs == 2) | (dirs == 3)).unsqueeze(-1).float()
        mask_double = ((dirs == 1) | (dirs == 4)).unsqueeze(-1).float()

        self.left_lstm.flatten_parameters()
        self.right_lstm.flatten_parameters()
        self.double_lstm.flatten_parameters()

        left, _ = self.left_lstm(position_input[:, : self.position_len, :])
        left = self.left_fc(self.dropout(left[:, -1, :])) * mask_left

        right, _ = self.right_lstm(position_input[:, self.position_len :, :])
        right = self.right_fc(self.dropout(right[:, -1, :])) * mask_right

        double, _ = self.double_lstm(position_input)
        double = self.double_fc(self.dropout(double[:, -1, :])) * mask_double

        return left + right + double


def stage2_lstm(**kwargs):
    return Stage2LSTM(**kwargs)
