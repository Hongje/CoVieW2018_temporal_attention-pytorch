"""This is old version"""
"""Cannot run at ver2"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Temporal_FC1_GAP(nn.Module):
    def __init__(self, data_length=[1024,128], class_length=[29,285]):
        super(Temporal_FC1_GAP, self).__init__()
        self.fc1_conv = nn.Conv1d(sum(data_length), class_length[0], 1)


    def forward(self, x, length):
        x = x.transpose(2,1)
        out = x.float()
        length = length.float()

        out = (out / 64) - 2

        out = self.fc1_conv(out)

        out = (out.sum(2))
        length = length.unsqueeze(1)
        length = length.expand_as(out)
        out = out / length
        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out

class Temporal_FC1_GAP_action(nn.Module):
    def __init__(self, data_length=[1024,128], class_length=[29,285]):
        super(Temporal_FC1_GAP_action, self).__init__()
        self.fc1_conv = nn.Conv1d(sum(data_length), class_length[1], 1)


    def forward(self, x, length):
        x = x.transpose(2,1)
        out = x.float()
        length = length.float()

        out = (out / 64) - 2

        out = self.fc1_conv(out)

        out = (out.sum(2))
        length = length.unsqueeze(1)
        length = length.expand_as(out)
        out = out / length
        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out