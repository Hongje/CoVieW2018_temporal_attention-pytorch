import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Temporal_conv1d(nn.Module):
    def __init__(self, data_length=[1024,128], class_length=[29,285], only_scene=False, only_action=False):
        super(Temporal_conv1d, self).__init__()
        if only_scene:
            out_num = class_length[0]
        elif only_action:
            out_num = class_length[1]
        else:
            out_num = sum(class_length)

        conv1_filters = sum(data_length)
        # conv1_filters = 2048

        self.conv1 = nn.Conv1d(sum(data_length), conv1_filters, 3, padding=1, bias=True)
        self.temporal_GAP = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(conv1_filters, out_num)


    def forward(self, x):
        # out = x.float()
        # out = (out / 64) - 2

        out = torch.transpose(x, 1, 2)
        out = self.conv1(out)
        out = self.temporal_GAP(out)
        out = out.squeeze(2)
        out = self.fc1(out)
        return out
