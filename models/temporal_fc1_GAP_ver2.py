import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Temporal_FC1_GAP_ver2(nn.Module):
    def __init__(self, data_length=[1024,128], class_length=[29,285], only_scene=False, only_action=False):
        super(Temporal_FC1_GAP_ver2, self).__init__()
        if only_scene:
            out_num = class_length[0]
        elif only_action:
            out_num = class_length[1]
        else:
            out_num = sum(class_length)

        self.fc1_conv = nn.Conv1d(sum(data_length), out_num, 1, bias=True)
        self.temporal_GAP = nn.AdaptiveAvgPool1d(1)


    def forward(self, x):
        out = x.transpose(2,1)
        # out = x.float()
        # out = (out / 64) - 2

        out = self.fc1_conv(out)
        out = self.temporal_GAP(out)
        out = out.squeeze(2)
        return out
        