'''Temporal_GAP_FC1 in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Temporal_GAP_FC1(nn.Module):
    def __init__(self, data_length=[1024,128], class_length=[29,285], only_scene=False, only_action=False):
        super(Temporal_GAP_FC1, self).__init__()
        if only_scene:
            out_num = class_length[0]
        elif only_action:
            out_num = class_length[1]
        else:
            out_num = sum(class_length)

        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(sum(data_length), out_num)


    def forward(self, x):
        out = x.transpose(2,1)
        # out = x.float()
        # out = (out / 64) - 2
    
        out = self.GAP(out)
        out = out.squeeze(2)
        out = self.fc1(out)
        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out