'''Temporal_GAP_FC1 in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Temporal_voting_FC1(nn.Module):
    def __init__(self, data_length=[1024,128], class_length=[29,285]):
        super(Temporal_voting_FC1, self).__init__()
        self.GAP = nn.AdaptiveAvgPool1d(1)
        # self.fc1 = nn.Linear(sum(data_length), class_length[0])
        self.fc1 = nn.Conv1d(sum(data_length), class_length[0], 1, bias=True)


    def forward(self, x):
        out = x.transpose(2,1)
        # out = x.float()
        # out = (out / 64) - 2

        out = self.fc1(out)

        vote = out.max(1)[1][0]
        result = torch.zeros([1,29]).cuda()
        for i in range(out.shape[2]):
            result[0,vote[i]] += 1
        
        result2 = torch.zeros([1,29]).cuda()
        result2[0, result.max(1)[1]] = 1

        return result2

class Temporal_voting_FC1_action(nn.Module):
    def __init__(self, data_length=[1024,128], class_length=[29,285]):
        super(Temporal_voting_FC1_action, self).__init__()
        self.GAP = nn.AdaptiveAvgPool1d(1)
        # self.fc1 = nn.Linear(sum(data_length), class_length[1])
        self.fc1 = nn.Conv1d(sum(data_length), class_length[1], 1, bias=True)


    def forward(self, x):
        out = x.transpose(2,1)
        # out = x.float()
        # out = (out / 64) - 2

        out = self.fc1(out)

        vote = out.max(1)[1][0]
        result = torch.zeros([1,285]).cuda()
        for i in range(out.shape[2]):
            result[0,vote[i]] += 1
        
        result2 = torch.zeros([1,285]).cuda()
        result2[0, result.max(1)[1]] = 1

        return result2