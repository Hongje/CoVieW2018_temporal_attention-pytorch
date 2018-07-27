"""This is old version"""
"""Cannot run at ver2"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Temporal_Attention(nn.Module):
    def __init__(self, data_length=[1024,128], class_length=[29,285]):
        super(Temporal_Attention, self).__init__()
        self.attention = nn.Conv1d(in_channels=sum(data_length), out_channels=1, kernel_size=1, bias=False)
        self.fc1 = nn.Linear(sum(data_length), class_length[0])


    def forward(self, x, length):
        x = x.transpose(2,1)
        out = x.float()
        length = length.float()

        out = (out / 64) - 2

        T_attention = self.attention(out)
        T_attention = F.sigmoid(T_attention)
        T_attention_expand = T_attention.expand_as(out)

        out = out * T_attention_expand
        out = (out.sum(2))
        T_attention = T_attention.sum(2)
        T_attention = T_attention.expand_as(out)
        out = out / T_attention

        out = self.fc1(out)

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out

class Temporal_Attention_bias(nn.Module):
    def __init__(self, data_length=[1024,128], class_length=[29,285]):
        super(Temporal_Attention_bias, self).__init__()
        self.attention = nn.Conv1d(in_channels=sum(data_length), out_channels=1, kernel_size=1, bias=True)
        self.fc1 = nn.Linear(sum(data_length), class_length[0])


    def forward(self, x, length):
        x = x.transpose(2,1)
        out = x.float()
        length = length.float()

        out = (out / 64) - 2

        T_attention = self.attention(out)
        T_attention = F.sigmoid(T_attention)
        T_attention_expand = T_attention.expand_as(out)

        out = out * T_attention_expand
        out = (out.sum(2))
        T_attention = T_attention.sum(2)
        T_attention = T_attention.expand_as(out)
        out = out / T_attention

        out = self.fc1(out)

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out


class Temporal_Attention_bias_action(nn.Module):
    def __init__(self, data_length=[1024,128], class_length=[29,285]):
        super(Temporal_Attention_bias_action, self).__init__()
        self.attention = nn.Conv1d(in_channels=sum(data_length), out_channels=1, kernel_size=1, bias=True)
        self.fc1 = nn.Linear(sum(data_length), class_length[1])


    def forward(self, x, length):
        x = x.transpose(2,1)
        out = x.float()
        length = length.float()

        out = (out / 64) - 2

        T_attention = self.attention(out)
        T_attention = F.sigmoid(T_attention)
        T_attention_expand = T_attention.expand_as(out)

        out = out * T_attention_expand
        out = (out.sum(2))
        T_attention = T_attention.sum(2)
        T_attention = T_attention.expand_as(out)
        out = out / T_attention

        out = self.fc1(out)

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out