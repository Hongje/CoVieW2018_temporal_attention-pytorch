import torch
import torch.nn as nn
import torch.nn.functional as F


class Temporal_Attention_ver3_kernel3(nn.Module):
    def __init__(self, data_length=[1024,128], class_length=[29,285], only_scene=False, only_action=False):
        super(Temporal_Attention_ver3_kernel3, self).__init__()
        if only_scene:
            out_num = class_length[0]
        elif only_action:
            out_num = class_length[1]
        else :
            out_num = sum(class_length)

        self.attention = nn.Conv1d(in_channels=sum(data_length), out_channels=1, kernel_size=3, padding=1, bias=False)
        self.fc1 = nn.Linear(sum(data_length), out_num)


    def forward(self, x):
        out = x.transpose(2,1)
        # out = x.float()
        # out = (out / 64) - 2
        
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


class Temporal_Attention_bias_ver3_kernel3(nn.Module):
    def __init__(self, data_length=[1024,128], class_length=[29,285], only_scene=False, only_action=False,
                 temporal_bias_init = None):
        super(Temporal_Attention_bias_ver3_kernel3, self).__init__()
        if only_scene:
            out_num = class_length[0]
        elif only_action:
            out_num = class_length[1]
        else :
            out_num = sum(class_length)

        self.attention = nn.Conv1d(in_channels=sum(data_length), out_channels=1, kernel_size=3, padding=1, bias=True)
        if temporal_bias_init is not None:
            self.attention.bias = torch.nn.Parameter(torch.tensor([float(temporal_bias_init)]))
        self.fc1 = nn.Linear(sum(data_length), out_num)


    def forward(self, x):
        out = x.transpose(2,1)
        # out = x.float()
        # out = (out / 64) - 2
        
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
