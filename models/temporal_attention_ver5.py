import torch
import torch.nn as nn
import torch.nn.functional as F


class Temporal_Attention_ver5(nn.Module):
    def __init__(self, data_length=[1024,128], class_length=[29,285], only_scene=False, only_action=False):
        super(Temporal_Attention_ver5, self).__init__()
        if only_scene:
            out_num = class_length[0]
        elif only_action:
            out_num = class_length[1]
        else:
            out_num = sum(class_length)

        self.attention = nn.LSTM(input_size=sum(data_length), hidden_size=1, num_layers=1, bias=True,
                                 batch_first=True, dropout=0, bidirectional=False)
        self.fc1 = nn.Linear(sum(data_length), out_num)


    def forward(self, x):
        out = x #.transpose(2,1)
        # out = x.float()
        # out = (out / 64) - 2
        
        T_attention, _ = self.attention(out)
        T_attention = F.sigmoid(T_attention)
        T_attention_expand = T_attention.expand_as(out)

        out = out * T_attention_expand
        out = (out.sum(1))

        T_attention = T_attention.sum(1)
        T_attention = T_attention.expand_as(out)
        out = out / T_attention

        out = self.fc1(out)

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out
