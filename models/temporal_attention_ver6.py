import torch
import torch.nn as nn
import torch.nn.functional as F


class Temporal_Attention_ver6(nn.Module):
    def __init__(self, data_length=[1024,128], class_length=[29,285], only_scene=False, only_action=False):
        super(Temporal_Attention_ver6, self).__init__()
        if only_scene:
            out_num = class_length[0]
        elif only_action:
            out_num = class_length[1]
        else:
            out_num = sum(class_length)

        attention_hidden_size = sum(data_length)

        self.attention = nn.GRU(input_size=sum(data_length), hidden_size=attention_hidden_size, num_layers=2, bias=True,
                                batch_first=True, dropout=0, bidirectional=False)
        self.attention_fc = nn.Conv1d(in_channels=attention_hidden_size, out_channels=1, kernel_size=1, bias=True)
        self.fc1 = nn.Linear(sum(data_length), out_num)


    def forward(self, x):
        out = x #.transpose(2,1)
        # out = x.float()
        # out = (out / 64) - 2
        
        T_attention, _ = self.attention(out)
        T_attention = T_attention.transpose(2,1)
        T_attention = self.attention_fc(T_attention)
        T_attention = F.sigmoid(T_attention)
        T_attention = T_attention.transpose(2,1)
        T_attention_expand = T_attention.expand_as(out)

        out = out * T_attention_expand
        out = (out.sum(1))

        T_attention = T_attention.sum(1)
        T_attention = T_attention.expand_as(out)
        out = out / T_attention

        out = self.fc1(out)

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out


class Temporal_Attention_bidirection_ver6(nn.Module):
    def __init__(self, data_length=[1024,128], class_length=[29,285], only_scene=False, only_action=False):
        super(Temporal_Attention_bidirection_ver6, self).__init__()
        if only_scene:
            out_num = class_length[0]
        elif only_action:
            out_num = class_length[1]
        else:
            out_num = sum(class_length)

        attention_hidden_size = sum(data_length)

        self.attention = nn.GRU(input_size=sum(data_length), hidden_size=attention_hidden_size, num_layers=1, bias=True,
                                batch_first=True, dropout=0, bidirectional=False)
        self.attention2 = nn.GRU(input_size=sum(data_length), hidden_size=attention_hidden_size, num_layers=1, bias=True,
                                batch_first=True, dropout=0, bidirectional=False)
        self.attention_fc = nn.Conv1d(in_channels=attention_hidden_size, out_channels=1, kernel_size=1, bias=True)
        self.fc1 = nn.Linear(sum(data_length), out_num)


    def forward(self, x):
        out = x #.transpose(2,1)
        # out = x.float()
        # out = (out / 64) - 2
        
        T_attention, _ = self.attention(out)
        inv_idx = torch.arange(T_attention.size(1)-1, -1, -1).long().cuda()
        T_attention = T_attention.index_select(1, inv_idx)
        T_attention, _ = self.attention2(T_attention)
        T_attention = T_attention.transpose(2,1)
        T_attention = self.attention_fc(T_attention)
        T_attention = F.sigmoid(T_attention)
        T_attention = T_attention.transpose(2,1)
        T_attention_expand = T_attention.expand_as(out)

        out = out * T_attention_expand
        out = (out.sum(1))

        T_attention = T_attention.sum(1)
        T_attention = T_attention.expand_as(out)
        out = out / T_attention

        out = self.fc1(out)

        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out
