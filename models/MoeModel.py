import torch
import torch.nn as nn
import torch.nn.functional as F

class MoeModel(nn.Module):
    def __init__(self, data_length=[1024,128], class_length=[29,285], only_scene=False, only_action=False,
                 num_mixtures = 2):
        super(MoeModel, self).__init__()
        if only_scene:
            out_num = class_length[0]
        elif only_action:
            out_num = class_length[1]
        else:
            out_num = sum(class_length)

        self.num_mixtures = num_mixtures
        self.out_num = out_num

        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.gates = nn.Linear(sum(data_length), out_num * (num_mixtures+1))
        self.experts = nn.Linear(sum(data_length), out_num * num_mixtures)


    def forward(self, x):
        out = x.transpose(2,1)
        # out = x.float()
        # out = (out / 64) - 2
        
        out = self.GAP(out)
        out = out.squeeze(2)

        gate_activations = self.gates(out)
        gate_activations = gate_activations.view(-1, self.out_num, (self.num_mixtures+1))
        gate_activations = F.softmax(gate_activations, dim=2)

        expert_activations = self.experts(out)
        expert_activations = expert_activations.view(-1, self.out_num, self.num_mixtures)
        expert_activations = F.sigmoid(expert_activations)

        out = gate_activations[:,:,:-1] * expert_activations
        out = out.sum(2)

        out = torch.log(out) # for cross entropy loss

        return out