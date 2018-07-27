'''Temporal_GAP_FC1 in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Temporal_Pooling_voting(nn.Module):
    def __init__(self, data_length=[1024,128], class_length=[29,285], only_scene=False, only_action=False,
                 selector_init = False):
        super(Temporal_Pooling_voting, self).__init__()
        if only_scene:
            out_num = class_length[0]
        elif only_action:
            out_num = class_length[1]
        else:
            out_num = sum(class_length)

        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.MAX = nn.AdaptiveMaxPool1d(1)
        
        if selector_init:
            gap_selector = torch.ones([sum(data_length),1]) * 10
            max_selector = torch.ones([sum(data_length),1])
            min_selector = torch.ones([sum(data_length),1])
            selector_tensor = torch.cat((gap_selector,max_selector,min_selector), dim=1)
            self.pool_selector = nn.Parameter(selector_tensor, requires_grad=True)
        else :
            self.pool_selector = nn.Parameter(torch.randn([sum(data_length),3]), requires_grad=True)

        self.fc1 = nn.Linear(sum(data_length), out_num)


    def forward(self, x):
        out = x.transpose(2,1)
        # out = x.float()
        # out = (out / 64) - 2
    
        pool_gap = self.GAP(out).squeeze()
        pool_max = self.MAX(out).squeeze()
        pool_min, _ = torch.min(out, dim=2, keepdim=True)
        pool_min = pool_min.squeeze()

        selector = F.softmax(self.pool_selector, dim=1)

        pool_gap = torch.mul(pool_gap, selector[:,0])
        pool_max = torch.mul(pool_max, selector[:,1])
        pool_min = torch.mul(pool_min, selector[:,2])

        out = pool_gap + pool_max + pool_min
        out = out.unsqueeze(0)
        out = self.fc1(out)
        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out


class Temporal_N_Pooling_voting(nn.Module):
    def __init__(self, data_length=[1024,128], class_length=[29,285], only_scene=False, only_action=False,
                 selector_init = False, feature_dim = 1152):
        super(Temporal_N_Pooling_voting, self).__init__()
        if only_scene:
            out_num = class_length[0]
        elif only_action:
            out_num = class_length[1]
        else:
            out_num = sum(class_length)

        self.fc_conv = self.attention = nn.Conv1d(in_channels=sum(data_length), out_channels=feature_dim, kernel_size=1, bias=True)

        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.MAX = nn.AdaptiveMaxPool1d(1)
        
        if selector_init:
            gap_selector = torch.ones([feature_dim,1]) * 10
            max_selector = torch.ones([feature_dim,1])
            min_selector = torch.ones([feature_dim,1])
            selector_tensor = torch.cat((gap_selector,max_selector,min_selector), dim=1)
            self.pool_selector = nn.Parameter(selector_tensor, requires_grad=True)
        else :
            self.pool_selector = nn.Parameter(torch.randn([feature_dim,3]), requires_grad=True)

        self.fc1 = nn.Linear(feature_dim, out_num)


    def forward(self, x):
        out = x.transpose(2,1)
        # out = x.float()
        # out = (out / 64) - 2
        
        out = self.fc_conv(out)
    
        pool_gap = self.GAP(out).squeeze()
        pool_max = self.MAX(out).squeeze()
        pool_min, _ = torch.min(out, dim=2, keepdim=True)
        pool_min = pool_min.squeeze()

        selector = F.softmax(self.pool_selector, dim=1)

        pool_gap = torch.mul(pool_gap, selector[:,0])
        pool_max = torch.mul(pool_max, selector[:,1])
        pool_min = torch.mul(pool_min, selector[:,2])

        out = pool_gap + pool_max + pool_min
        out = out.unsqueeze(0)
        out = self.fc1(out)
        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out


class Temporal_N_Pooling(nn.Module):
    def __init__(self, data_length=[1024,128], class_length=[29,285], only_scene=False, only_action=False,
                 feature_dim = 1152):
        super(Temporal_N_Pooling, self).__init__()
        if only_scene:
            out_num = class_length[0]
        elif only_action:
            out_num = class_length[1]
        else:
            out_num = sum(class_length)

        self.fc_conv = self.attention = nn.Conv1d(in_channels=sum(data_length), out_channels=feature_dim, kernel_size=1, bias=True)

        self.GAP = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(feature_dim, out_num)


    def forward(self, x):
        out = x.transpose(2,1)
        # out = x.float()
        # out = (out / 64) - 2
        
        out = self.fc_conv(out)
    
        out = self.GAP(out).squeeze(2)
        out = self.fc1(out)
        # nn.CrossEntropyLoss() has softmax(). Therefore, there is no softmax() on this network.
        return out