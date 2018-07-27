# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms

import numpy as np
from math import exp
import os
import sys

from data_loader import Coview_frame_dataset
from models import *
from utils import progress_bar


train_data_path = 'data/train/'
test_data_path = 'data/validate/'

only_scene = True
only_action = False # if you want use two label at the same time, then turn off both options
batchsize = 1024

load_weights = True
load_weights_path = 'weights/Temporal_GAP_FC1_only_scene_epoch63_acc95.2%/ckpt.pt'
# load_weights_path = 'weights/Temporal_GAP_FC1_action_only_action_epoch80_acc86.6%/ckpt.pt'


data_feature = ['rgb','audio']
data_length = [1024,128]
class_length = [29,285]
weight_l2_regularization = 5e-4
data_loader_worker_num = 2
data_rgb_audio_concat = True
max_data_temporal_length = 300


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


"""Load Network"""
net = Temporal_voting_FC1(data_length=data_length, class_length=class_length)
# net = Temporal_voting_FC1_action(data_length=data_length, class_length=class_length)


net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    # torch.backends.cudnn.enabled=False

if load_weights:
    # Load checkpoint.
    print('==> Resuming from trained model..')
    assert os.path.isfile(load_weights_path), 'Error: no weight file! %s'%(load_weights_path)
    checkpoint = torch.load(load_weights_path)
    # net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    best_epoch = start_epoch


# model_dict = net.state_dict()
# pretrained_dict = {k: v for k, v in checkpoint['net'].items() if k in net.state_dict()}
# model_dict.update(pretrained_dict) 
# net.load_state_dict(model_dict)


model_dict = net.state_dict()
model_dict.update({checkpoint['net'].items()[1][0]: checkpoint['net'].items()[1][1]})
model_dict.update({checkpoint['net'].items()[0][0]: checkpoint['net'].items()[0][1].unsqueeze(-1)})
net.load_state_dict(model_dict)

# criterion = nn.CrossEntropyLoss()



"""Load Dataset"""
transform_test = None
testset = Coview_frame_dataset(root=test_data_path, train=False, rgb_audio_concat=data_rgb_audio_concat,
                               transform=transform_test, max_data_temporal_length=max_data_temporal_length,
                               only_scene=only_scene, only_action=only_action)
testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=data_loader_worker_num)


# Test
def test(epoch):
    global best_acc
    global best_epoch
    net.eval()
    # test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        
        progress_bar(0, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (0, 0, 0, 0))
        for sub_batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            # loss = criterion(outputs, targets)

            # test_loss += loss.item()
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (sub_batch_idx+1) % batchsize == 0:
                progress_bar(sub_batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (0 / (sub_batch_idx + 1), 100. * correct / total, correct, total))

        if (sub_batch_idx+1) % batchsize != 0:
            progress_bar(sub_batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (0 / (sub_batch_idx + 1), 100. * correct / total, correct, total))
                  




test(0)
print('The best test accuracy: %f  epoch: %d'%(best_acc, best_epoch))