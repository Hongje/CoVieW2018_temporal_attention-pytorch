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
from data_loader import Coview_frame_dataset_final
from models import *
from utils import progress_bar

import csv


train_data_path = 'data/train/'
test_data_path = 'data/test/'


# only_scene = True
# only_action = False

load_weights = True
load_weights_path_scene = 'results/Temporal_Attention_bias_ver2_epoch82_acc94.467_/ckpt.pt'
load_weights_path_action = 'results/Temporal_GAP_FC1_noise10_only_action_epoch150_acc86.967_/ckpt.pt'

save_result_path = 'results/Temporal_Attention.csv'

data_feature = ['rgb','audio']
data_length = [1024,128]
class_length = [29,285]
data_loader_worker_num = 4
data_rgb_audio_concat = True
max_data_temporal_length = 300

top_k = 20


device = 'cuda' if torch.cuda.is_available() else 'cpu'




"""Load scene Network"""
only_scene = True
only_action = False
# net1 = Temporal_GAP_FC1(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net1 = Temporal_FC1_GAP_ver2(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net1 = Temporal_conv1d(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net1 = Temporal_Attention_ver2(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
net1 = Temporal_Attention_bias_ver2(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net1 = Temporal_Attention_ver3_kernel3(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net1 = Temporal_Attention_bias_ver3_kernel3(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net1 = Temporal_Attention_ver4_None(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net1 = Temporal_Attention_bias_ver4_None(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net1 = Temporal_Attention_ver4_ReLU(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net1 = Temporal_Attention_bias_ver4_ReLU(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net1 = Temporal_Attention_ver4_Sigmoid(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net1 = Temporal_Attention_bias_ver4_Sigmoid(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net1 = Temporal_Attention_ver4_Softmax(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net1 = Temporal_Attention_bias_ver4_Softmax(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)

"""Load action Network"""
only_scene = False
only_action = True
net2 = Temporal_GAP_FC1(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net2 = Temporal_FC1_GAP_ver2(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net2 = Temporal_conv1d(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net2 = Temporal_Attention_ver2(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net2 = Temporal_Attention_bias_ver2(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net2 = Temporal_Attention_ver3_kernel3(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net2 = Temporal_Attention_bias_ver3_kernel3(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net2 = Temporal_Attention_ver4_None(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net2 = Temporal_Attention_bias_ver4_None(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net2 = Temporal_Attention_ver4_ReLU(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net2 = Temporal_Attention_bias_ver4_ReLU(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net2 = Temporal_Attention_ver4_Sigmoid(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net2 = Temporal_Attention_bias_ver4_Sigmoid(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net2 = Temporal_Attention_ver4_Softmax(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net2 = Temporal_Attention_bias_ver4_Softmax(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)



net1 = net1.to(device)
net2 = net2.to(device)
if device == 'cuda':
    # net1 = torch.nn.DataParallel(net1)
    # net2 = torch.nn.DataParallel(net2)
    cudnn.benchmark = True
    # torch.backends.cudnn.enabled=False

if load_weights:
    # Load checkpoint.
    print('==> Resuming from trained model..')
    assert os.path.isfile(load_weights_path_scene), 'Error: no weight file! %s'%(load_weights_path_scene)
    checkpoint = torch.load(load_weights_path_scene)
    net1.load_state_dict(checkpoint['net'])

    assert os.path.isfile(load_weights_path_action), 'Error: no weight file! %s'%(load_weights_path_action)
    checkpoint = torch.load(load_weights_path_action)
    net2.load_state_dict(checkpoint['net'])


"""Load Dataset"""
transform_test = None
testset = Coview_frame_dataset_final(root=test_data_path, train=False, rgb_audio_concat=data_rgb_audio_concat,
                                     transform=transform_test, max_data_temporal_length=max_data_temporal_length)
testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=data_loader_worker_num)

# Test
def test(epoch):
    with open(save_result_path, 'w') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['video_id','scene_01','scene_02','scene_03','scene_04','scene_05','scene_06'
                            ,'scene_07','scene_08','scene_09','scene_10','scene_11','scene_12','scene_13'
                            ,'scene_14','scene_15','scene_16','scene_17','scene_18','scene_19','scene_20'
                            ,'action_01','action_02','action_03','action_04','action_05','action_06','action_07'
                            ,'action_08','action_09','action_10','action_11','action_12','action_13','action_14'
                            ,'action_15','action_16','action_17','action_18','action_19','action_20'])

        with torch.no_grad():
            
            for sub_batch_idx, (inputs, ids) in enumerate(testloader):
                inputs = inputs.to(device)

                inputs = inputs.float()
                inputs = (inputs / 64) - 2

                outputs_scene = net1(inputs)
                outputs_scene = outputs_scene.cpu().numpy()
                scene_topk = sorted(range(len(outputs_scene[0])), key=lambda i: outputs_scene[0][i], reverse=True)[:top_k]

                outputs_action = net2(inputs)
                outputs_action = outputs_action.cpu().numpy()
                action_topk = sorted(range(len(outputs_action[0])), key=lambda i: outputs_action[0][i], reverse=True)[:top_k]


                csv_lists=[]
                csv_lists.append('%s'%(ids))
                for i in range(top_k):
                    csv_lists.append(scene_topk[i] + 1)
                for i in range(top_k):
                    csv_lists.append(action_topk[i] + 1 + 29)

                csv_writer.writerow(csv_lists)


                progress_bar(sub_batch_idx, len(testloader))

test(0)
