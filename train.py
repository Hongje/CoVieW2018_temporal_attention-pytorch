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

model_save_path = 'weights/Temporal_Attention_bias_ver4_Softmax/'

only_scene = True
only_action = False # if you want use two label at the same time, then turn off both options
batchsize = 1024
num_epoch = 1000    # It should less than 10000000
base_learning_rate = 0.0001
learning_rate_decay_epoch = 300
learning_rate_decay_rate = 1./3  # It should float & less than 1
model_save_period_epoch = 50
input_gaussian_noise_variance = 0 # 0 is off, mean(abs())==0.7979 when 1.

load_weights = False
load_weights_path = 'weights/Temporal_GAP_FC1_only_scene_epoch63_acc95.2%/ckpt.pt'

Temporal_Attention_load_GAP_FC_weights = False
GAP_FC_weights_path = 'weights/Temporal_GAP_FC1_only_scene_epoch63_acc95.2%/ckpt.pt'
Attention_weights_path = 'weights/Temporal_Attention_bias_ver2_only_scene_epoch100_acc95.3%/ckpt.pt'


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


if not os.path.isdir(model_save_path):
    os.mkdir(model_save_path)



"""Load Network"""
# net = Temporal_GAP_FC1(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net = Temporal_FC1_GAP_ver2(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net = Temporal_conv1d(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net = Temporal_Attention_ver2(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net = Temporal_Attention_bias_ver2(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net = Temporal_Attention_ver3_kernel3(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net = Temporal_Attention_bias_ver3_kernel3(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net = Temporal_Attention_ver4_None(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net = Temporal_Attention_bias_ver4_None(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net = Temporal_Attention_ver4_ReLU(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net = Temporal_Attention_bias_ver4_ReLU(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net = Temporal_Attention_ver4_Sigmoid(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net = Temporal_Attention_bias_ver4_Sigmoid(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
# net = Temporal_Attention_ver4_Softmax(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)
net = Temporal_Attention_bias_ver4_Softmax(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action)


net = net.to(device)
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    # torch.backends.cudnn.enabled=False

if load_weights:
    # Load checkpoint.
    print('==> Resuming from trained model..')
    assert os.path.isfile(load_weights_path), 'Error: no weight file! %s'%(load_weights_path)
    checkpoint = torch.load(load_weights_path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    best_epoch = start_epoch

if Temporal_Attention_load_GAP_FC_weights:
    print('==> load GAP_FC model weights..')
    assert os.path.isfile(GAP_FC_weights_path), 'Error: no weight file! %s'%(GAP_FC_weights_path)
    assert os.path.isfile(Attention_weights_path), 'Error: no weight file! %s'%(Attention_weights_path)
    checkpoint_FC = torch.load(GAP_FC_weights_path)
    checkpoint_Attention = torch.load(Attention_weights_path)
    model_dict = net.state_dict()
    model_dict.update(checkpoint_Attention['net'])
    pretrained_dict = {k: v for k, v in checkpoint_FC['net'].items() if k in net.state_dict()}
    model_dict.update(pretrained_dict) 
    net.load_state_dict(model_dict)
    # net.state_dict()['module.fc1.bias']
    # net.state_dict()['module.attention.bias']


training_setting_file = open(os.path.join(model_save_path,'training_settings.txt'), 'a')
training_setting_file.write('----- training options ------\n')
training_setting_file.write('only_scene = %r\n'%only_scene)
training_setting_file.write('only_action = %r\n'%only_action)
training_setting_file.write('batchsize = %d\n'%batchsize)
training_setting_file.write('num_epoch = %d\n'%num_epoch)
training_setting_file.write('base_learning_rate = %f\n'%base_learning_rate)
training_setting_file.write('learning_rate_decay_epoch = %d\n'%learning_rate_decay_epoch)
training_setting_file.write('learning_rate_decay_rate = %f\n'%learning_rate_decay_rate)
training_setting_file.write('model_save_period_epoch = %d\n'%model_save_period_epoch)
training_setting_file.write('load_weights = %r\n'%load_weights)
training_setting_file.write('start_epoch = %d\n'%start_epoch)
training_setting_file.write('input_gaussian_noise_variance = %d\n'%input_gaussian_noise_variance)
training_setting_file.write('-----------------------------\n')
training_setting_file.write('\n\n')
training_setting_file.close()

criterion = nn.CrossEntropyLoss()



"""Load Dataset"""
# transform_train = transforms.Compose([
#     # transforms.RandomCrop(32, padding=4),
#     # transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
#
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
transform_train = None
transform_test = None

trainset = Coview_frame_dataset(root=train_data_path, train=True, rgb_audio_concat=data_rgb_audio_concat,
                                transform=transform_train, max_data_temporal_length=max_data_temporal_length,
                                only_scene=only_scene, only_action=only_action)
trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=data_loader_worker_num)

testset = Coview_frame_dataset(root=test_data_path, train=False, rgb_audio_concat=data_rgb_audio_concat,
                               transform=transform_test, max_data_temporal_length=max_data_temporal_length,
                               only_scene=only_scene, only_action=only_action)
testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=data_loader_worker_num)



optimizer = torch.optim.Adam(net.parameters(), lr=base_learning_rate, weight_decay=weight_l2_regularization)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=learning_rate_decay_epoch, gamma=learning_rate_decay_rate)


# Training
def train(epoch, learning_rate):
    print('\nEpoch: %d' % epoch)

    net.train()
    train_loss = 0
    correct = 0
    total = 0
    optimizer.zero_grad()
    progress_bar(0, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (0, 0, 0, 0))
    for sub_batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        inputs = inputs.float()
        inputs = (inputs / 64) - 2
        if input_gaussian_noise_variance > 0:
            gaussian_noise = torch.randn(inputs.shape) * input_gaussian_noise_variance
            gaussian_noise = gaussian_noise.to(device)
            inputs = inputs + gaussian_noise
        
        outputs = net(inputs)
        loss = criterion(outputs, targets) / batchsize
        loss.backward()

        train_loss += float(loss.item())
        _, predicted = outputs.max(1)

        total += targets.size(0)
        correct += float(predicted.eq(targets).sum().item())

        if (sub_batch_idx+1) % batchsize == 0:
            optimizer.step()
            optimizer.zero_grad()

            progress_bar(sub_batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss * batchsize / (sub_batch_idx + 1), 100. * correct / total, correct, total))
    
    if (sub_batch_idx+1) % batchsize != 0:
        optimizer.step()
        optimizer.zero_grad()

        progress_bar(sub_batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss * batchsize / (sub_batch_idx + 1), 100. * correct / total, correct, total))



# Test
def test(epoch):
    global best_acc
    global best_epoch
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        
        progress_bar(0, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (0, 0, 0, 0))
        for sub_batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            inputs = inputs.float()
            inputs = (inputs / 64) - 2

            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += float(loss.item())
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += int(predicted.eq(targets).sum().item())

            if (sub_batch_idx+1) % batchsize == 0:
                progress_bar(sub_batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss / (sub_batch_idx + 1), 100. * correct / total, correct, total))

        if (sub_batch_idx+1) % batchsize != 0:
            progress_bar(sub_batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss / (sub_batch_idx + 1), 100. * correct / total, correct, total))
                  


    # Save checkpoint.
    acc = 100. * correct / total


    if ((epoch+1) % model_save_period_epoch) == 0:
        print('Saving... periodically')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(model_save_path, 'weights_%07d.pt'%(epoch+1)))
    
    if acc > best_acc:
        print('Saving... best test accuracy')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(model_save_path, 'ckpt.pt'))
        best_acc = acc
        best_epoch = epoch

    print('The best test accuracy: %f  epoch: %d'%(best_acc, best_epoch))


for epoch in range(start_epoch, start_epoch+num_epoch):
    scheduler.step()
    train(epoch, base_learning_rate)
    test(epoch)
