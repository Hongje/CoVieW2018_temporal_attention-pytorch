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

model_save_path = 'weights/Temporal_2048_Pooling_only_action/'

only_scene = False
only_action = True # if you want use two label at the same time, then turn off both options
batchsize = 1024
num_epoch = 1000    # It should less than 10000000
base_learning_rate = 0.0002
learning_rate_decay_epoch = 300
learning_rate_decay_rate = 1./3  # It should float & less than 1
model_save_period_epoch = 10
input_gaussian_noise_variance = 0 # 0 is off, mean(abs())==0.7979 when 1.

train_models_num = 16

load_weights = False
load_weights_path = 'weights/Temporal_Attention_ver5_only_scene_/'

Temporal_Attention_load_GAP_FC_weights = False
GAP_FC_weights_path = 'weights/Temporal_GAP_FC1_only_scene_epoch63_acc95.2%/ckpt.pt'
Attention_weights_path = 'weights/Temporal_Attention_bias_ver2_only_scene_epoch100_acc95.3%/ckpt.pt'

temporal_bias_initialization_value = None # None is default
JH_selector_init = False # False if default
JH_feature_dim = 2048 # 1152 is default


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
best_index = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
best_epoch = start_epoch

best_acc_models = []
best_epoch_models = []
for i in range(train_models_num):
    best_acc_models.append(0)
    best_epoch_models.append(0)


if not os.path.isdir(model_save_path):
    os.mkdir(model_save_path)


"""Load Network"""
net = []
for i in range(train_models_num):
    # net.append(Temporal_GAP_FC1(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action))
    # net.append(Temporal_FC1_GAP_ver2(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action))
    # net.append(Temporal_conv1d(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action))
    # net.append(Temporal_Attention_ver2(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action))
    # net.append(Temporal_Attention_bias_ver2(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action, temporal_bias_init = temporal_bias_initialization_value))
    # net.append(Temporal_Attention_ver2_softmax(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action))
    # net.append(Temporal_Attention_bias_ver2_softmax(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action, temporal_bias_init = temporal_bias_initialization_value))
    # net.append(Temporal_Attention_ver3_kernel3(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action))
    # net.append(Temporal_Attention_bias_ver3_kernel3(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action, temporal_bias_init = temporal_bias_initialization_value))
    # net.append(Temporal_Attention_ver4_None(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action))
    # net.append(Temporal_Attention_bias_ver4_None(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action, temporal_bias_init = temporal_bias_initialization_value))
    # net.append(Temporal_Attention_ver4_ReLU(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action))
    # net.append(Temporal_Attention_bias_ver4_ReLU(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action, temporal_bias_init = temporal_bias_initialization_value))
    # net.append(Temporal_Attention_ver4_Sigmoid(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action))
    # net.append(Temporal_Attention_bias_ver4_Sigmoid(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action, temporal_bias_init = temporal_bias_initialization_value))
    # net.append(Temporal_Attention_ver4_Softmax(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action))
    # net.append(Temporal_Attention_bias_ver4_Softmax(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action, temporal_bias_init = temporal_bias_initialization_value))
    # net.append(Temporal_Attention_ver5(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action))
    # net.append(Temporal_Attention_ver6(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action))
    # net.append(Temporal_Attention_bidirection_ver6(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action))
    # net.append(Temporal_Pooling_voting(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action, selector_init = JH_selector_init))
    # net.append(Temporal_N_Pooling_voting(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action, selector_init = JH_selector_init, feature_dim = JH_feature_dim))
    net.append(Temporal_N_Pooling(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action, feature_dim = JH_feature_dim))
    # net.append(MoeModel(data_length=data_length, class_length=class_length, only_scene=only_scene, only_action=only_action, num_mixtures=2))

for i in range(train_models_num):
    net[i] = net[i].to(device)
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
    # torch.backends.cudnn.enabled=False

if load_weights:
    # Load checkpoint.
    print('==> Resuming from trained model..')
    # assert os.path.isfile(load_weights_path), 'Error: no weight file! %s'%(load_weights_path)
    
    for i in range(train_models_num):
        checkpoint = torch.load(os.path.join(load_weights_path, 'weights_idx_%04d.pt'%(i)))
        net[i].load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        best_epoch = start_epoch

if Temporal_Attention_load_GAP_FC_weights:
    print('==> load GAP_FC model weights..')
    assert os.path.isfile(GAP_FC_weights_path), 'Error: no weight file! %s'%(GAP_FC_weights_path)
    assert os.path.isfile(Attention_weights_path), 'Error: no weight file! %s'%(Attention_weights_path)
    checkpoint_FC = torch.load(GAP_FC_weights_path)
    checkpoint_Attention = torch.load(Attention_weights_path)
    for i in range(train_models_num):
        model_dict = net[i].state_dict()
        model_dict.update(checkpoint_Attention['net'])
        pretrained_dict = {k: v for k, v in checkpoint_FC['net'].items() if k in net[i].state_dict()}
        model_dict.update(pretrained_dict) 
        net[i].load_state_dict(model_dict)
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
# training_setting_file.write('model_save_period_epoch = %d\n'%model_save_period_epoch)
training_setting_file.write('load_weights = %r\n'%load_weights)
training_setting_file.write('start_epoch = %d\n'%start_epoch)
training_setting_file.write('input_gaussian_noise_variance = %d\n'%input_gaussian_noise_variance)
training_setting_file.write('train_models_num = %d\n'%train_models_num)
training_setting_file.write('-----------------------------\n')
training_setting_file.write('\n\n')
training_setting_file.close()

criterion = []
for i in range(train_models_num):
    criterion.append(nn.CrossEntropyLoss())
# for i in range(train_models_num):
#     criterion.append(nn.NLLLoss())



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


optimizer = []
scheduler = []
for i in range(train_models_num):
    optimizer.append(torch.optim.Adam(net[i].parameters(), lr=base_learning_rate, weight_decay=weight_l2_regularization))
for i in range(train_models_num):
    scheduler.append(torch.optim.lr_scheduler.StepLR(optimizer[i], step_size=learning_rate_decay_epoch, gamma=learning_rate_decay_rate))



# Training
def train(epoch, learning_rate):
    print('\nEpoch: %d' % epoch)

    train_loss = []
    correct = []
    for i in range(train_models_num):
        net[i].train()
        train_loss.append(0)
        correct.append(0)
        optimizer[i].zero_grad()
    total = 0
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
        
        outputs = []
        for i in range(train_models_num):
            outputs.append(net[i](inputs))
        
        loss = []
        for i in range(train_models_num):
            loss.append(criterion[i](outputs[i], targets) / batchsize)

        for i in range(train_models_num):
            loss[i].backward()

        predicted = []
        for i in range(train_models_num):
            train_loss[i] += loss[i].item()
            _, predicted_tmp = outputs[i].max(1)
            predicted.append(predicted_tmp)
        del loss

        total += targets.size(0)
        for i in range(train_models_num):
            correct[i] += predicted[i].eq(targets).sum().item()
        del predicted

        if (sub_batch_idx+1) % batchsize == 0:
            for i in range(train_models_num):
                optimizer[i].step()
                optimizer[i].zero_grad()

            train_loss_mean = sum(train_loss) / float(len(train_loss))
            correct_mean = sum(correct) / float(len(correct))

            progress_bar(sub_batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss_mean * batchsize / (sub_batch_idx + 1), 100. * correct_mean / total, int(correct_mean), total))
    
    if (sub_batch_idx+1) % batchsize != 0:
        for i in range(train_models_num):
            optimizer[i].step()
            optimizer[i].zero_grad()

        train_loss_mean = sum(train_loss) / float(len(train_loss))
        correct_mean = sum(correct) / float(len(correct))

        progress_bar(sub_batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss_mean * batchsize / (sub_batch_idx + 1), 100. * correct_mean / total, int(correct_mean), total))



# Test
def test(epoch):
    global best_acc
    global best_index
    global best_epoch
    global best_acc_models
    global best_epoch_models

    test_loss = []
    correct = []
    for i in range(train_models_num):
        net[i].eval()
        test_loss.append(0)
        correct.append(0)
    total = 0
    with torch.no_grad():
        
        progress_bar(0, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (0, 0, 0, 0))
        for sub_batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            inputs = inputs.float()
            inputs = (inputs / 64) - 2
            
            outputs = []
            for i in range(train_models_num):
                outputs.append(net[i](inputs))

            loss = []
            for i in range(train_models_num):
                loss.append(criterion[i](outputs[i], targets) / batchsize)

            predicted = []
            for i in range(train_models_num):
                test_loss[i] += loss[i].item()
                _, predicted_tmp = outputs[i].max(1)
                predicted.append(predicted_tmp)
            del loss

            total += targets.size(0)
            for i in range(train_models_num):
                correct[i] += predicted[i].eq(targets).sum().item()
            del predicted

            if (sub_batch_idx+1) % batchsize == 0:
                test_loss_mean = sum(test_loss) / float(len(test_loss))
                correct_mean = sum(correct) / float(len(correct))

                progress_bar(sub_batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss_mean * batchsize / (sub_batch_idx + 1), 100. * correct_mean / total, int(correct_mean), total))

        if (sub_batch_idx+1) % batchsize != 0:
            test_loss_mean = sum(test_loss) / float(len(test_loss))
            correct_mean = sum(correct) / float(len(correct))

            progress_bar(sub_batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss_mean * batchsize / (sub_batch_idx + 1), 100. * correct_mean / total, int(correct_mean), total))
                  


    # Save checkpoint.
    acc = []
    for i in range(train_models_num):
        acc.append(100. * correct[i] / total)

    for i in range(train_models_num):
        if best_acc_models[i] < acc[i]:
            best_acc_models[i] = acc[i]
            best_epoch_models[i] = epoch

    max_acc_index = acc.index(max(acc))
    max_acc_value = max(acc)

    # if ((epoch+1) % model_save_period_epoch) == 0:
    #     print('Saving... periodically')
    #     state = {
    #         'net': net.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     torch.save(state, os.path.join(model_save_path, 'weights_%07d.pt'%(epoch+1)))
    
    if max_acc_value > best_acc:
        print('Saving... best test accuracy')
        state = {
            'net': net[max_acc_index].state_dict(),
            'acc': max_acc_value,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(model_save_path, 'ckpt.pt'))
        # torch.save(state, os.path.join(model_save_path, 'weights_%07d.pt'%(epoch+1)))
        best_acc = max_acc_value
        best_index = max_acc_index
        best_epoch = epoch

    if ((epoch+1) % model_save_period_epoch) == 0:
        for i in range(train_models_num):
            state = {
                'net': net[i].state_dict(),
                'acc': acc[i],
                'epoch': epoch,
            }
            torch.save(state, os.path.join(model_save_path, 'weights_idx_%04d.pt'%(i)))
            torch.save(state, os.path.join(model_save_path, 'weights_idx_%04d_epoch_%04d.pt'%(i, epoch)))


    print('The best test accuracy: %f  epoch: %d  index: %d'%(best_acc, best_epoch, best_index))
    for i in range(train_models_num):
        print('model number: %03d | loss: %.3f | best acc: %.3f%%  current acc: %.3f%%  (%d/%d) | best epoch: %d'%(i, (test_loss[i] * batchsize / (sub_batch_idx + 1)), best_acc_models[i], acc[i], correct[i], total, best_epoch_models[i]))


for epoch in range(start_epoch, start_epoch+num_epoch):
    for i in range(train_models_num):
        scheduler[i].step()
    train(epoch, base_learning_rate)
    test(epoch)