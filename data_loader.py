# coding: utf-8

import torch
from torch.utils.data import Dataset

import torchvision

import numpy as np
import os
import sys



class Coview_frame_dataset(Dataset):
    """ Coview 2018 frame Dataset
    This code was created with reference to pytorch's CIFAR10 DATASET code
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR10
    """
    train_list = [
        'coview_frame_train_0.npz',
        'coview_frame_train_1.npz',
        'coview_frame_train_2.npz',
        'coview_frame_train_3.npz',
        'coview_frame_train_4.npz',
        'coview_frame_train_5.npz',
        'coview_frame_train_6.npz',
        'coview_frame_train_7.npz',
        'coview_frame_train_8.npz',
        'coview_frame_train_9.npz',
    ]
    test_list = [
        'coview_frame_val.npz'
    ]
    def __init__(self, root, train=True, rgb_audio_concat=True, transform=None,
                 target_transform=None, max_data_temporal_length=300,
                 only_scene=False, only_action=False):
        self.rgb_audio_concat = rgb_audio_concat
        self.max_data_temporal_length = max_data_temporal_length
        self.only_scene = only_scene
        self.only_action = only_action

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        assert not(self.only_scene & self.only_action), "The 'only_scene' and 'only_action' options can not be turned on at the same time."

        # if not self._check_integrity():
        #     raise RuntimeError('Dataset not found or corrupted.' +
        #                        ' You can use download=True to download it')

        if self.train:
            self.train_data_rgb = []
            self.train_data_audio = []
            self.train_labels = []
            # self.train_ids=[]
            for fentry in self.train_list:
                f = fentry
                file = os.path.join(self.root, f)
                fo = np.load(file, encoding='bytes')
                self.train_data_rgb.append(fo['rgb'])
                self.train_data_audio.append(fo['audio'])
                self.train_labels.append(fo['labels'])

                # self.train_ids.append(fo['ids'])

                fo.close()

            self.train_data_rgb = np.concatenate(self.train_data_rgb)
            self.train_data_audio = np.concatenate(self.train_data_audio)
            self.train_labels = np.concatenate(self.train_labels)
            self.train_labels = self.train_labels - 1

            # for i in range(self.train_data_rgb.size):
            #     self.train_data_rgb[i] = np.float32(self.train_data_rgb[i])
            #     self.train_data_audio[i] = np.float32(self.train_data_audio[i])
            #     self.train_data_rgb[i] = (self.train_data_rgb[i] / 64) - 2
            #     self.train_data_audio[i] = (self.train_data_audio[i] / 64) - 2

            # self.train_ids=np.concatenate(self.train_ids)


        else:
            f = self.test_list[0]
            file = os.path.join(self.root, f)
            fo = np.load(file, encoding='bytes')
            self.test_data_rgb = fo['rgb']
            self.test_data_audio = fo['audio']
            self.test_labels = fo['labels']
            self.test_labels = self.test_labels - 1
            fo.close()

            # for i in range(self.test_data_rgb.size):
            #     self.test_data_rgb[i] = np.float32(self.test_data_rgb[i])
            #     self.test_data_audio[i] = np.float32(self.test_data_audio[i])
            #     self.test_data_rgb[i] = (self.test_data_rgb[i] / 64) - 2
            #     self.test_data_audio[i] = (self.test_data_audio[i] / 64) - 2


        # self.data_path = npz_path
        # self.npz_file = np.load(self.data_path)
        # self.length = len(self.npz_file['ids'])

    def __getitem__(self, index):

        # label = self.npz_file['labels'][idx]
        # rgb = self.npz_file['rgb'][idx]
        # audio = self.npz_file['audio'][idx]
        #
        # label = torch.FloatTensor(label).unsqueeze(0)
        # rgb = torch.FloatTensor(rgb).unsqueeze(0)
        # audio = torch.FloatTensor(audio).unsqueeze(0)
        #
        # return {'gt': label, 'rgb': rgb, 'audio': audio}

        if self.train:
            # rgb = np.float32(self.train_data_rgb[index])
            # audio = np.float32(self.train_data_audio[index])
            # rgb = (rgb / 64) - 2
            # audio = (audio / 64) - 2
            rgb = self.train_data_rgb[index]
            audio = self.train_data_audio[index]
            target = self.train_labels[index]

            # ids=self.train_ids[index]
        else:
            # rgb = np.float32(self.test_data_rgb[index])
            # audio = np.float32(self.test_data_audio[index])
            # rgb = (rgb / 64) - 2
            # audio = (audio / 64) - 2
            rgb = self.test_data_rgb[index]
            audio = self.test_data_audio[index]
            target = self.test_labels[index]

        if rgb.shape[1] != 1024:
            rgb = np.squeeze(rgb, axis = 0)
            audio = np.squeeze(audio, axis = 0)

        # data_length = len(rgb)

        # check_tmp = False
        # if data_length == 1:
        # print('id:',ids)
        #     print('1 - len:', data_length, 'rgb:', rgb.shape, 'audio:', audio.shape)
        #     check_tmp = True
        # print('1 - len:', data_length, 'rgb:', rgb.shape, 'audio:', audio.shape)
        # if data_length == 1:
        #     rgb = np.squeeze(rgb)
        #     audio = np.squeeze(audio)
        #     data_length = len(rgb)
        
        # if data_length < self.max_data_temporal_length:
        #     rgb = np.pad(rgb, ((0, self.max_data_temporal_length - data_length), (0, 0)), 'constant',constant_values=128)
        #     audio = np.pad(audio, ((0, self.max_data_temporal_length - data_length), (0, 0)), 'constant',constant_values=128)
        # elif data_length == 301:
        #     rgb = np.delete(rgb,-1,0)
        #     audio = np.delete(audio, -1, 0)
        #     data_length = 300

        # if check_tmp:
        #     print('2 - len:', data_length, 'rgb:', rgb.shape, 'audio:', audio.shape)
        #     if data_length==1024:
        #         print(rgb)
        #         print(audio)
        #         print(target)
        # print('2 - len:', data_length, 'rgb:', rgb.shape, 'audio:', audio.shape)



        if self.only_scene:
            target = target[0]
        elif self.only_action:
            target = target[1] - 29

        if self.rgb_audio_concat:
            data = np.concatenate((rgb, audio), axis=1)

            if self.transform is not None:
                data = self.transform(data)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return data, target
        else:
            if self.transform is not None:
                rgb = self.transform(rgb)
                audio = self.transform(audio)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return rgb, audio, target


    def __len__(self):
        if self.train:
            return len(self.train_labels)
        else:
            return len(self.test_labels)

    # def _check_integrity(self):
    #     root = self.root
    #     for fentry in (self.train_list + self.test_list):
    #         filename, md5 = fentry[0], fentry[1]
    #         fpath = os.path.join(root, self.base_folder, filename)
    #         if not check_integrity(fpath, md5):
    #             return False
    #     return True


class Coview_frame_dataset_final(Dataset):
    """ Coview 2018 frame Dataset
    This code was created with reference to pytorch's CIFAR10 DATASET code
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR10
    """
    train_list = [
        'coview_frame_train_0.npz',
        'coview_frame_train_1.npz',
        'coview_frame_train_2.npz',
        'coview_frame_train_3.npz',
        'coview_frame_train_4.npz',
        'coview_frame_train_5.npz',
        'coview_frame_train_6.npz',
        'coview_frame_train_7.npz',
        'coview_frame_train_8.npz',
        'coview_frame_train_9.npz',
        'coview_frame_val.npz'
    ]
    test_list = [
        'coview_frame_test_nolabel.npz'
    ]
    def __init__(self, root, train=True, rgb_audio_concat=True, transform=None,
                 target_transform=None, max_data_temporal_length=300,
                 only_scene=False, only_action=False):
        self.rgb_audio_concat = rgb_audio_concat
        self.max_data_temporal_length = max_data_temporal_length
        self.only_scene = only_scene
        self.only_action = only_action

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        assert not(self.only_scene & self.only_action), "The 'only_scene' and 'only_action' options can not be turned on at the same time."

        if self.train:
            self.train_data_rgb = []
            self.train_data_audio = []
            self.train_labels = []
            # self.train_ids=[]
            for fentry in self.train_list:
                f = fentry
                file = os.path.join(self.root, f)
                fo = np.load(file, encoding='bytes')
                self.train_data_rgb.append(fo['rgb'])
                self.train_data_audio.append(fo['audio'])
                self.train_labels.append(fo['labels'])
                # self.train_ids.append(fo['ids'])

                fo.close()

            self.train_data_rgb = np.concatenate(self.train_data_rgb)
            self.train_data_audio = np.concatenate(self.train_data_audio)
            self.train_labels = np.concatenate(self.train_labels)
            self.train_labels = self.train_labels - 1

        else:
            f = self.test_list[0]
            file = os.path.join(self.root, f)
            fo = np.load(file, encoding='bytes')
            self.test_data_rgb = fo['rgb']
            self.test_data_audio = fo['audio']
            # self.test_labels = fo['labels']
            # self.test_labels = self.test_labels - 1
            self.test_ids = fo['ids']
            fo.close()

    def __getitem__(self, index):

        if self.train:
            rgb = self.train_data_rgb[index]
            audio = self.train_data_audio[index]
            target = self.train_labels[index]
            # ids=self.train_ids[index]

        else:
            rgb = self.test_data_rgb[index]
            audio = self.test_data_audio[index]
            # target = self.test_labels[index]
            ids = self.test_ids[index]

        if rgb.shape[1] != 1024:
            rgb = np.squeeze(rgb, axis=0)
            audio = np.squeeze(audio, axis=0)


        if self.train:
            if self.only_scene:
                target = target[0]
            elif self.only_action:
                target = target[1] - 29


        if self.rgb_audio_concat:
            data = np.concatenate((rgb, audio), axis=1)

            if self.transform is not None:
                data = self.transform(data)

            if self.train:
                if self.target_transform is not None:
                    target = self.target_transform(target)

                return data, target
            else:
                return data, ids
        else:
            if self.transform is not None:
                rgb = self.transform(rgb)
                audio = self.transform(audio)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return rgb, audio, target


    def __len__(self):
        if self.train:
            return len(self.train_labels)
        else:
            return len(self.test_ids)
