#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    @File:         dataset_train.py
    @Author:       haifeng
    @Since:        python3.9
    @Version:      V1.0
    @Date:         2022/1/6 21:21
    @Description:  
-------------------------------------------------
    @Change:
        2022/1/6 21:21
-------------------------------------------------
"""
import numpy
import torch.utils.data
import torch
from torchvision import transforms
# from scipy.ndimage import imread
import os
import os.path
import glob
import pandas
from sklearn import preprocessing

# def make_dataset(root, train=True):
#     """
#     read the dataset
#     """
#     dataset = []
#
#     if train:
#         dirgt = os.path.join(root, 'train_data/groundtruth')
#         dirimg = os.path.join(root, 'train_data/imgs')
#
#         for fGT in glob.glob(os.path.join(dirgt, '*.jpg')):
#             # for k in range(45)
#             fName = os.path.basename(fGT)
#             fImg = 'train_ori' + fName[8:]
#             dataset.append([os.path.join(dirimg, fImg), os.path.join(dirgt, fName)])
#
#     return dataset

shape = [32, 32, 4]


def get_dataset(matrix):
    dataset = []
    for s in matrix:
        a = s.strip().split(' ')
        col = []
        for b in a:
            row = []
            b = float(b)
            row.append(b)
            col.append(row)
        dataset.append(numpy.array([col]).reshape((shape[0], shape[1], shape[2])))
    return dataset


class DatasetTrain(torch.utils.data.Dataset):
    """
    children class
    """

    def __init__(self, x_train, y_train, transform=None, train=True):
        """
        initialize each parameter
        """
        super(DatasetTrain, self).__init__()
        self.train = train
        # self.train_set_path = make_dataset(root, train)
        # file_name = root
        # # dataframe = pandas.read_csv(file_name)
        # self.X = numpy.array(get_dataset(dataframe['X']))
        # self.Y = dataframe['Y']
        for i in range(0, len(x_train)):
            x_train[i] = numpy.array(x_train[i]).reshape(7, 1, 1023)
        self.X = numpy.array(x_train)
        self.Y = numpy.array(y_train)

        encoder = preprocessing.LabelEncoder()
        self.Y = encoder.fit_transform(self.Y)

    def __getitem__(self, index):
        """
        get the data
        return [img, label]
        idx: read the data one by one
        """
        if self.train:
            # img_path, gt_path = self.train_set_path[index]
            # img = imread(img_path)
            # img = np.atleast_3d(img).transpose(2, 0, 1).astype(np.float32)
            # img = (img - img.min()) / (img.max() - img.min())
            # img = torch.from_numpy(img).float()
            '''
            img_path: custom the path according the data
            then convert to float
            '''

            # gt = imread(gt_path)
            # gt = np.atleast_3d(gt).transpose(2, 0, 1)
            # gt = gt / 255.0
            # gt = torch.from_numpy(gt).float()
            '''
            read gt
            if it is a classify problem, it could name as 0 or 1 according the folder
            '''

            # return img, gt
            return self.X[index], self.Y[index]

    def __len__(self):
        """
        return the length
        """
        # return len(self.train_set_path)
        return len(self.X)
