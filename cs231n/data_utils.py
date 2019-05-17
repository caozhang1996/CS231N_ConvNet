#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 12:40:13 2019

@author: caozhang
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import os
import pickle
import platform


def load_pickle(f):
    """
    """
    version = platform.python_version_tuple
    if version[0] == '2':
        return pickle.load(f)
    elif version[0] == '3':
        return pickle.load(f, encoding='latin1')
    raise ValueError('Invalid python version: {}'.format(version))
    
    
def load_CIFAR10_batch(file_name):
    """
    load single cifar batch
    """
    with open(file_name, 'rb') as f:
        data_dict = load_pickle(f)
        X = data_dict['data']
        Y = data_dict['labels']
        # 以float类型读取便于数值计算
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float') # data_dict['data']对应的值应该是10000张图像的平铺
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(root_dir):
    """
    load all cifar batch
    """
    Xtr = []
    Ytr = []
    for i in range(1, 6):
        file_name = os.path.join(root_dir, 'data_bacth_%d' % i)
        X, Y = load_CIFAR10_batch(file_name)
        Xtr.append(X)          # shape of Xtr: [5, 10000, 32, 32, 3]
        Ytr.append(Y)
    Xtr = np.concatenate(Xtr)  #shape of Xtr: [50000, 32, 32, 3]
    Ytr = np.concatenate(Ytr)
    Xte, Yte = load_CIFAR10_batch(os.path.join(root_dir, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000,
                     substract_mean=True):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. 
    """
    root_dir = '../dataset/cifar-10-batches-py'
    X_train , y_train, X_test, y_test = load_CIFAR10(root_dir)
    
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]
    
    # Normalize the data: subtract the mean image
    if substract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image
         
    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()
    
    return {'X_train': X_train, 
            'y_train': y_train,
            'X_val': X_val, 
            'y_val': y_val,
            'X_test': X_test, 
            'y_test': y_test,
            }

