#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 11:03:38 2019

@author: caozhang
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

def affine_forward(x, w, b):
    """
    affine:仿射
    Inputs:
        x: A numpy arrary containing input data, of shape(N, D)
        w: A numpy arrary of weights, of shape(D, M) 权重
        b: A numpy arrary of biases, of shape(M,) 偏值
    Outputs:
        out: output, of shape (N, M)
        cache: (x, w, b), cache: 隐藏
    """
    out = None
    flatten_x = np.reshape(x, (x.shape[0], -1))
    out = flatten_x.dot(w) + b            # out = W x + b: 矩阵乘法
    cache = (x, w, b)                     # 返回为后面其他操作所用
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.
    Inputs: 
        - dout: Upstream derivative, of shape (N, M) 上一层的导数 
        - cache: Tuple of: 
        - x: Input data, of shape (N, d_1, ... d_k) 
        - w: Weights, of shape (D, M) 
        - b: biases, of shape (M,) 
    Returns a tuple of: 
        - dx: Gradient with respect to x, of shape (N, d1, ..., d_k) 
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    flatten_x = np.reshape(x, (x.shape[0], -1))
    dx = dout.dot(w.T)
    dw = (flatten_x.T).dout(dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db


def relu_forward(x):
    """
    Compute the forword pass for a layer of rectified linear units(ReLUs)
    Input:
        x: Input, of any shape
    Outputs:
        out: output, of the same shape as x 
        cache: x
    """
    out = None
    out = np.maximum(0, x)   # 得到一个形状和x一样的矩阵, x中比0小的元素都替换成0
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units(Relus:修正线性单元)
    Inputs:
        - dout: Upstream derivative, of any shape
        - cache: Input x, of same shape as dout
    Returns:
        dx: Gradient with respect to x
    """
    dx, x = None, cache
    dx = dout
    dx[x < 0] = 0
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.
    Inputs:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
            - 'stride'
            - 'pad'
    Returns:
        - out: Output data, of shape (N, F, H', W') where H' and W' are given by
               H' = 1 + (H + 2 * pad - HH) / stride
               W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
    """
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape 
    stride = conv_param['stride']
    pad = conv_param['pad']
    
    # 计算卷积结果矩阵的大小并全部分配零值
    new_H = 1 + int((H + 2 * pad - HH) / stride)
    new_W = 1 + int((W + 2 * pad - WW) / stride)
    out = np.zeros([N, F, new_H, new_W])
    
    #卷积开始
    for n in range(N):
        for f in range(F):
            # 临时输出值，先加上偏移项b[f]
            conv_newH_newW = np.ones([new_H, new_W]) * b[f]
            
            for c in range(C):
                padded_x = np.lib.pad(x[n, c], pad_width=pad, mode='constant', constant_values=0)
                for i in range(new_H):
                    for j in range(new_W):
                        conv_newH_newW[i, j] += np.sum(padded_x[i*stride:i*stride+HH, j*stride:j*stride+HH]*
                                                       w[f, c, :, :])
            out[n, f] = conv_newH_newW
            
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.
    Inputs:
        - dout: Upstream derivatives
        - cache: a tuple of (x, w, b, conv_param) as in conv_forward_naive
    Returns a tuple of:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b
    """
    x, w, b, conv_param = cache
    pad = conv_param['pad']
    stride = conv_param['stride']
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    N, F, new_H, new_W = dout.shape
    
    # 求dx，要先求对填充了的x的导数
    padded_x = np.lib.pad(x, pad_width=pad, mode='constant', constant_values=0)
    padded_dx = np.zeros_like(padded_x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    
    for n in range(N):       #第n张图片
        for f in range(F):   #第f个卷积核 
            for i in range(new_H):
                for j in range(new_W):
                    db[f] += dout[n, f, i, j]    # dg对db求导为1*dout
                    dw[f] += padded_x[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] * dout[n, f, i, j]
                    padded_dx[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] += w[f] * dout[n, f, i, j]
    # 去掉填充部分
    dx = padded_dx[:, :, pad:pad+H, pad:pad+W]
    return dx, dw, db

                     
def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the backward pass for a max pooling layer.
    Inputs:
        - x: Input data, of shape (N, C, H, W)
         - pool_param: A dictionary with the following keys:
            - 'pool_height'
            - 'pool_width'
            - 'stride'
    Returns a tuple of:
        - out: Output data
        - cache: (x, pool_param)
    """
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    
    # 计算最大池化结果矩阵
    new_H = 1 + int((H - pool_height) / stride)
    new_W = 1 + int((W - pool_width) / stride)
    out = np.zeros([N, C, new_H, new_W])
    
    for n in range(N):
        for c in range(C):
            for i in range(new_H):
                for j in range(new_W):
                    out[n, c, i, j] = np.max(x[n, c, i*stride:i*stride+pool_height, 
                                               j*stride:j*stride+pool_width])
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.
    Inputs:
        - dout: Upstream derivatives
        - cache: a tuple of (x, pool_param) as in the max_pool_forward_naive
    Returns a tuple of:
        - dx: Gradient with respect to x
    """
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    
    new_H = 1 + int((H - pool_height) / stride)
    new_W = 1 + int((W - pool_width) / stride)
    dx = np.zeros_like(x)
    
    for n in range(N):
        for c in range(C):
            for i in range(new_H):
                for j in range(new_W):
                    window = x[n, c, i*stride:i*stride+pool_height, j*stride:j*stride+pool_width]
                    # window == np.max(window):window中最大值为1，其余之为0
                    dx[n, c, i*stride:i*stride+pool_height, 
                       j*stride:j*stride+pool_width] = (window == np.max(window)) * dout[n, c, i, j] 
    return dx

        