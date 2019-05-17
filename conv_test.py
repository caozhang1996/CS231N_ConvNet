#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 15:33:25 2019

@author: caozhang
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from cs231n.layer import *

def relative_error(x, y):
    """
    returns relative error
    """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

# Test the conv
x_shape = (2, 3, 4, 4)
w_shape = (3, 3, 4, 4) 
x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape) 
w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape) 
b = np.linspace(-0.1, 0.2, num=3) 
conv_param = {'stride': 2, 'pad': 1} 
out, _ = conv_forward_naive(x, w, b, conv_param)
correct_out = np.array([[[[-0.08759809, -0.10987781], 
                           [-0.18387192, -0.2109216 ]], 
                          [[ 0.21027089, 0.21661097], 
                           [ 0.22847626, 0.23004637]], 
                          [[ 0.50813986, 0.54309974], 
                           [ 0.64082444, 0.67101435]]], 
                          [[[-0.98053589, -1.03143541],
                            [-1.19128892, -1.24695841]], 
                           [[ 0.69108355, 0.66880383], 
                            [ 0.59480972, 0.56776003]], 
                           [[ 2.36270298, 2.36904306], 
                            [ 2.38090835, 2.38247847]]]])

print ('Testing conv_forward_naive')
print ('difference: ', relative_error(out, correct_out))


# Test the max pool
x_shape = (2, 3, 4, 4)
x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)
pool_param = {'pool_width': 2, 'pool_height': 2, 'stride': 2}

out, _ = max_pool_forward_naive(x, pool_param)

correct_out = np.array([[[[-0.26315789, -0.24842105],
                          [-0.20421053, -0.18947368]],
                         [[-0.14526316, -0.13052632],
                          [-0.08631579, -0.07157895]],
                         [[-0.02736842, -0.01263158],
                          [ 0.03157895,  0.04631579]]],
                        [[[ 0.09052632,  0.10526316],
                          [ 0.14947368,  0.16421053]],
                         [[ 0.20842105,  0.22315789],
                          [ 0.26736842,  0.28210526]],
                         [[ 0.32631579,  0.34105263],
                          [ 0.38526316,  0.4       ]]]])

print('Testing max_pool_forward_naive function:')
print('difference: ', relative_error(out, correct_out))

