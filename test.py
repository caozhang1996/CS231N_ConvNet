#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 21:35:32 2019

@author: caozhang
"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mimge

from cs231n.im2col import *

x = np.array([[[[1, 2, 3], 
                [4, 5, 6], 
                [7, 8, 9]],
               [[10, 11, 12],
                [13, 14, 15],
                [16, 17, 18]],
               [[19, 20, 21],
                [22, 23, 24],
                [25, 26, 27]]
               ]])
N, C, H, W = x.shape
field_height=3
field_width=3
stride = 1
padding = 1
out_height = (H + 2 * padding - field_height) / stride + 1 # out_height=3
out_width = (W + 2 * padding - field_width) / stride + 1   # out_width=3

i0 = np.repeat(np.arange(field_height), field_width)
i0 = np.tile(i0, C)
i1 = stride * np.repeat(np.arange(field_height), field_width)
j0 = np.tile(np.arange(field_width), field_height * C)
j1 = stride * np.tile(np.arange(out_width), out_height)
(k, i, j) = get_im2col_indices(x.shape, field_height=3, field_width=3)
cols = im2col_indices(x, field_height=3, field_width=3)

print ('i0:')
print (i0)
print ('i1:')
print (i1)
print ('j0:')
print (j0)
print ('j1:')
print (j1)

print ('k:')
print (k)
print ('i:')
print (i)
print ('j:')
print (j)

print (cols)
