#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 21:04:00 2019

@author: caozhang
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from cs231n.layer import *
from scipy.misc import imread, imresize, imshow
import numpy as np
import matplotlib.pyplot as plt

kitten = imread('./pictures/kitten.jpg')
puppy = imread('./pictures/puppy.jpg')
lena = imread('./pictures/lena.jpg')

# resize the picture and put them in a data matrix
img_size = 200
x = np.zeros((3, 3, img_size, img_size))
kitten_resize = imresize(kitten, (img_size, img_size))
puppy_resize = imresize(puppy, (img_size, img_size))
lena_resize = imresize(lena, (img_size, img_size))


def image_via_conv():
    """
    """
    x[0, :, :, :] = kitten_resize.transpose(2, 0, 1)
    x[1, :, :, :] = puppy_resize.transpose(2, 0, 1)
    x[2, :, :, :] = lena_resize.transpose(2, 0, 1)
    
    # Set up a convolutional weights holding 2 filters, each 3x3
    w = np.zeros((4, 3, 3, 3))
    
    # The first filter converts the image to grayscale. 
    # Set up the red, green, and blue channels of the filter. 
    w[0, 0, :, :] = [[0, 0, 0], [0, 0.3, 0], [0, 0, 0]] 
    w[0, 1, :, :] = [[0, 0, 0], [0, 0.6, 0], [0, 0, 0]] 
    w[0, 2, :, :] = [[0, 0, 0], [0, 0.1, 0], [0, 0, 0]]
    
    # 第二, 三，四个卷积核分别检测红绿蓝通道中的水平边缘。
    w[1, 0, :, :] = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    w[2, 1, :, :] = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    w[3, 2, :, :] = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    
    # Vector of biases. We don't need any bias for the grayscale 
    # filter, but for the edge detection filter we want to add 128 
    # to each output so that nothing is negative.
    b = np.array([0, 128, 128, 128])
    conv_param = {'stride': 1,
                  'pad': 1,
                  }
    pool_param = {'pool_height': 2,
                  'pool_width': 2,
                  'stride': 2,
                  }
    conv_out, _ = conv_forward_naive(x, w, b, conv_param)
    relu_out = np.maximum(conv_out, 0)
    pool_out, _ = max_pool_forward_naive(relu_out, pool_param)
    return conv_out, relu_out, pool_out


def imshow_single_image(img, normalize=True): 
    """ Tiny helper to show images as uint8 and remove axis labels """ 
    if normalize: 
        img_max, img_min = np.max(img), np.min(img) 
        img = 255.0 * (img - img_min) / (img_max - img_min)
    plt.imshow(img.astype('uint8'), cmap='gray')  # matplotlib.pyplot显示灰度图时需要加上 cmap='gray'
    plt.gca().axis('off')


def imshow_outs():
    """
    """
    conv_out, relu_out, pool_out = image_via_conv()
    images = [kitten_resize, puppy_resize, lena_resize]
    for i in range(3):
        plt.figure(i)       
        plt.subplot(4, 4, 1)
        imshow_single_image(images[i], normalize=False)
        plt.title('Original image')
        
        plt.subplot(4, 4, 2)
        imshow_single_image(conv_out[i, 0])
        plt.title('Conv out')
        
        plt.subplot(4, 4, 6)
        imshow_single_image(conv_out[i, 1])
        
        plt.subplot(4, 4, 10)
        imshow_single_image(conv_out[i, 2])
        
        plt.subplot(4, 4, 14)
        imshow_single_image(conv_out[i, 3])
        
        plt.subplot(4, 4, 3)
        imshow_single_image(relu_out[i, 0])
        plt.title('Relu out')
        
        plt.subplot(4, 4, 7)
        imshow_single_image(relu_out[i, 1])
        
        plt.subplot(4, 4, 11)
        imshow_single_image(relu_out[i, 2])
        
        plt.subplot(4, 4, 15)
        imshow_single_image(relu_out[i, 3])
        
        plt.subplot(4, 4, 4)
        imshow_single_image(pool_out[i, 0])
        plt.title('Pool out')
        
        plt.subplot(4, 4, 8)
        imshow_single_image(pool_out[i, 1])
        
        plt.subplot(4, 4, 12)
        imshow_single_image(pool_out[i, 2])
        
        plt.subplot(4, 4, 16)
        imshow_single_image(pool_out[i, 3])
    plt.show()


if __name__ == '__main__':
    imshow_outs()