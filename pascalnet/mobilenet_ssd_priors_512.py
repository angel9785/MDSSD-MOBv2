#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 14:58:39 2019

@author: viswanatha
"""

import numpy as np

from box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


image_size = 513
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2
#
# specs = [
#     SSDSpec(32, 16, SSDBoxSizes(40, 125), [2]),
#     SSDSpec(16, 32, SSDBoxSizes(125, 210), [2, 3]),
#     SSDSpec(8, 64, SSDBoxSizes(210, 295), [2, 3]),
#     SSDSpec(4, 128, SSDBoxSizes(295, 380), [2, 3])
#     ,
#     SSDSpec(2, 256, SSDBoxSizes(380, 465), [2]),
#     SSDSpec(1, 512, SSDBoxSizes(465, 550), [2])
# ]
specs = [
    SSDSpec(33, 16, SSDBoxSizes(60, 130), [2]),
    SSDSpec(17, 31, SSDBoxSizes(130, 200), [2]),
    SSDSpec(9, 57, SSDBoxSizes(200, 270), [2, 3]),
    SSDSpec(5, 103, SSDBoxSizes(270, 340), [2, 3]),
    SSDSpec(3, 171, SSDBoxSizes(340, 410), [2, 3])
    ,
    SSDSpec(2, 257, SSDBoxSizes(410, 480), [2]),
    SSDSpec(1, 513, SSDBoxSizes(480, 550), [2])
]
'''
specs = [
    SSDSpec(19, 16, SSDBoxSizes(60, 105), [2, 3]),
    SSDSpec(10, 32, SSDBoxSizes(105, 150), [2, 3]),
    SSDSpec(5, 64, SSDBoxSizes(150, 195), [2, 3]),
    SSDSpec(3, 100, SSDBoxSizes(195, 240), [2, 3]),
    SSDSpec(2, 150, SSDBoxSizes(240, 285), [2, 3]),
    SSDSpec(1, 300, SSDBoxSizes(285, 330), [2, 3])
]
'''
priors = generate_ssd_priors(specs, image_size)

#print (priors.shape)
