#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 14:58:39 2019

@author: viswanatha
"""

import numpy as np

from box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


image_size = 300
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

# specs = [
#
#     SSDSpec(25, 16, SSDBoxSizes(30, 95), [2]),
#     SSDSpec(13, 30, SSDBoxSizes(95, 160), [2, 3]),
#     SSDSpec(7, 55, SSDBoxSizes(160, 225), [2, 3]),
#     SSDSpec(4, 97, SSDBoxSizes(225, 290), [2, 3]),
#     SSDSpec(2, 193, SSDBoxSizes(290, 355), [2]),
#     SSDSpec(1, 385, SSDBoxSizes(355, 420), [2])
#
# ]
# '''
specs = [
    SSDSpec(19, 16, SSDBoxSizes(60, 105), [2]),
    SSDSpec(10, 32, SSDBoxSizes(105, 150), [2, 3]),
    SSDSpec(5, 64, SSDBoxSizes(150, 195), [2, 3]),
    SSDSpec(3, 100, SSDBoxSizes(195, 240), [2, 3]),
    SSDSpec(2, 150, SSDBoxSizes(240, 285), [2]),
    SSDSpec(1, 300, SSDBoxSizes(285, 330), [2])
]
# '''
priors = generate_ssd_priors(specs, image_size)

# print (priors.shape)
