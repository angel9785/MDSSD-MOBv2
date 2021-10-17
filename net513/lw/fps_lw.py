#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 15:45:16 2019

@author: viswanatha
"""

from utils import *
from datasets import KAISTdataset
from tqdm import tqdm
from pprint import PrettyPrinter
from mobilenet_ssd_priors513 import priors

from lw_model513_v2 import SSD_LW
import time

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

pp = PrettyPrinter()
priors_cxcy = priors
priors_cxcy = priors_cxcy.to(device)
# Parameters
data_folder = "/home/fereshteh/code_513/lw"
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 1
# batch_size = 1
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = './' \
             '513mob2_fine_checkpoint_ssd_lw_bn3.pth.tar'


checkpoint = torch.load(checkpoint)

model = SSD_LW(num_classes=2, backbone_network="MobileNetV2")

model.load_state_dict(checkpoint['model'])#model parameter
model = model.to(device)

model.eval()

test_dataset = KAISTdataset(data_folder,
                                       split='sanitest1',
                                         keep_difficult=keep_difficult)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)

def evaluate(test_loader, model):
    a = 0
    t = 0
    model.eval()


    with torch.no_grad():
        # Batches

        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)
            start_time = time.time()
            # Forward prop.
            predicted_locs, predicted_scores = model(images)
                   # Detect objects in SSD output
            stop_time = time.time()
            a = a + (stop_time - start_time)
            t = t + 1
        print(t, a)
        print("[INFO] approx. FPS: {:.2f}".format(t / a))

    # Print AP for each class
    # pp.pprint('AP% .3f' % APs)

    # print('\nMean Average Precision (mAP): %.3f' % mAP)
    # fps.stop()
    # print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

if __name__ == '__main__':
    evaluate(test_loader, model)


