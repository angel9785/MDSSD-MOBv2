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
from mobilenet_ssd_priors import priors

from deconv_fusion_v3 import SSD_FUSION
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib.pyplot as plt
# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()
priors_cxcy = priors
priors_cxcy = priors_cxcy.to(device)
# Parameters
data_folder = "/home/fereshteh/code_fusion"
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 1
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
checkpoint ="d3_mob2_NEW_fine_checkpoint_ssd300_fusion5.pth.tar"
#
checkpoint = torch.load(checkpoint)

model = SSD_FUSION(num_classes=2, backbone_network="MobileNetV2")
# model = SSD(num_classes=2, backbone_network="MobileNetV1")
	# global model
model.load_state_dict(checkpoint['model'])#model parameter
#

model = model.to(device)

kaist_path='/home/fereshteh/kaist'
# create_data_lists(kaist_path, output_folder=data_folder)

# Load test data
test_dataset = KAISTdataset(data_folder,
                                split='sanitest1',
                                keep_difficult=keep_difficult)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)
# t
def evaluate(test_loader, model):
    """
    Evaluate.
    :param test_loader: DataLoader for test data
    :param model: model
    """
    a=0
    t=0
    # Make sure it's in eval mode
    model.eval()



    with torch.no_grad():
        # Batches
        for i, (im,images_rgb,images_lw, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images_rgb = images_rgb.to(device)  # (N, 3, 300, 300)
            images_lw = images_lw.to(device)
            #print(model(images))
            # Forward prop.
            start_time = time.time()
            predicted_locs, predicted_scores = model(images_rgb,images_lw)
            stop_time = time.time()
            a = a + (stop_time - start_time)
            t = t + 1
        print(t, a)
        print("[INFO] approx. FPS: {:.2f}".format(t / a))


if __name__ == '__main__':
    evaluate(test_loader, model)


