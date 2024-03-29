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
# from mobilev2ssd import SSD
# from second_score import SSD_FUSION
# from secondmodel2 import SSD_FUSION
from fusion_513_v2 import SSD_FUSION
# from imutils.video import FPS
import time
# from second_model_cocat import SSD_FUSION
# from first_model2_sum import SSD_FUSION
# from first_model2_concat import SSD_FUSION
# from second_model_concat2 import SSD_FUSION
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib.pyplot as plt
# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Parameters
data_folder = "/home/fereshteh/code_fusion"
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 1
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
checkpoint ="BEST_dec_mob2_NEW_fine_checkpoint_ssd300_fusion_bn8.pth.tar"
#
checkpoint = torch.load(checkpoint)

model = SSD_FUSION(num_classes=2, backbone_network="MobileNetV2")
# model = SSD(num_classes=2, backbone_network="MobileNetV1")
	# global model
model.load_state_dict(checkpoint['model'])#model parameter
#

model = model.to(device)


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

    # Make sure it's in eval mode
    model.eval()


    a=0
    t=0

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


