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
from mobilenet_ssd_priors2 import priors
# from mobilev2ssd import SSD
# from first_model2_concat import SSD_FUSION
# from first_model2_concat import SSD_FUSION
# from second_model_cocat import SSD_FUSION
# from first_model2_sum import SSD_FUSION
from mob2 import SSD_FUSION
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import time

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
# checkpoint = './BEST_fine_checkpoint_ssd300_fusion2.pth.tar'
checkpoint ="now_mob2_NEW_fine_checkpoint_ssd300_fusion.pth.tar"
# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint)

model = SSD_FUSION(num_classes=2, backbone_network="MobileNetV2")
# model = SSD(num_classes=2, backbone_network="MobileNetV1")
	# global model
model.load_state_dict(checkpoint['model'])#model parameter
# model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)

# Switch to eval mode
model.eval()
kaist_path='/home/fereshteh/kaist'
# create_data_lists(kaist_path, output_folder=data_folder)

# Load test data
test_dataset = KAISTdataset(data_folder,
                                split='sanitest1',
                                keep_difficult=keep_difficult)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)

def evaluate(test_loader, model):
    a=0
    t=0
    model.eval()

    with torch.no_grad():
        # Batches
        for i, (im,images_rgb,images_lw, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images_rgb = images_rgb.to(device)  # (N, 3, 300, 300)
            images_lw = images_lw.to(device)
            start_time = time.time()
            predicted_locs, predicted_scores = model(images_rgb, images_lw)
            stop_time = time.time()
            a = a + (stop_time - start_time)
            t = t + 1
        print(t, a)
        print("[INFO] approx. FPS: {:.2f}".format(t / a))

            # Detect objects in SSD output


    #         # Store this batch's results for mAP calculation
    #         boxes = [b.to(device) for b in boxes]
    #         labels = [l.to(device) for l in labels]
    #         difficulties = [d.to(device) for d in difficulties]
    #
    #         det_boxes.extend(det_boxes_batch)
    #         det_labels.extend(det_labels_batch)
    #         det_scores.extend(det_scores_batch)
    #         true_boxes.extend(boxes)
    #         true_labels.extend(labels)
    #         true_difficulties.extend(difficulties)
    #
    #     # Calculate mAP
    #     APs, mAP, precision,recall,f ,n_easy_class_objects,true_positives,false_positives= calculate_mAP1(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)
    #
    # # Print AP for each class
    # # pp.pprint('AP% .3f' % APs)
    # print('AP', APs)
    # # pp.pprint("precision% .3f" % precision)
    # print("precision", precision)
    # # pp.pprint("recall% .3f" % recall)
    # print("recall", recall)
    # print("n_easy_class_objects", n_easy_class_objects)
    # print("true_positives", true_positives)
    # print("false_positives", false_positives)
    # f
    #
    # print('\nMean Average Precision (mAP): %.3f' % mAP)


if __name__ == '__main__':
    evaluate(test_loader, model)


