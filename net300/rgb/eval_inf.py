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
from mobilev2ssd import SSD
# from rgb_model import SSD_RGB
from REFINE_MODEL import SSD_RGB_SATAGE1
from REFINE_MODEL import SSD_RGB_SATAGE2
from REFINE_MODEL import jumper as JUMPER
from loss import MultiBoxLoss
import time
# import time
# from imutils.video import FPS

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib.pyplot as plt
# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()
priors_cxcy = priors
priors_cxcy = priors_cxcy.to(device)
# Parameters
data_folder = "/home/fereshteh/codelw_new"
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 1
# batch_size = 1
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = './LW_refine_fine_checkpoint_ssd300_3.pth.tar'

# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint)

model = SSD_RGB_SATAGE1(num_classes=2, backbone_network="MobileNetV1")
    # global model
model.load_state_dict(checkpoint['model1'])  # model parameter
model = model.to(device)
model.eval()
model2 = SSD_RGB_SATAGE2(num_classes=2, backbone_network="MobileNetV1")
# global model
model2.load_state_dict(checkpoint['model2'])  # model parameter
model2 = model2.to(device)
model2.eval()
jumper = JUMPER(2, 0.2)

jumper.load_state_dict(checkpoint['jumper'])  # model parameter
jumper.to("cpu")
jumper.eval()

# biases = list()
# not_biases = list()
# param_names_biases = list()
# param_names_not_biases = list()
# lr = 1e-3  # learning rate
# momentum = 0.9  # momentum
# weight_decay = 1e-4  # weight decay
# grad_clip = None  # clip if g
# for param_name, param in model.named_parameters():
#     if param.requires_grad:
#         if param_name.endswith('.bias'):
#             biases.append(param)
#             param_names_biases.append(param_name)
#         else:
#             not_biases.append(param)
#             param_names_not_biases.append(param_name)
# optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
#                             lr=lr, momentum=momentum, weight_decay=weight_decay)
# optimizer.load_state_dict(checkpoint['optimizer'])  # Optimizing parameters
# adjust_learning_rate(optimizer, 1)
# # Switch to eval mode
# model.eval()
kaist_path='/home/fereshteh/kaist'
# create_data_lists(kaist_path, output_folder=data_folder)

# Load test data
test_dataset = KAISTdataset(data_folder,
			                           split='allsanitest',
			                             keep_difficult=keep_difficult)
# test_dataset = KAISTdataset(data_folder,
# 			                           split='allsanitest',
# 			                             keep_difficult=keep_difficult)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)
criterion1 = MultiBoxLoss(priors_cxcy=model.priors).to(device)
# with open(os.path.join(data_folder, 'DAY' + '_images.json'), 'r') as j:
#     images = json.load(j)
# with open(os.path.join(data_folder, 'DAY' + '_objects.json'), 'r') as j:
#     objects = json.load(j)
# with open(os.path.join(data_folder, 'SANITEST' + '_images.json'), 'r') as j:
#     images1 = json.load(j)
# with open(os.path.join(data_folder, 'SANITEST' + '_objects.json'), 'r') as j:
#     objects1 = json.load(j)
# l=0
# l1=0
# for i in range(len(images)):
#     l = l + len((objects[i]['labels']))
# for i in range(len(images1)):
#     l1 = l1 + len((objects[i]['labels']))
# print(l)
# print(l1)
def evaluate(test_loader, model):
    """
    Evaluate.
    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()

    refine_locs = list()
    # ID=list()# it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py
    # fps = FPS().start()
    with torch.no_grad():
        # Batches
        for i, (imagesp,images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)
            #print(model(images))
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            # difficulties = [d.to(device) for d in difficulties]
            # labels=torch.stack(labels)
            # Forward prop.
            predicted_locs, predicted_scores = model(images)
            loss1, true = criterion1(predicted_locs, predicted_scores, boxes, labels)
            roi, refine_locs, ID = jumper(imagesp, predicted_locs, predicted_scores, true)
            # roi = [roi.to(device) for l in roi]
            if (roi.__len__()!=0):
                # roi=torch.cat(roi,dim=0)


                roi = roi.to(device)
                refine_scores = model2(roi)
                # tor
                refine_scores = refine_scores.to(device)
            else:
                refine_scores=torch.empty(size=(1,0,2))
            print(refine_locs.shape)
            print(refine_scores.shape)
            print(ID.shape)
            ID = ID.view(1, 10)
            refine_locs = refine_locs.view(1, 10, 4)
            refine_scores = refine_scores.view(1, 10, 2)
            # print(refine_locs.shape)
            # print(refine_scores.shape)

            # difficulties = [d.to(device) for d in difficulties]
              # scalar
            # fps.update()
            # true = true.to(device)
            # Detect objects in SSD output
            # ID.extend(ID1)

            # if (refine_scores.nelement() == 0):
            det_boxes_batch, det_labels_batch, det_scores_batch = detect_objects(refine_locs, model, priors_cxcy, predicted_locs,
                                                           refine_scores,
                                                           min_score=0.00,
                                                           max_overlap=0.5, top_k=200,
                                                           n_classes=2)
            # else:
            #     det_boxes_batch=torch.empty(ID.__len__())
            #     det_labels_batch=torch.empty(ID.__len__())
            #     det_scores_batch=torch.empty(ID.__len__())
            # det_boxes_batch, det_labels_batch, det_scores_batch = detect_objects(model=model, priors_cxcy=priors_cxcy, predicted_locs=predicted_locs, predicted_scores=predicted_scores, n_classes=2,
            #                                                                            min_score=0.25, max_overlap=0.5,
            #                                                                            top_k=200)
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos
            # refine_locs.extend(refine_locs1)
            # Store this batch's results for mAP calculation
            # refine_locs = [b.to(device) for b in refine_locs]
            # ID = [l.to(device) for l in ID]
            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(refine_locs)
            true_labels.extend(ID)
            # true_difficulties.extend(difficulties)
        # true_difficulties=zeros(true_boxes.__len__())
        # Calculate mAP
        APs, mAP, precision,recall,f ,n_easy_class_objects,true_positives,false_positives= calculate_mAP2(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)

    # Print AP for each class
    # pp.pprint('AP% .3f' % APs)
    print('AP', APs)
    # pp.pprint("precision% .3f" % precision)
    print("precision", precision)
    # pp.pprint("recall% .3f" % recall)
    print("recall", recall)
    print("n_easy_class_objects", n_easy_class_objects)
    print("true_positives", true_positives)
    print("false_positives", false_positives)
    f

    # print('\nMean Average Precision (mAP): %.3f' % mAP)
    # fps.stop()
    # print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

if __name__ == '__main__':
    evaluate(test_loader, model)


