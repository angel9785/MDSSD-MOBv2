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
# from mobilev2ssd import SSD
# from mob2_lw import SSD_LW
from lw_model513_v2 import SSD_LW
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib.pyplot as plt
# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()
priors_cxcy = priors
priors_cxcy = priors_cxcy.to(device)
# Parameters
data_folder = "/home/fereshteh/code_513/lw"
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size =1
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = './BEST_513mob2_fine_checkpoint_ssd_lw_bn_new7.pth.tar'
# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint)
print(checkpoint["epoch"])
model = SSD_LW(num_classes=2, backbone_network="MobileNetV2")
# model = SSD(num_classes=2, backbone_network="MobileNetV1")
	# global model
model.load_state_dict(checkpoint['model'])#model parameter
# model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)

model = model.to(device)
biases = list()
not_biases = list()
param_names_biases = list()
param_names_not_biases = list()
lr = 1e-3  # learning rate
momentum = 0.9  # momentum
weight_decay = 1e-4  # weight decay
grad_clip = None  # clip if g
for param_name, param in model.named_parameters():
    if param.requires_grad:
        if param_name.endswith('.bias'):
            biases.append(param)
            param_names_biases.append(param_name)
        else:
            not_biases.append(param)
            param_names_not_biases.append(param_name)
optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                            lr=lr, momentum=momentum, weight_decay=weight_decay)
optimizer.load_state_dict(checkpoint['optimizer'])  # Optimizing parameters
adjust_learning_rate(optimizer, 1)
# Switch to eval mode
model.eval()
kaist_path='/home/fereshteh/kaist'
# create_data_lists(kaist_path, output_folder=data_folder)

# Load test data
test_dataset = KAISTdataset(data_folder,
			                           split='sanitest1',
			                             keep_difficult=keep_difficult)
# test_dataset = KAISTdataset(data_folder,
# 			                           split='allsanitest',
# 			                             keep_difficult=keep_difficult)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)
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
    true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)
            #print(model(images))
            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = detect_objects(model=model, priors_cxcy=priors_cxcy, predicted_locs=predicted_locs, predicted_scores=predicted_scores, n_classes=2,
                                                                                       min_score=0.15, max_overlap=0.5,
                                                                                       top_k=200)
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

        # Calculate mAP
        APs, mAP, precision,recall,f ,n_easy_class_objects,true_positives,false_positives,lamr= calculate_mAP_kaist(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)

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
    print("lamr", lamr)
    print('\nMean Average Precision (mAP): %.3f' % mAP)


if __name__ == '__main__':
    evaluate(test_loader, model)


