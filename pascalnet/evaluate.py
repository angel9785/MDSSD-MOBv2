#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 15:45:16 2019

@author: viswanatha
"""

from utils import *
from datasets import PascalVOCDataset
from tqdm import tqdm
from pprint import PrettyPrinter
from mobilenet_ssd_priors_512 import priors
import warnings
# from rgb_model1 import SSD_RGB
from dec512 import SSD_RGB
warnings.filterwarnings("ignore", category=UserWarning)
# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()
priors_cxcy = priors
priors_cxcy = priors_cxcy.to(device)
# Parameters
config_file_path = "/home/fereshteh/code/SSD_MobileNet-master/config.json"

with open(config_file_path, "r") as fp:
    config = json.load(fp)
voc07_path = config['voc07_path']

    #voc12_path = 'VOCdevkit/VOC2012'
voc12_path = config['voc12_path']
# create_data_lists(voc07_path, voc12_path, output_folder=config['data_folder'])
data_folder = 'dataset'
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 5
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# checkpoint = '/home/fereshteh/code/SSD_MobileNet-master/dec_voc_checkpoint_ssd300_new2.pth.tar'
checkpoint = '/home/fereshteh/code/SSD_MobileNet-master/513pascalvoc_dec_checkpoint_ssd_new_BN2.pth.tar'
# checkpoint = '/home/fereshteh/code/SSD_MobileNet-master/513pascalvoc_dec_checkpoint_ssd_new10.pth.tar'
voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()} # Inverse mapping
n_classes = len(label_map)  # number of different types of objects
backbone_network='MobileNetV2'
# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint)
# model = model.to(device)
model = SSD_RGB(num_classes=n_classes,backbone_network=backbone_network)
# model = checkpoint['model']
model.load_state_dict(checkpoint['model'])
print(checkpoint['epoch'])

# model.load_state_dict(checkpoint['model'])#model parameter
model = model.to(device)
model.eval()

# Switch to eval mode
# model.eval()

# Load test data
test_dataset = PascalVOCDataset(data_folder,
                                split='test',
                                keep_difficult=keep_difficult)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)


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
            det_boxes_batch, det_labels_batch, det_scores_batch = detect_objects(model=model, priors_cxcy=priors_cxcy, predicted_locs=predicted_locs, predicted_scores=predicted_scores, n_classes=n_classes,
                                                                                       min_score=0.02, max_overlap=0.45,
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
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)

    # Print AP for each class
    pp.pprint(APs)

    print('\nMean Average Precision (mAP): %.3f' % mAP)


if __name__ == '__main__':
    evaluate(test_loader, model)
