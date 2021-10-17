#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 15:45:16 2019

@author: viswanatha
"""
##dlib
from utils import *
import dlib

from datasets import KAISTdataset
from tqdm import tqdm
from pprint import PrettyPrinter
from mobilenet_ssd_priors2 import priors
import argparse
# from sort import *
# from sort_multi_object_tracking import SORTMultiObjectTracking

# from mobilev2ssd import SSD
# from second_model_sum import SSD_FUSION
# from second_score import SSD_FUSION
# from secondmodel2 import SSD_FUSION
from mob2 import SSD_FUSION
# from filterpy.kalman import KalmanFilter
# from first_model2_sum import SSD_FUSION
# from first_model2_concat import SSD_FUSION
# from second_model_concat2 import SSD_FUSION
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib.pyplot as plt
# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()
priors_cxcy = priors
priors_cxcy = priors_cxcy.to(device)
# Parameters
# mot_tracker = Sort()
data_folder = "/home/fereshteh/code_fusion"
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 1
workers = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--tracker", type=str, default="kcf",
	help="OpenCV object tracker type")
args = vars(ap.parse_args())

OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"boosting": cv2.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.TrackerTLD_create,
	"medianflow": cv2.TrackerMedianFlow_create,
	"mosse": cv2.TrackerMOSSE_create,
    "GOTURN":cv2.TrackerGOTURN_create
}

# checkpoint = './second_sum_fine_checkpoint_ssd300_fusion3.pth.tar'
# checkpoint = './deconv_fine_checkpoint_ssd300_fusion4.pth.tar'
# checkpoint = './NEW19_fine_checkpoint_ssd300_fusion_8.pth.tar'
# checkpoint ="first_sum_fine_checkpoint_ssd300_fusion_2.pth.tar"
# checkpoint = './score_second_fine_checkpoint_ssd300_fusion8.pth.tar'
# checkpoint = 'second_fine_checkpoint_ssd300_fusion2.pth (copy).tar'
# checkpoint = 'NEW19_fine_checkpoint_ssd300_fusion_8.pth (copy).tar'
# '''
# checkpoint = 'BEST_NEW_fine_checkpoint_ssd300_fusion.pth (copy).tar'
# '''
# checkpoint = 'NEW_fine_checkpoint_ssd300_fusion12.pth (5th copy).tar'
checkpoint ="mob2_NEW_fine_checkpoint_ssd300_fusion4.pth (another copy).tar"
# checkpoint = 'second_fine_checkpoint_ssd300_fusion16.pth.tar'
# checkpoint = './second_fine_checkpoint_ssd300_fusion11.pth.tarfine_checkpoint_ssd300_fusion11.pth.tarscore_second_fine_checkpoint_ssd300_fusion2.pth.tar'
# checkpoint = './BEST_second_sum_fine_checkpoint_ssd300_fusion2.pth.tar'
# checkpoint = 'BEST_fine_checkpoint_ssd300_fusion2.pth.tar'
# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint)

model = SSD_FUSION(num_classes=2, backbone_network="MobileNetV2")

# model = SSD(num_classes=2, backbone_network="MobileNetV1")
	# global model
model.load_state_dict(checkpoint['model'])#model parameter
# model.load_state_dict(checkpoint['state_dict'])
print(checkpoint['epoch'])
print(checkpoint['loss'])

model = model.to(device)
biases = list()
not_biases = list()
param_names_biases = list()
param_names_not_biases = list()
lr = 1e-4  # learning rate
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
                                split='SANITEST',
                                keep_difficult=keep_difficult)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)
# train_dataset = KAISTdataset(data_folder,
#                                 split='allsanitrain1',
#                                 keep_difficult=keep_difficult)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
#                                               collate_fn=train_dataset.collate_fn, num_workers=workers, pin_memory=True)

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
    track_boxes1 = list()
    track_success1 = list()
    # track_boxes = list()
    track_success = list()
    det_boxes1 = list()
    det_labels1 = list()
    det_scores1 = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils2.pil
    bir=list()
    bil=list()
    p=0
    global b
    global u
    with torch.no_grad():
        # Batches

        # trackers = cv2.MultiTracker_create()
        # # trackers = OPENCV_OBJECT_TRACKERS["mosse"]()
        k=1
        det_bbox1=[]
        det_label1=[]
        det_score1=[]
        trackers = []
        for i, (ir,images_rgb,images_lw, boxes1, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):


            images_rgb = images_rgb.to(device)  # (N, 3, 300, 300)
            images_lw = images_lw.to(device)
            track_boxes=[]

            bir1=ir[0]
            bir1 = cv2.cvtColor(bir1, cv2.COLOR_BGR2RGB)

            original_dims = torch.FloatTensor(
                [640, 512, 640, 512]).unsqueeze(0)



            predicted_locs, predicted_scores = model(images_rgb,images_lw)
            # predicted_scores=predicted_scores+ R+ L
            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = detect_objects2(model=model, priors_cxcy=priors_cxcy, predicted_locs=predicted_locs, predicted_scores=predicted_scores, n_classes=2,
                                                                                   min_score=0.1, max_overlap=0.5,
                                                                                   top_k=200)

            det_boxes = torch.cat(det_boxes_batch, dim=0)  # (n_detections, 4)
            det_labels = torch.cat(det_labels_batch, dim=0)  # (n_detections)
            det_scores = torch.cat(det_scores_batch, dim=0)  # (n_detections)

            det_class_boxes1 = det_boxes[det_labels == 1]  # (n_class_detections, 4)
            det_class_scores1 = det_scores[det_labels == 1]  # (n_class_detections)
            n_class_detections1 = det_class_boxes1.size(0)
            det_class_images = list()
            det_class_boxes = list()
            det_class_scores = list()

            # det_class_boxes1=det_class_boxes1.unsqueeze(1)
            # det_class_scores1=det_class_scores1.unsqueeze(1)
            for i in range(0, n_class_detections1):
                if ((det_class_boxes1[i][3] - det_class_boxes1[i][1] >= (100/512))):
                     # (n_class_detections)
                    det_class_boxes.append(det_class_boxes1[i].unsqueeze(0))  # (n_class_detections, 4)
                    det_class_scores.append(det_class_scores1[i].unsqueeze(0))
            det_class_images = torch.LongTensor(det_class_images).to(device)  # (n_detections)
            # print(det_class_boxes)
            # trackers = cv2.MultiTracker_create()
            if len(trackers) != 0:


                # loop over each of the trackers
                for t in trackers:
                    # update the tracker and grab the position of the tracked
                    # object
                    t.update(bir1)
                    pos = t.get_position()
                    # unpack the position object
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())
                    track_boxes.append([startX,startY,endX,endY])
                    print([startX,startY,endX,endY])
                trackers = []
            n = len(det_class_boxes)

            bb3=[]
            # if len(trackers) == 0:

            for i in range(0, n):


                det2=(det_class_boxes[i].cpu()*original_dims)
                # print(det2)
                bb = np.hstack(det2.cpu()).astype(np.int32)



            # bb[2]=bb[2] - bb[0]
            # bb[3] = bb[3] - bb[1]
                print(bb,"det")
                if bb[2]>640:
                    bb[2]=640
                if bb[3]>512:
                    bb[3]=512
                if bb[0]<0:
                    bb[0]=0
                if bb[1]<0:
                    bb[1]=0

                bb1 = [bb[0], bb[1], bb[2], bb[3]]
                print(bb1, "det to tr")
                # bb1 = np.array(bb1)
                t = dlib.correlation_tracker()
                rect = dlib.rectangle(bb[0], bb[1], bb[2], bb[3])
                t.start_track(bir1, rect)
                print(t,"tr")
                trackers.append(t)


                print(bb3)



                # print(bb1)
                # print(bir1.shape[0])
                # print(bir1.shape,bb1)
                # if (bb1[0] >= 0 & bb1[2] >= 0 & bb1[2] +bb1[0]<= bir1.shape[0] & bb1[1] >= 0 & bb1[3] >= 0 & bb1[3]+bb1[1]  <= bir1.shape[1]):
                # trackers = cv2.MultiTracker_create()

            # trackers.add(tracker, bir1, bb1)

            boxes1 = [b.to(device) for b in boxes1]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            det_boxes1.extend(det_boxes_batch)
            det_labels1.extend(det_labels_batch)
            det_scores1.extend(det_scores_batch)
            true_boxes.extend(boxes1)
            true_labels.extend(labels)
            track_boxes1.append(track_boxes)
            true_difficulties.extend(difficulties)
            bir.extend(ir)



        h = 0

        for i in range(bir.__len__()):
            # bir="/home/fereshteh/result_t/"+str(i)+".png"


            g= calculate_mAP_tr(h,bir[i],det_boxes1[i], det_labels1[i], det_scores1[i], true_boxes[i], true_labels[i], true_difficulties[i],track_boxes1[i])
            h=h+1


    # Print AP for each class
    # pp.pprint('AP% .3f' % APs)
    # print('AP', APs)
    # # pp.pprint("precision% .3f" % precision)
    # print("precision", precision)
    # # pp.pprint("recall% .3f" % recall)
    # print("recall", recall)
    # print("n_easy_class_objects", n_easy_class_objects)
    # print("true_positives", true_positives)
    # print("false_positives", false_positives)


    # print('\nMean Average Precision (mAP): %.3f' % mAP)


if __name__ == '__main__':

    evaluate(test_loader, model)


