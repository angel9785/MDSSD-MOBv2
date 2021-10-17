#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 19:48:54 2019

@author: viswanatha
"""

import torch.nn as nn
import math
import torch.nn.functional as F
import torch
from utils3 import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiBoxLoss(nn.Module):
    """
    The MultiBox loss, a loss function for object detection.
    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        # self.lastLocLoss=10

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def forward(self, predicted_locs1, predicted_scores1, predicted_locs2, predicted_scores2, boxes, labels):
        """
        Forward propagation.
        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param boxes: true  object bounding boxes in boundary coordinates, a list of N tensors
        :param labels: true object labels, a list of N tensors
        :return: multibox loss, a scalar
        """
        batch_size = predicted_locs1.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores1.size(2)

        assert n_priors == predicted_locs1.size(1) == predicted_scores1.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (N, 8732, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # (N, 8732)
        # sum=0
        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap,overlap1 = find_jaccard_overlap(boxes[i],
                                           self.priors_xy)  # (n_objects, 8732)

            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (8732)

            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # To remedy this -
            # First, find the prior that has the maximum overlap for each object.
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]  # (8732)
            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (8732)

            # Store
            true_classes[i] = label_for_each_prior
            # sum=true_classes[i].sum()+sum
            # if (sum==0 & i==batch_size-1):
            #     continue



            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy((boxes[i])[object_for_each_prior]), self.priors_cxcy)  # (8732, 4)
            #
            # for m1, box in enumerate(true_locs[i]):
            #     for f1, boxi in enumerate(box):
            #         # for h1, boxih in enumerate(boxi):
            #
            #         if(math.isinf(boxi)):
            #             print("blyat")
        # Identify priors that are positive (object/non-background)
        positive_priors = (true_classes != 0) # (N, 8732)
        # if  (positive_priors.sum()==0):
        #     return
        # LOCALIZATION LOSS
        n_positives = positive_priors.sum(dim=1)
        # Localization loss is computed only over positive (non-background) priors
        loc_loss1 = self.smooth_l1(predicted_locs1[positive_priors], true_locs[positive_priors])  # (), scalar
        loc_loss2 = self.smooth_l1(predicted_locs2[positive_priors], true_locs[positive_priors])  # (), scalar

        # if ((math.isnan(loc_loss))):
        #     if (n_positives.sum().float()!=0):
        #         print("blyat")

        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 8732)
        # So, if predicted_locs has the shape (N, 8732, 4), predicted_locs[positive_priors] will have (total positives, 4)

        # CONFIDENCE LOSS

        # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
        # That is, FOR EACH IMAGE,
        # we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
        # This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance

        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # First, find the loss for all priors
        conf_loss_all1 = self.cross_entropy(predicted_scores1.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
        conf_loss_all1 = conf_loss_all1.view(batch_size, n_priors)  # (N, 8732)

        # We already know which priors are positive
        conf_loss_pos1 = conf_loss_all1[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg1 = conf_loss_all1.clone()  # (N, 8732)
        conf_loss_neg1[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg1, _ = conf_loss_neg1.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
        hardness_ranks1 = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg1).to(device)  # (N, 8732)
        hard_negatives1 = hardness_ranks1 < n_hard_negatives.unsqueeze(1)  # (N, 8732)






        conf_loss_hard_neg1 = conf_loss_neg1[hard_negatives1]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors


        conf_loss1= (conf_loss_hard_neg1.sum() + conf_loss_pos1.sum()) / n_positives.sum().float()  # (), scalar

        conf_loss_all2 = self.cross_entropy(predicted_scores2.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
        conf_loss_all2 = conf_loss_all2.view(batch_size, n_priors)  # (N, 8732)

        # We already know which priors are positive
        conf_loss_pos2 = conf_loss_all2[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg2 = conf_loss_all2.clone()  # (N, 8732)
        conf_loss_neg2[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg2, _ = conf_loss_neg2.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
        hardness_ranks2 = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg2).to(device)  # (N, 8732)
        hard_negatives2 = hardness_ranks2 < n_hard_negatives.unsqueeze(1)  # (N, 8732)

        conf_loss_hard_neg2 = conf_loss_neg2[hard_negatives2]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors

        conf_loss2 = (conf_loss_hard_neg2.sum() + conf_loss_pos2.sum()) / n_positives.sum().float()
        # if ((math.isnan(conf_loss)==1)):
        #     if (n_positives.sum().float() != 0):
        #         print("blyat")
        # # TOTAL LOSS
        if (n_positives.sum().float()==0):
            return torch.Tensor([0]),torch.Tensor([0])

        return conf_loss1 + self.alpha * loc_loss1, conf_loss2 + self.alpha * loc_loss2



