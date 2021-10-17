#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 15:45:16 2019

@author: viswanatha
"""
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Sequential, ModuleList, ReLU
import torch
import math
from mobilenet_ssd_priors import *
from utils import *
from torchvision import transforms
import torch.nn.functional as nnf
from MobileNetV2 import MobileNetV2, MobileNetV2_pretrained
import torchvision.transforms.functional as F1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# resize = transforms.Resize((19, 19))


class RGB_MobileNetV1(nn.Module):
    def __init__(self, num_classes=2):
        super(RGB_MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),  # 0.0.weight
                nn.BatchNorm2d(oup),  # 0.1.weight 0.1.bias
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),  # 1.0.weight
                nn.BatchNorm2d(inp),  # 1.1.weight 1.1.bias
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),  # 1.3.weight
                nn.BatchNorm2d(oup),  # 1.4.weight 1.4.bias
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(  # 14 layers
            conv_bn(3, 32, 2),  # layer 0
            conv_dw(32, 64, 1),  # layer 1
            conv_dw(64, 128, 2),  # layer 2
            conv_dw(128, 128, 1),  # layer 3
            conv_dw(128, 256, 2),  # layer 4
            conv_dw(256, 256, 1),  # layer 5
            conv_dw(256, 512, 2),  # layer 6
            conv_dw(512, 512, 1),  # layer 7
            conv_dw(512, 512, 1),  # layer 8
            conv_dw(512, 512, 1),  # layer 9
            conv_dw(512, 512, 1),  # layer 10
            conv_dw(512, 512, 1),  # layer 11
            conv_dw(512, 1024, 2),  # layer 12
            conv_dw(1024, 1024, 1),  # layer 13
        )
        self.fc = nn.Linear(1024, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


class RGB_PredictionConvolutions1(nn.Module):
    """
    Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.
    The bounding boxes (locations) are predicted as encoded offsets w.r.t each of the 8732 prior (default) boxes.
    See 'cxcy_to_gcxgcy' in utils.py for the encoding definition.
    The class scores represent the scores of each object class in each of the 8732 bounding boxes located.
    A high score for 'background' = no object.
    """

    def __init__(self, n_classes):
        """
        :param n_classes: number of different types of objects
        """
        super(RGB_PredictionConvolutions1, self).__init__()

        self.n_classes = n_classes

        # Number of prior-boxes we are considering per position in each feature map
        n_boxes = {'conv5': 4,
                   'conv11': 4,
                   'conv13': 6,
                   'conv14_2': 6,
                   'conv15_2': 6
            # ,
            #        'conv16_2': 4,
            #        'conv17_2': 4
                   }
        # 4 prior-boxes implies we use 4 different aspect ratios, etc.

        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)

        self.loc_conv5 = nn.Conv2d(256, n_boxes['conv5'] * 4, kernel_size=3, padding=1)
        self.loc_conv11 = nn.Conv2d(512, n_boxes['conv11'] * 4, kernel_size=3, padding=1)
        self.loc_conv13 = nn.Conv2d(1024, n_boxes['conv13'] * 4, kernel_size=3, padding=1)
        self.loc_conv14_2 = nn.Conv2d(512, n_boxes['conv14_2'] * 4, kernel_size=3, padding=1)
        self.loc_conv15_2 = nn.Conv2d(256, n_boxes['conv15_2'] * 4, kernel_size=3, padding=1)
        # self.loc_conv16_2 = nn.Conv2d(256, n_boxes['conv16_2'] * 4, kernel_size=3, padding=1)
        # self.loc_conv17_2 = nn.Conv2d(256, n_boxes['conv17_2'] * 4, kernel_size=3, padding=1)

        # Class prediction convolutions (predict classes in localization boxes)
        self.cl_conv5 = nn.Conv2d(256, n_boxes['conv5'] * n_classes, kernel_size=3, padding=1)
        # self.cl_conv11 = nn.Conv2d(512, n_boxes['conv11'] * 4, kernel_size=3, padding=1)
        self.cl_conv11 = nn.Conv2d(512, n_boxes['conv11'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv13 = nn.Conv2d(1024, n_boxes['conv13'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv14_2 = nn.Conv2d(512, n_boxes['conv14_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_conv15_2 = nn.Conv2d(256, n_boxes['conv15_2'] * n_classes, kernel_size=3, padding=1)
        # self.cl_conv16_2 = nn.Conv2d(256, n_boxes['conv16_2'] * n_classes, kernel_size=3, padding=1)
        # self.cl_conv17_2 = nn.Conv2d(256, n_boxes['conv17_2'] * n_classes, kernel_size=3, padding=1)

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv5_feats, conv11_feats, conv13_feats, conv14_2_feats, conv15_2_feats):  # , conv16_2_feats, conv17_2_feats
        batch_size = conv11_feats.size(0)

        l_conv5 = self.loc_conv5(conv5_feats)  # (N, 16, 19, 19)
        l_conv5 = l_conv5.permute(0, 2, 3,
                                    1).contiguous()  # (N, 19, 19, 16), to match prior-box order (after .view())
        l_conv5 = l_conv5.view(batch_size, -1, 4)  # (N, 1444, 4), there are a total 1444 boxes on this feature map

        l_conv11 = self.loc_conv11(conv11_feats)  # (N, 16, 19, 19)
        l_conv11 = l_conv11.permute(0, 2, 3,
                                    1).contiguous()  # (N, 19, 19, 16), to match prior-box order (after .view())
        l_conv11 = l_conv11.view(batch_size, -1, 4)  # (N, 1444, 4), there are a total 1444 boxes on this feature map

        l_conv13 = self.loc_conv13(conv13_feats)  # (N, 24, 10, 10)
        l_conv13 = l_conv13.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 24)
        l_conv13 = l_conv13.view(batch_size, -1, 4)  # (N, 600, 4), there are a total 600 boxes on this feature map

        l_conv14_2 = self.loc_conv14_2(conv14_2_feats)  # (N, 24, 5, 5)
        l_conv14_2 = l_conv14_2.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 24)
        l_conv14_2 = l_conv14_2.view(batch_size, -1, 4)  # (N, 150, 4)

        l_conv15_2 = self.loc_conv15_2(conv15_2_feats)  # (N, 24, 3, 3)
        l_conv15_2 = l_conv15_2.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 24)
        l_conv15_2 = l_conv15_2.view(batch_size, -1, 4)  # (N, 54, 4)

        # l_conv16_2 = self.loc_conv16_2(conv16_2_feats)  # (N, 16, 2, 2)
        # l_conv16_2 = l_conv16_2.permute(0, 2, 3, 1).contiguous()  # (N, 2, 2, 16)
        # l_conv16_2 = l_conv16_2.view(batch_size, -1, 4)  # (N, 16, 4)
        #
        # l_conv17_2 = self.loc_conv17_2(conv17_2_feats)  # (N, 16, 1, 1)
        # l_conv17_2 = l_conv17_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 16)
        # l_conv17_2 = l_conv17_2.view(batch_size, -1, 4)  # (N, 4, 4)

        # Predict classes in localization boxes
        c_conv5 = self.cl_conv5(conv5_feats)  # (N, 4 * n_classes, 19, 19)
        c_conv5 = c_conv5.permute(0, 2, 3,
                                    1).contiguous()  # (N, 19, 19, 4 * n_classes), to match prior-box order (after .view())
        c_conv5 = c_conv5.view(batch_size, -1,
                                 self.n_classes)  # (N, 1444, n_classes), there are a total 1444 boxes on this feature map

        c_conv11 = self.cl_conv11(conv11_feats)  # (N, 4 * n_classes, 19, 19)
        c_conv11 = c_conv11.permute(0, 2, 3,
                                    1).contiguous()  # (N, 19, 19, 4 * n_classes), to match prior-box order (after .view())
        c_conv11 = c_conv11.view(batch_size, -1,
                                 self.n_classes)  # (N, 1444, n_classes), there are a total 1444 boxes on this feature map

        c_conv13 = self.cl_conv13(conv13_feats)  # (N, 6 * n_classes, 10, 10)
        c_conv13 = c_conv13.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 6 * n_classes)
        c_conv13 = c_conv13.view(batch_size, -1,
                                 self.n_classes)  # (N, 600, n_classes), there are a total 600 boxes on this feature map

        c_conv14_2 = self.cl_conv14_2(conv14_2_feats)  # (N, 6 * n_classes, 5, 5)
        c_conv14_2 = c_conv14_2.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 6 * n_classes)
        c_conv14_2 = c_conv14_2.view(batch_size, -1, self.n_classes)  # (N, 150, n_classes)

        c_conv15_2 = self.cl_conv15_2(conv15_2_feats)  # (N, 6 * n_classes, 3, 3)
        c_conv15_2 = c_conv15_2.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 6 * n_classes)
        c_conv15_2 = c_conv15_2.view(batch_size, -1, self.n_classes)  # (N, 54, n_classes)

        # c_conv16_2 = self.cl_conv16_2(conv16_2_feats)  # (N, 4 * n_classes, 2, 2)
        # c_conv16_2 = c_conv16_2.permute(0, 2, 3, 1).contiguous()  # (N, 2, 2, 4 * n_classes)
        # c_conv16_2 = c_conv16_2.view(batch_size, -1, self.n_classes)  # (N, 16, n_classes)
        #
        # c_conv17_2 = self.cl_conv17_2(conv17_2_feats)  # (N, 4 * n_classes, 1, 1)
        # c_conv17_2 = c_conv17_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 4 * n_classes)
        # c_conv17_2 = c_conv17_2.view(batch_size, -1, self.n_classes)  # (N, 4, n_classes)

        # A total of 2268 boxes
        # Concatenate in this specific order (i.e. must match the order of the prior-boxes)
        locs = torch.cat([l_conv5, l_conv11, l_conv13, l_conv14_2, l_conv15_2], dim=1)  # (N, 2268, 4)  , l_conv16_2, l_conv17_2
        classes_scores = torch.cat([c_conv5, c_conv11, c_conv13, c_conv14_2, c_conv15_2],dim=1)  # (N, 2268, n_classes) , c_conv16_2, c_conv17_2

        return locs, classes_scores


# class RGB_PredictionConvolutions2(nn.Module):
#     """
#     Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.
#     The bounding boxes (locations) are predicted as encoded offsets w.r.t each of the 8732 prior (default) boxes.
#     See 'cxcy_to_gcxgcy' in utils.py for the encoding definition.
#     The class scores represent the scores of each object class in each of the 8732 bounding boxes located.
#     A high score for 'background' = no object.
#     """
#
#     def __init__(self, n_classes):
#
#         super(RGB_PredictionConvolutions2, self).__init__()
#
#         self.n_classes = n_classes
#         def conv_bn(inp, oup, stride):
#             return nn.Sequential(
#                 nn.Conv2d(inp, oup, 3, stride, 1),  # 0.0.weight, bias=False
#
#                 # nn.BatchNorm2d(oup),  # 0.1.weight 0.1.bias
#                 nn.ReLU(inplace=True)
#             )
#
#         def conv_dw(inp, oup, stride):
#             return nn.Sequential(
#                 nn.Conv2d(inp, inp, 3, stride, 1, groups=inp),  # 1.0.weight, bias=False
#                   # 1.1.weight 1.1.bias
#                 nn.ReLU(inplace=True),
#
#                 nn.Conv2d(inp, oup, 1, 1, 0),  # 1.3.weight, bias=False
#                  # 1.4.weight 1.4.bias
#                 nn.ReLU(inplace=True),
#             )
#
#
#         self.model = nn.Sequential(  # 14 layers
#             conv_bn(3, 32, 2),  # layer 0
#             conv_dw(32, 64, 1),  # layer 1
#             conv_dw(64, 128, 2),  # layer 2
#             conv_dw(128, 128, 1),  # layer 3
#             conv_dw(128, 256, 2),  # layer 4
#             conv_dw(256, 256, 1),  # layer 5
#             conv_dw(256, 512, 2),  # layer 6
#             conv_dw(512, 512, 1),  # layer 7
#             conv_dw(512, 512, 1),  # layer 8
#             conv_dw(512, 512, 1),  # layer 9
#             conv_dw(512, 512, 1),  # layer 10
#             conv_dw(512, 512, 1),  # layer 11
#             conv_dw(512, 1024, 2),
#             conv_dw(1024, 1024, 1),# layer 12
#               # layer 13
#         )
#         n_boxes = {'conv5': 1}
#
#         self.cl_conv1 = nn.Conv2d(1024, n_boxes['conv5'] * n_classes, kernel_size=3, padding=1)
#
#         self._initialize_weights()
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.xavier_uniform_(m.weight)
#                 # if (m.bias != False):
#                 nn.init.constant_(m.bias, 0.)
#
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def forward(self, top_scores_loc):  # , conv16_2_feats, conv17_2_feats
#         batch_size = top_scores_loc.size(0)
#         classes_scores=[]
#
#
#
#         # sample_feat = self.model(top_scores_loc)
#         #self.model.to(device)
#
#         sample_feat = self.model(top_scores_loc)
#         c_conv1 = self.cl_conv1(sample_feat)
#         c_conv1 = c_conv1.permute(0, 2, 3,1).contiguous()
#         c_conv1 = c_conv1.view(-1, batch_size,self.n_classes)
#         classes_scores.append(c_conv1)
#         classes_scores = torch.stack(classes_scores, dim=0)
#         classes_scores = classes_scores.view(-1,self.n_classes)
#
#
#         # classes_scores = classes_scores.permute(1, 0, 2).contiguous()
#         return classes_scores
# auxiliary_conv = [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]

class RGB_AuxillaryConvolutions(nn.Module):

    def __init__(self):
        super(RGB_AuxillaryConvolutions, self).__init__()


        self.extras = ModuleList([
            Sequential(  # 5*5
                Conv2d(in_channels=1024, out_channels=256, kernel_size=1),  # 10*10*256
                ReLU(),
                Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),  # 5*5*512
                ReLU()
            ),
            Sequential(  # 3*3
                Conv2d(in_channels=512, out_channels=128, kernel_size=1),  # 5*5*128
                ReLU(),
                Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),  # 3*3*256
                ReLU()
            )
            # ,
            # Sequential( # 2*2
            #     Conv2d(in_channels=256, out_channels=128, kernel_size=1), #3*3*128
            #     ReLU(),
            #     Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1), #2*2*256
            #     ReLU()
            # ),
            # Sequential( # 1*1
            #     Conv2d(in_channels=256, out_channels=128, kernel_size=1), #2*2*128
            #     ReLU(),
            #     Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1), #1*1*256
            #     ReLU()
            # )
        ])

        self.init_conv2d()

    def init_conv2d(self):
        for c in self.children():
            for layer in c:
                if isinstance(layer, nn.Conv2d):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(c.bias, 0.)

    def forward(self, RGB_inp_features_10x10):
        RGB_features = []
        x = RGB_inp_features_10x10
        for layer in self.extras:
            x = layer(x)
            RGB_features.append(x)

        RGB_features_5x5 = RGB_features[0]
        RGB_features_3x3 = RGB_features[1]
        # RGB_features_2x2 = RGB_features[2]
        # RGB_features_1x1 = RGB_features[3]
        return RGB_features_5x5, RGB_features_3x3  # , RGB_features_2x2, RGB_features_1x1



class SSD_RGB(nn.Module):
    def __init__(self, num_classes):
        super(SSD_RGB, self).__init__()

        self.num_classes = num_classes
        self.priors = torch.FloatTensor(priors).to(device)



        self.RGB_base_net = RGB_MobileNetV1().model

        self.RGB_aux_network = RGB_AuxillaryConvolutions()

        self.RGB_prediction_network1 = RGB_PredictionConvolutions1(num_classes)

        # self.RGB_prediction_network2 = RGB_PredictionConvolutions2(num_classes)

    def forward(self, RGB_image):

        x = RGB_image

        source_layer_indexes = [
            6,
            12,
            14, ]
        start_layer_index = 0
        flag = 0
        # x = x.to('cuda')
        # self.RGB_base_net=self.RGB_base_net.to(device)
        for end_layer_index in source_layer_indexes:
            for layer in self.RGB_base_net[start_layer_index: end_layer_index]:
                x = layer(x)
            layer_output = x
            start_layer_index = end_layer_index
            if flag == 0:
                RGB_features_38x38 = layer_output
            elif flag == 1:
                RGB_features_19x19 = layer_output
            elif flag == 2:
                RGB_features_10x10 = layer_output
            flag += 1
        # self.RGB_aux_network=self.RGB_aux_network.to(device)
        RGB_features_5x5, RGB_features_3x3 = self.RGB_aux_network(RGB_features_10x10)  # , RGB_features_2x2, RGB_features_1x1

        # self.RGB_prediction_network1=self.RGB_prediction_network1.to(device)
        RGB_locs, RGB_class_scores = self.RGB_prediction_network1.forward(RGB_features_38x38, RGB_features_19x19, RGB_features_10x10, RGB_features_5x5, RGB_features_3x3)

        # batch_size = RGB_locs.size(0)
        # predicted_scores = F.softmax(RGB_class_scores, dim=2)
        # priors_cxcy = priors
        # priors_cxcy = priors_cxcy.to(device)
        # predicted_scores1=predicted_scores.clone()
        # scores_refine=[]
        # decoded_locs_refine=[]
        # top_predicted_locs=[]
        # sample=[]
        # # h=(predicted_scores>0.8).sum(dim=1)
        # # print(h)
        # for j in range(batch_size):
        #     # crop1 = []
        #     decoded_locs = cxcy_to_xy(gcxgcy_to_cxcy(RGB_locs[j], priors_cxcy))
        #     val,ind=torch.topk(predicted_scores[j,:,1], 5)
        #     for i in sorted(ind):
        #         # if(predicted_scores[j][i][1]>0.8):
        #
        #         box_location =decoded_locs[i].tolist()
        #         to_pil = transforms.ToPILImage()
        #         RGB = to_pil(RGB_image[j].cpu())
        #         im = RGB.crop((box_location[0] * 300, box_location[1] * 300, box_location[2] * 300, box_location[3] * 300))
        #         resize2 = transforms.Resize((19, 19))
        #         to_tensor2 = transforms.ToTensor()
        #         sample = (to_tensor2(resize2(im)))
        #         sample=sample.to(device)
        #         self.RGB_prediction_network2=self.RGB_prediction_network2.to(device)
        #         predicted_scores1[j][i] = self.RGB_prediction_network2.forward(sample.unsqueeze(0))
                    # crop1.append(sample)
        #     crop2 = torch.stack(crop1, dim=0)
        #     crop.append(crop2)
        # crop=[]
        # if (predicted_scores
        # for i in range(batch_size):
        #     # scores, sort_ind = predicted_scores[i][:, 1].sort(dim=0, descending=True)
        #     RGB_locs1 = RGB_locs[i][sort_ind]
        #     locs_refine = RGB_locs1[0:50]
        #     top_predicted_locs.append(locs_refine)
        #     decoded_locs = cxcy_to_xy(gcxgcy_to_cxcy(RGB_locs[i][:], priors_cxcy))
        #
        #     # RGB_locs = RGB_locs[sort_ind]
        #     # locs_refine = RGB_locs[0:50]
        #     # top_predicted_locs.append(locs_refine)
        #     decoded_locs = decoded_locs[sort_ind]
        #     # scores_refine.append(scores[0:50])
        #     decoded_locs_refine=decoded_locs[0:50]
        #     crop1 = []
        #     for j in range (0,50):
        #         box_location = decoded_locs_refine[j].tolist()
        #         box_location=box_location
        #         to_pil = transforms.ToPILImage()
        #         RGB=to_pil(RGB_image[i].cpu())
        #
        #         im=RGB.crop((box_location[0]*300, box_location[1]*300,box_location[2]*300, box_location[3]*300))
        #         resize2 = transforms.Resize((19, 19))
        #
        #         to_tensor2 = transforms.ToTensor()
        #         sample = to_tensor2(resize2(im))
        #
        #         crop1.append(sample)
        #         # crop.append(F1.resize(im, 19))
        #     crop2 = torch.stack(crop1, dim=0)
        #     crop.append(crop2)
        # predicted_scores = torch.cat(predicted_scores,dim=1)
        # decoded_locs_refine = torch.cat(decoded_locs_refine, dim=1)
        # # crop = crop.unqueeze(0)
        # crop = torch.stack(crop, dim=0)
        # crop = crop.to(device)
        # # true_images = torch.LongTensor(crop).to(device)
        # top_predicted_locs= torch.stack(top_predicted_locs, dim=0)
        # top_predicted_locs = top_predicted_locs.view(-1,4)
        #
        # top_RGB_scores = self.RGB_prediction_network2.forward(crop)
        # top_RGB_scores=torch.cat(top_RGB_scores, dim=0)
        return RGB_locs, RGB_class_scores

#
# # '''
# import numpy as np
# import torch
# import time
# from PIL import Image, ImageDraw, ImageFont
# from torchvision import transforms
#
# # img = np.random.rand(8, 3, 300, 300)
# # img = torch.Tensor(img)
# img_path = '/home/fereshteh/codergb/I02658.png'
# original_image = Image.open(img_path, mode='r')
# original_image = original_image.convert('RGB')
# resize = transforms.Resize((300, 300))
# to_tensor = transforms.ToTensor()
# # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
# #                                    std=[0.229, 0.224, 0.225])
#
# original_image =to_tensor(resize(original_image))
# image=[original_image,original_image,original_image,original_image,original_image,original_image,original_image,original_image]
# # image = (to_tensor(resize(original_image)))
# images = torch.stack(image, dim=0)
#     # Move to default device
# img = images.to(device)
#
# model = SSD_RGB(2)
# model = model.to(device)
# s1=time.time()
# loc, classes = model.forward(img)
# print(loc, classes)
# s2=time.time()
# s3=s2-s1
# print(s3)
#
# # print (loc.shape, classes.shape)
# # for param_name, param in model.RGB_DeConvolutions.named_parameters():
# #     print(param_name)
#     # if (param_name=='RGB_prediction_network.cl_conv17_2.bias'):
# #     param.copy_(state_dict['RGB_aux_network.'+param_name])
# # '''
