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
from mobilenet_ssd_priors513 import *
from MobileNetV2 import MobileNetV2, MobileNetV2_pretrained

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LW_MobileNetV1(nn.Module):
    def __init__(self, num_classes=2):
        super(LW_MobileNetV1, self).__init__()

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
        x = x.view(-1, 1280)
        x = self.fc(x)
        return x


class LW_PredictionConvolutions(nn.Module):
    """
    Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.
    The bounding boxes (locations) are predicted as encoded offsets w.r.t each of the 8732 prior (default) boxes.
    See 'cxcy_to_gcxgcy' in utils.py for the encoding definition.
    The class scores represent the scores of each object class in each of the 8732 bounding boxes located.
    A high score for 'background' = no object.
    """

    def __init__(self, n_classes, backbone_net):
        """
        :param n_classes: number of different types of objects
        """
        super(LW_PredictionConvolutions, self).__init__()

        self.n_classes = n_classes

        # Number of prior-boxes we are considering per position in each feature map
        n_boxes = {'conv11': 4,
                   'conv13': 4,
                   'conv14_2': 6,
                   'conv15_2': 6,
                   'conv16_2': 6,
                   'conv17_2': 4,
                   'conv18_2': 4}
        # 4 prior-boxes implies we use 4 different aspect ratios, etc.

        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        if backbone_net == 'MobileNetV2':
            self.loc_conv11 = nn.Conv2d(96, n_boxes['conv11'] * 4, kernel_size=3, padding=1)
            self.loc_conv13 = nn.Conv2d(1280, n_boxes['conv13'] * 4, kernel_size=3, padding=1)
            self.loc_conv14_2 = nn.Conv2d(512, n_boxes['conv14_2'] * 4, kernel_size=3, padding=1)
            self.loc_conv15_2 = nn.Conv2d(256, n_boxes['conv15_2'] * 4, kernel_size=3, padding=1)
            self.loc_conv16_2 = nn.Conv2d(256, n_boxes['conv16_2'] * 4, kernel_size=3, padding=1)
            self.loc_conv17_2 = nn.Conv2d(256, n_boxes['conv17_2'] * 4, kernel_size=3, padding=1)
            self.loc_conv18_2 = nn.Conv2d(256, n_boxes['conv18_2'] * 4, kernel_size=3, padding=1)
            # Class prediction convolutions (predict classes in localization boxes)
            self.cl_conv11 = nn.Conv2d(96, n_boxes['conv11'] * n_classes, kernel_size=3, padding=1)
            self.cl_conv13 = nn.Conv2d(1280, n_boxes['conv13'] * n_classes, kernel_size=3, padding=1)
            self.cl_conv14_2 = nn.Conv2d(512, n_boxes['conv14_2'] * n_classes, kernel_size=3, padding=1)
            self.cl_conv15_2 = nn.Conv2d(256, n_boxes['conv15_2'] * n_classes, kernel_size=3, padding=1)
            self.cl_conv16_2 = nn.Conv2d(256, n_boxes['conv16_2'] * n_classes, kernel_size=3, padding=1)
            self.cl_conv17_2 = nn.Conv2d(256, n_boxes['conv17_2'] * n_classes, kernel_size=3, padding=1)
            self.cl_conv18_2 = nn.Conv2d(256, n_boxes['conv18_2'] * n_classes, kernel_size=3, padding=1)
            # Initialize convolutions' parameters
            self.init_conv2d()
        elif backbone_net == 'MobileNetV1':
            self.loc_conv11 = nn.Conv2d(512, n_boxes['conv11'] * 4, kernel_size=3, padding=1)
            self.loc_conv13 = nn.Conv2d(1024, n_boxes['conv13'] * 4, kernel_size=3, padding=1)
            self.loc_conv14_2 = nn.Conv2d(512, n_boxes['conv14_2'] * 4, kernel_size=3, padding=1)
            self.loc_conv15_2 = nn.Conv2d(256, n_boxes['conv15_2'] * 4, kernel_size=3, padding=1)
            self.loc_conv16_2 = nn.Conv2d(256, n_boxes['conv16_2'] * 4, kernel_size=3, padding=1)
            self.loc_conv17_2 = nn.Conv2d(256, n_boxes['conv17_2'] * 4, kernel_size=3, padding=1)

            # Class prediction convolutions (predict classes in localization boxes)
            self.cl_conv11 = nn.Conv2d(512, n_boxes['conv11'] * n_classes, kernel_size=3, padding=1)
            self.cl_conv13 = nn.Conv2d(1024, n_boxes['conv13'] * n_classes, kernel_size=3, padding=1)
            self.cl_conv14_2 = nn.Conv2d(512, n_boxes['conv14_2'] * n_classes, kernel_size=3, padding=1)
            self.cl_conv15_2 = nn.Conv2d(256, n_boxes['conv15_2'] * n_classes, kernel_size=3, padding=1)
            self.cl_conv16_2 = nn.Conv2d(256, n_boxes['conv16_2'] * n_classes, kernel_size=3, padding=1)
            self.cl_conv17_2 = nn.Conv2d(256, n_boxes['conv17_2'] * n_classes, kernel_size=3, padding=1)

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

    def forward(self, conv11_feats, conv13_feats, conv14_2_feats, conv15_2_feats, conv16_2_feats, conv17_2_feats,conv18_2_feats):  # , conv16_2_feats, conv17_2_feats
        batch_size = conv11_feats.size(0)

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

        l_conv16_2 = self.loc_conv16_2(conv16_2_feats)  # (N, 16, 2, 2)
        l_conv16_2 = l_conv16_2.permute(0, 2, 3, 1).contiguous()  # (N, 2, 2, 16)
        l_conv16_2 = l_conv16_2.view(batch_size, -1, 4)  # (N, 16, 4)

        l_conv17_2 = self.loc_conv17_2(conv17_2_feats)  # (N, 16, 1, 1)
        l_conv17_2 = l_conv17_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 16)
        l_conv17_2 = l_conv17_2.view(batch_size, -1, 4)  # (N, 4, 4)

        l_conv18_2 = self.loc_conv18_2(conv18_2_feats)  # (N, 16, 1, 1)
        l_conv18_2 = l_conv18_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 16)
        l_conv18_2 = l_conv18_2.view(batch_size, -1, 4)  # (N, 4, 4)
        # Predict classes in localization boxes
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

        c_conv16_2 = self.cl_conv16_2(conv16_2_feats)  # (N, 4 * n_classes, 2, 2)
        c_conv16_2 = c_conv16_2.permute(0, 2, 3, 1).contiguous()  # (N, 2, 2, 4 * n_classes)
        c_conv16_2 = c_conv16_2.view(batch_size, -1, self.n_classes)  # (N, 16, n_classes)

        c_conv17_2 = self.cl_conv17_2(conv17_2_feats)  # (N, 4 * n_classes, 1, 1)
        c_conv17_2 = c_conv17_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 4 * n_classes)
        c_conv17_2 = c_conv17_2.view(batch_size, -1, self.n_classes)  # (N, 4, n_classes)
        c_conv18_2 = self.cl_conv18_2(conv18_2_feats)  # (N, 4 * n_classes, 1, 1)
        c_conv18_2 = c_conv18_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 4 * n_classes)
        c_conv18_2 = c_conv18_2.view(batch_size, -1, self.n_classes)  # (N, 4, n_classes)
        # A total of 2268 boxes
        # Concatenate in this specific order (i.e. must match the order of the prior-boxes)
        locs = torch.cat([l_conv11, l_conv13, l_conv14_2, l_conv15_2, l_conv16_2, l_conv17_2,l_conv18_2], dim=1)  # (N, 2268, 4)  , l_conv16_2, l_conv17_2
        classes_scores = torch.cat([c_conv11, c_conv13, c_conv14_2, c_conv15_2, c_conv16_2, c_conv17_2,c_conv18_2],
                                   dim=1)  # (N, 2268, n_classes) , c_conv16_2, c_conv17_2

        return locs, classes_scores


# auxiliary_conv = [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]

class LW_AuxillaryConvolutions(nn.Module):

    def __init__(self, backbone_net):
        super(LW_AuxillaryConvolutions, self).__init__()

        if backbone_net == "MobileNetV2":
            self.extras = ModuleList([
                Sequential(  # 5*5
                    Conv2d(in_channels=1280, out_channels=256, kernel_size=1),
                    ReLU(),
                    Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
                    ReLU()
                ),
                Sequential(  # 3*3
                    Conv2d(in_channels=512, out_channels=128, kernel_size=1),
                    ReLU(),
                    Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                    ReLU()
                )
                ,
                Sequential( #2*2
                    Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                    ReLU(),
                    Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                    ReLU()
                ),
                Sequential( #1*1
                    Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                    ReLU(),
                    Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                    ReLU()
                ),
                Sequential(  # 1*1
                    Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                    ReLU(),
                    Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                    ReLU()
                )
            ])

            self.init_conv2d()

        elif backbone_net == "MobileNetV1":
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
                ,
                Sequential( # 2*2
                    Conv2d(in_channels=256, out_channels=128, kernel_size=1), #3*3*128
                    ReLU(),
                    Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1), #2*2*256
                    ReLU()
                ),
                Sequential( # 1*1
                    Conv2d(in_channels=256, out_channels=128, kernel_size=1), #2*2*128
                    ReLU(),
                    Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1), #1*1*256
                    ReLU()
                )
            ])

            self.init_conv2d()

    def init_conv2d(self):
        for c in self.children():
            for layer in c:
                if isinstance(layer, nn.Conv2d):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(c.bias, 0.)

    def forward(self, LW_inp_features_17x17):
        LW_features = []
        x = LW_inp_features_17x17
        for layer in self.extras:
            x = layer(x)
            LW_features.append(x)

        LW_features_9x9 = LW_features[0]
        LW_features_5x5 = LW_features[1]
        LW_features_3x3 = LW_features[2]
        LW_features_2x2 = LW_features[3]
        LW_features_1x1 = LW_features[4]
        return LW_features_9x9,LW_features_5x5, LW_features_3x3 , LW_features_2x2, LW_features_1x1


class LW_DeConvolutions(nn.Module):

    def __init__(self):
        super(LW_DeConvolutions, self).__init__()

        def conv_3(inp):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, 1, 1),  # 1.0.weight
                nn.BatchNorm2d(inp),  # 1.1.weight 1.1.bias
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, inp, 3, 1, 1),  # 1.3.weight
                nn.BatchNorm2d(inp)  # 1.4.weight 1.4.bias

            )

        def late_conv_3(inp):
            return nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(inp, inp, 3, 1, 1),  # 1.0.weight
                nn.BatchNorm2d(inp),  # 1.1.weight 1.1.bias
                nn.ReLU(inplace=True),

            )

        def dconv(inp, oup):
            return nn.ConvTranspose2d(inp, oup, kernel_size=2, stride=2, padding=1, output_padding=1)

        # self.refine1 = conv_3(256)
        # self.refine2 = conv_3(256)
        self.refine3 = conv_3(256)
        self.refine5 = conv_3(256)
        self.refine9 = conv_3(512)
        self.refine17 = conv_3(1280)
        self.refine33 = conv_3(96)
        # self.dconv1 = dconv(256, 256, 0, 0)
        # self.dconv2 = dconv(256, 256, 1, 1)
        self.dconv3 = dconv(256, 256)
        self.dconv5 = dconv(256, 512)
        self.dconv9 = dconv(512, 1280)
        self.dconv17 = dconv(1280, 96)
        # self.late1 = late_conv_3(256)
        # self.late2 = late_conv_3(256)
        self.late3 = late_conv_3(256)
        self.late5 = late_conv_3(256)
        self.late9 = late_conv_3(512)
        self.late17 = late_conv_3(1280)
        self.late33 = late_conv_3(96)
        self.init_conv2d()

    def init_conv2d(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                for layer in c:  #
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0.)
            if isinstance(c, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)
            if isinstance(c, nn.BatchNorm2d):
                c.weight.data.fill_(1)
                c.bias.data.zero_()
        # for layer in c:
        #     if isinstance(layer, nn.Conv2d):
        #         nn.init.xavier_uniform_(layer.weight)
        #         nn.init.constant_(layer.bias, 0.)
        #     elif isinstance(layer, nn.ConvTranspose2d):
        #         nn.init.xavier_uniform_(layer.weight)
        #         nn.init.constant_(layer.bias, 0.)
        #     elif isinstance(layer, nn.BatchNorm2d):
        #         layer.weight.data.fill_(1)
        #         layer.bias.data.zero_()

    def forward(self, LW_features_33x33, LW_features_17x17,LW_features_9x9 ,LW_features_5x5,
                LW_features_3x3, LW_features_2x2, LW_features_1x1):  # , LW_features_2x2, LW_features_1x1

        # LW_refine1_1x1 = self.refine1(LW_features_1x1)
        # LW_refine_1x1 = self.late1(LW_refine1_1x1)
        # LW_refine_1x1=LW_features_1x1
        # LW_dconv_1x1 = self.dconv1(LW_refine_1x1)
        # LW_refine1_2x2 = self.refine2(LW_features_2x2)
        # x2=LW_refine1_2x2 + LW_dconv_1x1+LW_features_2x2
        # LW_refine_2x2 = self.late2(x2)
        # LW_dconv_2x2 = self.dconv2(LW_refine_2x2)
        # LW_refine1_3x3= self.refine3(LW_features_3x3)
        # x3=LW_refine1_3x3+LW_dconv_2x2+LW_features_3x3
        # LW_refine_3x3 = self.late3(x3)
        LW_refine_3x3 = LW_features_3x3
        LW_dconv_3x3 = self.dconv3(LW_refine_3x3)
        LW_refine1_5x5 = self.refine5(LW_features_5x5)
        x5 = LW_refine1_5x5 + LW_dconv_3x3 + LW_features_5x5
        LW_refine_5x5 = self.late5(x5)
        LW_dconv_5x5 = self.dconv5(LW_features_5x5)
        LW_refine1_9x9 = self.refine9(LW_features_9x9)
        x9 = LW_refine1_9x9 + LW_dconv_5x5 + LW_features_9x9
        LW_refine_9x9 = self.late9(x9)
        LW_dconv_9x9 = self.dconv9(LW_features_9x9)
        LW_refine1_17x17 = self.refine17(LW_features_17x17)
        x17 = LW_refine1_17x17 + LW_dconv_9x9 + LW_features_17x17
        LW_refine_17x17 = self.late17(x17)
        LW_dconv_17x17 = self.dconv17(LW_features_17x17)
        LW_refine1_33x33 = self.refine33(LW_features_33x33)
        x33 = LW_refine1_33x33 + LW_dconv_17x17 + LW_features_33x33
        LW_refine_33x33 = self.late33(x33)

        LW_refine_2x2=LW_features_2x2
        LW_refine_1x1=LW_features_1x1
        return LW_refine_33x33, LW_refine_17x17,LW_refine_9x9, LW_refine_5x5, LW_refine_3x3 , LW_refine_2x2, LW_refine_1x1


class SSD_LW(nn.Module):
    def __init__(self, num_classes, backbone_network):
        super(SSD_LW, self).__init__()

        self.num_classes = num_classes
        self.priors = torch.FloatTensor(priors).to(device)

        # self.base_net = MobileNetV1().model
        self.backbone_net = backbone_network
        if self.backbone_net == 'MobileNetV1':
            self.LW_base_net = LW_MobileNetV1().model
        elif self.backbone_net == 'MobileNetV2':
            self.LW_base_net = MobileNetV2_pretrained('mobilenet_v2.pth.tar').model
        else:
            raise ('SSD cannot be created with the provided base network')
        # self.base_net = MobileNetV2()

        self.LW_aux_network = LW_AuxillaryConvolutions(self.backbone_net)
        self.LW_DeConvolutions = LW_DeConvolutions()
        self.LW_prediction_network = LW_PredictionConvolutions(num_classes, self.backbone_net)
        # self.conv19 = nn.Conv2d(1024, 512, 1, 1, 0, bias=False)
        # self.conv10 = nn.Conv2d(2048, 1024, 1, 1, 0, bias=False)
        # self.conv5 = nn.Conv2d(1024, 512, 1, 1, 0, bias=False)
        # self.conv3 = nn.Conv2d(512, 256, 1, 1, 0, bias=False)
        # self.conv2 = nn.Conv2d(512, 256, 1, 1, 0, bias=False)
        # self.conv1 = nn.Conv2d(512, 256, 1, 1, 0, bias=False)

    def forward(self, LW_image):

        x = LW_image
        if self.backbone_net == 'MobileNetV1':
            source_layer_indexes = [
                12,
                14, ]
            start_layer_index = 0
            flag = 0
            # x = x.to('cuda')
            for end_layer_index in source_layer_indexes:
                for layer in self.LW_base_net[start_layer_index: end_layer_index]:
                    x = layer(x)
                layer_output = x
                start_layer_index = end_layer_index
                if flag == 0:
                    LW_features_33x33 = layer_output
                elif flag == 1:
                    LW_features_17x17 = layer_output
                flag += 1
            for layer in self.LW_base_net[end_layer_index:]:
                x = layer(x)

        elif self.backbone_net == 'MobileNetV2':
            for index, feat in enumerate(self.LW_base_net.features):
                x = feat(x)
                if index == 13:
                    LW_features_33x33 = x
                if index == 18:
                    LW_features_17x17 = x

        layer_output = x
        LW_features_9x9,LW_features_5x5, LW_features_3x3, LW_features_2x2, LW_features_1x1 = self.LW_aux_network(layer_output)  # , LW_features_2x2, LW_features_1x1
        LW_refine_33x33, LW_refine_17x17,LW_refine_9x9, LW_refine_5x5, LW_refine_3x3,LW_refine_2x2, LW_refine_1x1 = self.LW_DeConvolutions(LW_features_33x33,
                                                                                                     LW_features_17x17,LW_features_9x9,
                                                                                                     LW_features_5x5,
                                                                                                     LW_features_3x3,LW_features_2x2, LW_features_1x1)
        # , LW_refine_2x2, LW_refine_1x1  ,LW_features_2x2,LW_features_1x1
        # target_mean19 = torch.mean(LW_refine_33x33)
        # target_std19 = torch.std(LW_refine_33x33)
        # LW_refine_33x33 = LW_refine_33x33 - target_mean19
        # LW_refine_33x33 = LW_refine_33x33 / target_std19
        #
        # target_mean10 = torch.mean(LW_refine_17x17)
        # target_std10 = torch.std(LW_refine_17x17)
        # LW_refine_17x17 = LW_refine_17x17 - target_mean10
        # LW_refine_17x17 = LW_refine_17x17 / target_std10
        #
        # target_mean5 = torch.mean(LW_refine_5x5)
        # target_std5 = torch.std(LW_refine_5x5)
        # LW_refine_5x5 = LW_refine_5x5 - target_mean5
        # LW_refine_5x5 = LW_refine_5x5 / target_std5
        #
        # target_mean3 = torch.mean(LW_refine_3x3)
        # target_std3 = torch.std(LW_refine_3x3)
        # LW_refine_3x3 = LW_refine_3x3 - target_mean3
        # LW_refine_3x3 = LW_refine_3x3 / target_std3
        #
        # target_mean1 = torch.mean(LW_refine_1x1)
        # target_std1 = torch.std(LW_refine_1x1)
        # LW_refine_1x1 = LW_refine_1x1 - target_mean1
        # LW_refine_1x1 = LW_refine_1x1 / target_std1
        #
        # target_mean2 = torch.mean(LW_refine_2x2)
        # target_std2 = torch.std(LW_refine_2x2)
        # LW_refine_2x2 = LW_refine_2x2 - target_mean2
        # LW_refine_2x2 = LW_refine_2x2 / target_std2

        # FUSION_features_17x17 = torch.cat((LW_features_17x17, LW_refine_17x17), dim=1)
        # features_17x17 = self.conv10(FUSION_features_17x17)
        # FUSION_features_33x33 = torch.cat((LW_features_33x33, LW_refine_33x33), dim=1)
        # # return FUSION_features_17x17.size(1), FUSION_features_33x33.size(1)
        # # conv19 = nn.Conv2d(1024, 512, 1, 1, 0, bias=False)
        # features_33x33 = self.conv19(FUSION_features_33x33)
        # # return FUSION_features_17x17.size(), FUSION_features_33x33.size()
        # # layer_output = LW_features_17x17
        # # layer_output1 = LW_features_17x17
        # # LW_features_5x5, LW_features_3x3, LW_features_2x2, LW_features_1x1 = self.rgb_aux_network.forward(
        # #     layer_output)
        # # LW_features_5x5, LW_features_3x3, LW_features_2x2, LW_features_1x1 = self.lw_aux_network.forward(layer_output1)
        #
        # FUSION_features_5x5 = torch.cat((LW_features_5x5, LW_refine_5x5), dim=1)
        # features_5x5 = self.conv5(FUSION_features_5x5)
        # FUSION_features_3x3 = torch.cat((LW_features_3x3, LW_refine_3x3), dim=1)
        # features_3x3 = self.conv3(FUSION_features_3x3)
        # FUSION_features_2x2 = torch.cat((LW_features_2x2, LW_refine_2x2), dim=1)
        # features_2x2 = self.conv2(FUSION_features_2x2)
        # FUSION_features_1x1 = torch.cat((LW_features_1x1, LW_refine_1x1), dim=1)
        # features_1x1 = self.conv1(FUSION_features_1x1)
        # features = []
        # features.append(LW_features_33x33)  # torch.Size([1, 512, 19, 19])
        # features.append(LW_features_17x17)  # torch.Size([1, 1024, 10, 10])
        # features.append(LW_features_5x5)  # torch.Size([1, 512, 5, 5])
        # features.append(LW_features_3x3)  # torch.Size([1, 256, 3, 3])
        # features.append(LW_features_2x2)  # torch.Size([1, 256, 2, 2])
        # features.append(LW_features_1x1)  # torch.Size([1, 256, 1, 1])
        # features_33x33 = LW_features_33x33 + LW_refine_33x33
        # features_17x17 = LW_features_17x17 + LW_refine_17x17
        # features_5x5 = LW_features_5x5 + LW_refine_5x5
        # features_3x3 = LW_features_3x3 + LW_refine_3x3
        # features_2x2 = LW_features_2x2 + LW_refine_2x2
        # features_1x1 = LW_features_1x1 + LW_refine_1x1
        # LW_locs, LW_class_scores = self.LW_prediction_network.forward(LW_features_33x33, LW_features_17x17, LW_features_5x5, LW_features_3x3,
        #                                                      LW_features_2x2, LW_features_1x1)
        LW_locs, LW_class_scores = self.LW_prediction_network.forward(LW_refine_33x33, LW_refine_17x17,LW_refine_9x9, LW_refine_5x5, LW_refine_3x3,LW_refine_2x2, LW_refine_1x1)
        return LW_locs, LW_class_scores


# '''
# import numpy as np
# import torch
# import time
# from PIL import Image, ImageDraw, ImageFont
# from torchvision import transforms
#
# # img = np.random.rand(1, 3, 300, 300)
# # img = torch.Tensor(img)
# img_path = '/home/fereshteh/codergb/I02658.png'
# original_image = Image.open(img_path, mode='r')
# original_image = original_image.convert('LW')
# resize = transforms.Resize((300, 300))
# to_tensor = transforms.ToTensor()
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
#
# image = normalize(to_tensor(resize(original_image)))
# # image = (to_tensor(resize(original_image)))
#
# # Move to default device
# image = image.to(device)
#
# model = SSD_LW(2, 'MobileNetV1')
# model = model.to(device)
# s1 = time.time()
# loc, classes = model.forward(image.unsqueeze(0))
# print(loc.shape, classes.shape)
# s2 = time.time()
# s3 = s2 - s1
# print(s3)
# #
# # # print (loc.shape, classes.shape)
# # for param_name, param in model.LW_DeConvolutions.named_parameters():
# #     print(param_name)
# #     # if (param_name=='LW_prediction_network.cl_conv17_2.bias'):
# # #     param.copy_(state_dict['LW_aux_network.'+param_name])
# # '''
