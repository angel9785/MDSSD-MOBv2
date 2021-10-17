import torch.nn as nn
# import torch.add as add
import torch.nn.functional as F
from torch.nn import Conv2d, Sequential, ModuleList, ReLU
import torch
from itertools import product
from mobilenet_ssd_priors import *
from MobileNetV2 import MobileNetV2, MobileNetV2_pretrained
import time

# from torchvision.outograd import variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_rgb = torch.load('/home/fereshteh/codergb_new/fine_checkpoint_ssd300_7.pth.tar')
checkpoint_lw = torch.load('/home/fereshteh/codelw_new/BEST_fine_checkpoint_ssd300_lw_3.pth.tar')


class BaseNetwork(nn.Module):
    def __init__(self, num_classes=2):
        super(BaseNetwork, self).__init__()

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
        # def RGB_base_net():
        #
        #
        #     return RGB_base_net

        # def LW_base_net():
        #
        #
        #     return LW_base_net

        # self.module1 = RGB_base_net()
        # self.module2 = LW_base_net()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):

        x = input
        source_layer_indexes = [
            12,
            14, ]
        start_layer_index = 0
        flag = 0

        for end_layer_index in source_layer_indexes:
            for layer in self.model[start_layer_index: end_layer_index]:
                x = layer(x)
            layer_output = x
            start_layer_index = end_layer_index
            if flag == 0:
                features_19x19 = layer_output
                flag += 1
            elif flag == 1:
                features_10x10 = layer_output

        return features_19x19, features_10x10

    # def forward(self, x1, x2):
    #
    #
    #     x1 = x1.to('cuda')
    #
    #     x2 = x2.to('cuda')
    #
    #     ''''
    #     layer2=self.module2[start_layer_index]
    #     i=1
    #     for layer1 in self.module1[start_layer_index: 14]:
    #         x = [layer1(x[0]), layer2(x[1])]
    #         if i==12:
    #             RGB_features_19x19=x
    #         if i == 14:
    #             RGB_features_10x10 = x
    #         if (start_layer_index + i) < 14:
    #             layer2=self.module2[ start_layer_index+i]
    #             i = i + 1
    #     '''
    #
    #     source_layer_indexes = [
    #         12,
    #         14, ]
    #     start_layer_index = 0
    #     start_layer_index1 = 0
    #     flag = 0
    #     flag1=0
    #
    #     for end_layer_index in source_layer_indexes:
    #         for layer in self.module1[start_layer_index: end_layer_index]:
    #             x1 = layer(x1)
    #         layer_output = x1
    #         start_layer_index = end_layer_index
    #         if flag == 0:
    #             RGB_features_19x19 = layer_output
    #             flag += 1
    #         elif flag == 1:
    #             RGB_features_10x10 = layer_output
    #
    #     # for end_layer_index in source_layer_indexes:
    #         for layer1 in self.module2[start_layer_index1: end_layer_index]:
    #             x2 = layer1(x2)
    #         layer_output1 = x2
    #         start_layer_index1 = end_layer_index
    #         if flag1 == 0:
    #             LW_features_19x19 = layer_output1
    #             flag1 += 1
    #         elif flag1 == 1:
    #             LW_features_10x10 = layer_output1
    #
    #
    #     return RGB_features_19x19.size(), RGB_features_10x10.size(),LW_features_19x19.size(),LW_features_10x10.size()


class FUSION_PredictionConvolutions(nn.Module):
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
        super(FUSION_PredictionConvolutions, self).__init__()

        self.n_classes = n_classes

        # Number of prior-boxes we are considering per position in each feature map
        n_boxes = {'conv11': 4,
                   'conv13': 6,
                   'conv14_2': 6,
                   'conv15_2': 6,
                   'conv16_2': 4,
                   'conv17_2': 4}
        # 4 prior-boxes implies we use 4 different aspect ratios, etc.

        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        if backbone_net == 'MobileNetV2':
            self.loc_conv11 = nn.Conv2d(96, n_boxes['conv11'] * 4, kernel_size=3, padding=1)
            self.loc_conv13 = nn.Conv2d(1280, n_boxes['conv13'] * 4, kernel_size=3, padding=1)
            self.loc_conv14_2 = nn.Conv2d(512, n_boxes['conv14_2'] * 4, kernel_size=3, padding=1)
            self.loc_conv15_2 = nn.Conv2d(256, n_boxes['conv15_2'] * 4, kernel_size=3, padding=1)
            self.loc_conv16_2 = nn.Conv2d(256, n_boxes['conv16_2'] * 4, kernel_size=3, padding=1)
            self.loc_conv17_2 = nn.Conv2d(256, n_boxes['conv17_2'] * 4, kernel_size=3, padding=1)

            # Class prediction convolutions (predict classes in localization boxes)
            self.cl_conv11 = nn.Conv2d(96, n_boxes['conv11'] * n_classes, kernel_size=3, padding=1)
            self.cl_conv13 = nn.Conv2d(1280, n_boxes['conv13'] * n_classes, kernel_size=3, padding=1)
            self.cl_conv14_2 = nn.Conv2d(512, n_boxes['conv14_2'] * n_classes, kernel_size=3, padding=1)
            self.cl_conv15_2 = nn.Conv2d(256, n_boxes['conv15_2'] * n_classes, kernel_size=3, padding=1)
            self.cl_conv16_2 = nn.Conv2d(256, n_boxes['conv16_2'] * n_classes, kernel_size=3, padding=1)
            self.cl_conv17_2 = nn.Conv2d(256, n_boxes['conv17_2'] * n_classes, kernel_size=3, padding=1)

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

    def forward(self, conv11_feats, conv13_feats, conv14_2_feats, conv15_2_feats, conv16_2_feats, conv17_2_feats):
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

        # A total of 2268 boxes
        # Concatenate in this specific order (i.e. must match the order of the prior-boxes)
        locs = torch.cat([l_conv11, l_conv13, l_conv14_2, l_conv15_2, l_conv16_2, l_conv17_2], dim=1)  # (N, 2268, 4)
        classes_scores = torch.cat([c_conv11, c_conv13, c_conv14_2, c_conv15_2, c_conv16_2, c_conv17_2],
                                   dim=1)  # (N, 2268, n_classes)

        return locs, classes_scores


# auxiliary_conv = [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]

class FUSION_AuxillaryConvolutions(nn.Module):

    def __init__(self, backbone_net):
        super(FUSION_AuxillaryConvolutions, self).__init__()

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
                ),
                Sequential(  # 2*2
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
                ),
                Sequential(  # 2*2
                    Conv2d(in_channels=256, out_channels=128, kernel_size=1),  # 3*3*128
                    ReLU(),
                    Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),  # 2*2*256
                    ReLU()
                ),
                Sequential(  # 1*1
                    Conv2d(in_channels=256, out_channels=128, kernel_size=1),  # 2*2*128
                    ReLU(),
                    Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),  # 1*1*256
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

    def forward(self, FUSION_inp_features_10x10):
        FUSION_features = []
        x = FUSION_inp_features_10x10
        for layer in self.extras:
            x = layer(x)
            FUSION_features.append(x)

        FUSION_features_5x5 = FUSION_features[0]
        FUSION_features_3x3 = FUSION_features[1]
        FUSION_features_2x2 = FUSION_features[2]
        FUSION_features_1x1 = FUSION_features[3]
        return FUSION_features_5x5, FUSION_features_3x3, FUSION_features_2x2, FUSION_features_1x1


class SSD_FUSION(nn.Module):
    def __init__(self, num_classes, backbone_network):
        super(SSD_FUSION, self).__init__()

        self.num_classes = num_classes
        self.priors = torch.FloatTensor(priors).to(device)

        # self.base_net = MobileNetV1().model
        self.backbone_net = backbone_network
        # if self.backbone_net == 'MobileNetV1':
        #     self.RGB_base_net = RGB_MobileNetV1().model
        # elif self.backbone_net == 'MobileNetV2':
        #     self.RGB_base_net = MobileNetV2_pretrained('mobilenet_v2.pth.tar').model
        # else:
        #     raise ('SSD cannot be created with the provided base network')
        # self.base_net = MobileNetV2()

        self.RGB_base_net = BaseNetwork()
        self.LW_base_net = BaseNetwork()
        ''''
        state_dict_RGB = checkpoint_rgb['model']

        with torch.no_grad():
            for param_name, param in RGB_base_net.named_parameters():
                param.copy_(state_dict_RGB['RGB_base_net.' + param_name])
        LW_base_net = self.model

        state_dict_LW = checkpoint_lw['model']
        with torch.no_grad():
            for param_name, param in LW_base_net.named_parameters():
                param.copy_(state_dict_LW['LW_base_net.' + param_name])
                '''
        # self.BaseNetwork = BaseNetwork()
        self.FUSION_aux_network = FUSION_AuxillaryConvolutions(self.backbone_net)
        # self.lw_aux_network = FUSION_AuxillaryConvolutions(self.backbone_net)
        self.FUSION_prediction_network = FUSION_PredictionConvolutions(num_classes, self.backbone_net)
        self.conv19 = nn.Conv2d(1024, 512, 1, 1, 0)
        self.conv10 = nn.Conv2d(2048, 1024, 1, 1, 0)
        self.conv5 = nn.Conv2d(1024, 512, 1, 1, 0)
        self.conv3 = nn.Conv2d(512, 256, 1, 1, 0)
        self.conv2 = nn.Conv2d(512, 256, 1, 1, 0)
        self.conv1 = nn.Conv2d(512, 256, 1, 1, 0)

        # self.conv=conv()

    # def forward(self, RGB_image):
    def forward(self, x1, x2):
        x1 = x1.to(device)

        x2 = x2.to(device)
        RGB_features_19x19, RGB_features_10x10 = self.RGB_base_net.forward(x1)
        LW_features_19x19, LW_features_10x10 = self.LW_base_net.forward(x2)
        ''''
        layer2=self.module2[start_layer_index]
        i=1
        for layer1 in self.module1[start_layer_index: 14]:
            x = [layer1(x[0]), layer2(x[1])]
            if i==12:
                RGB_features_19x19=x
            if i == 14:
                RGB_features_10x10 = x
            if (start_layer_index + i) < 14:
                layer2=self.module2[ start_layer_index+i]
                i = i + 1
        '''

        # source_layer_indexes = [
        #     12,
        #     14, ]
        # start_layer_index = 0
        # start_layer_index1 = 0
        # flag = 0
        # flag1 = 0
        #
        # for end_layer_index in source_layer_indexes:
        #     for layer in self.RGB_base_net[start_layer_index: end_layer_index]:
        #         x1 = layer(x1)
        #     layer_output = x1
        #     start_layer_index = end_layer_index
        #     if flag == 0:
        #         RGB_features_19x19 = layer_output
        #         flag += 1
        #     elif flag == 1:
        #         RGB_features_10x10 = layer_output

        # for end_layer_index in source_layer_indexes:
        # for layer1 in self.lw_aux_network[start_layer_index1: end_layer_index]:
        #     x2 = layer1(x2)
        # layer_output1 = x2
        # start_layer_index1 = end_layer_index
        # if flag1 == 0:
        #     LW_features_19x19 = layer_output1
        #     flag1 += 1
        # elif flag1 == 1:
        #     LW_features_10x10 = layer_output1

        # return RGB_features_19x19.size(), RGB_features_10x10.size(), LW_features_19x19.size(), LW_features_10x10.size()
        # x = RGB_image
        # if self.backbone_net == 'MobileNetV1':
        #     source_layer_indexes = [
        #         12,
        #         14, ]
        #     start_layer_index = 0
        #     flag = 0
        #     # x = x.to('cuda')
        #     for end_layer_index in source_layer_indexes:
        #         for layer in self.RGB_base_net[start_layer_index: end_layer_index]:
        #             x = layer(x)
        #         layer_output = x
        #         start_layer_index = end_layer_index
        #         if flag == 0:
        #             RGB_features_19x19 = layer_output
        #         elif flag == 1:
        #             RGB_features_10x10 = layer_output
        #         flag += 1
        #     for layer in self.RGB_base_net[end_layer_index:]:
        #         x = layer(x)
        #
        # elif self.backbone_net == 'MobileNetV2':
        #     for index, feat in enumerate(self.RGB_base_net.features):
        #         x = feat(x)
        #         if index == 13:
        #             RGB_features_19x19 = x
        #         if index == 18:
        #             RGB_features_10x10 = x

        # FUSION_features_10x10 = torch.cat((RGB_features_10x10, LW_features_10x10), dim=1)
        # FUSION_features_10x10 = self.conv10(FUSION_features_10x10)
        FUSION_features_10x10 = RGB_features_10x10 + LW_features_10x10
        # FUSION_features_19x19 = torch.cat((RGB_features_19x19, LW_features_19x19), dim=1)
        # # return FUSION_features_10x10.size(1), FUSION_features_19x19.size(1)
        # # conv19 = nn.Conv2d(1024, 512, 1, 1, 0, bias=False)
        # FUSION_features_19x19 = self.conv19(FUSION_features_19x19)
        FUSION_features_19x19 = RGB_features_19x19 + LW_features_19x19
        # return FUSION_features_10x10.size(), FUSION_features_19x19.size()
        layer_output = FUSION_features_10x10
        # layer_output1 = LW_features_10x10
        FUSION_features_5x5, FUSION_features_3x3, FUSION_features_2x2, FUSION_features_1x1 = self.FUSION_aux_network.forward(
            layer_output)
        # LW_features_5x5, LW_features_3x3, LW_features_2x2, LW_features_1x1 = self.lw_aux_network.forward(layer_output1)

        # FUSION_features_5x5 = torch.cat((RGB_features_5x5, LW_features_5x5), dim=1)
        # FUSION_features_5x5 = self.conv5(FUSION_features_5x5)
        # FUSION_features_3x3 = torch.cat((RGB_features_3x3, LW_features_3x3), dim=1)
        # FUSION_features_3x3 = self.conv3(FUSION_features_3x3)
        # FUSION_features_2x2 = torch.cat((RGB_features_2x2, LW_features_2x2), dim=1)
        # FUSION_features_2x2 = self.conv2(FUSION_features_2x2)
        # FUSION_features_1x1 = torch.cat((RGB_features_1x1, LW_features_1x1), dim=1)
        # FUSION_features_1x1 = self.conv1(FUSION_features_1x1)
        features = []
        features.append(FUSION_features_19x19)  # torch.Size([1, 512, 19, 19])
        features.append(FUSION_features_10x10)  # torch.Size([1, 1024, 10, 10])
        features.append(FUSION_features_5x5)  # torch.Size([1, 512, 5, 5])
        features.append(FUSION_features_3x3)  # torch.Size([1, 256, 3, 3])
        features.append(FUSION_features_2x2)  # torch.Size([1, 256, 2, 2])
        features.append(FUSION_features_1x1)  # torch.Size([1, 256, 1, 1])

        FUSION_locs, FUSION_class_scores = self.FUSION_prediction_network.forward(FUSION_features_19x19,
                                                                                  FUSION_features_10x10,
                                                                                  FUSION_features_5x5,
                                                                                  FUSION_features_3x3,
                                                                                  FUSION_features_2x2,
                                                                                  FUSION_features_1x1)

        return FUSION_locs, FUSION_class_scores


'''''
img1 = torch.rand(1, 3, 300, 300)
img2 = torch.rand(1, 3, 300, 300)
model = SSD_FUSION(2, 'MobileNetV1')
model = model.to(device)
# print(model)
s1 = time.time()
#
# img1=img1.to(device)
# img2=img2.to(device)
FUSION_locs, FUSION_class_scores = model.forward(img1, img2)
# print(a,b,c,d)
# print(FUSION_locs, FUSION_class_scores)
s2 = time.time()
s3 = s2 - s1
print(s3)

# for param_name, param in model.named_parameters():
#     print(param_name)
'''


