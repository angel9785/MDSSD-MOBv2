
# !/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  10 15:45:16 2019

@author: viswanatha
"""
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
# from mobilessd import SSD
from loss import MultiBoxLoss
# from model import SSD300, MultiBoxLoss
from datasets import KAISTdataset
from utils import *
# from second_model_sum import SSD_FUSION
from fusion_513_v2 import SSD_FUSION
import argparse
import math
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# checkpoint_rgb = torch.load('/home/fereshteh/codergb_new/new_fine_checkpoint_ssd3001_10.pth.tar')
# checkpoint_rgb = torch.load('/home/fereshteh/codergb_new/fine_checkpoint_ssd300_7.pth.tar')
# checkpoint_rgb = torch.load('/home/fereshteh/code_513/rgbmodel/513_mob2_fine_checkpoint_ssd_rgb_bn_new5.pth.tar')
# state_dict_RGB = checkpoint_rgb['model']
# # checkpoint_lw = torch.load('/home/fereshteh/codelw_new/new_fine_checkpoint_ssd300_lw_5.pth.tar')
# # checkpoint_lw = torch.load('/home/fereshteh/codelw_new/BEST_fine_checkpoint_ssd300_lw_3.pth.tar')
# checkpoint_lw = torch.load('/home/fereshteh/code_513/lw/513mob2_fine_checkpoint_ssd_lw_bn_new6.pth.tar')
# state_dict_LW = checkpoint_lw['model']
def train(train_loader, model, criterion, optimizer, epoch, grad_clip):
    """
    One epoch's training.
    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    print_freq = 200  # print training or validation status every __ batches

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (im, images_rgb, images_lw, boxes, labels, _) in enumerate(train_loader):
        # if (i==8000):
        #     adjust_learning_rate(optimizer, 0.1)
        data_time.update(time.time() - start)

        # Move to default device
        images_rgb = images_rgb.to(device)  # (batch_size (N), 3, 300, 300)
        images_lw = images_lw.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images_rgb, images_lw)  # (N, 8732, 4), (N, 8732, n_classes)

        predicted_locs = predicted_locs.to(device)
        predicted_scores = predicted_scores.to(device)

        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar
        if (math.isnan(loss)):
            continue

        # Backward prop.
        if (loss != 0):
            optimizer.zero_grad()

            loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images_rgb.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'trainLoss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                       batch_time=batch_time,
                                                                       data_time=data_time, loss=losses))
    del predicted_locs, predicted_scores, images_rgb, images_lw, boxes, labels  # free some memory since their histories may be stored


def validate(val_loader, model, criterion):
    """
    One epoch's validation.
    :param val_loader: DataLoader for validation data
    :param model: model
    :param criterion: MultiBox loss
    :return: average validation loss
    """
    print_freq = 200
    model.eval()  # eval mode disables dropout

    batch_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()

    # Prohibit gradient computation explicity because I had some problems with memory
    with torch.no_grad():
        # Batches
        for i, (im, images_rgb, images_lw, boxes, labels, difficulties) in enumerate(val_loader):

            # Move to default device
            images_rgb = images_rgb.to(device)  # (N, 3, 300, 300)
            images_lw = images_lw.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # Forward prop.
            predicted_locs, predicted_scores = model(images_rgb, images_lw)  # (N, 8732, 4), (N, 8732, n_classes)

            # Loss
            predicted_locs = predicted_locs.to(device)
            predicted_scores = predicted_scores.to(device)

            loss = criterion(predicted_locs, predicted_scores, boxes, labels)

            losses.update(loss.item(), images_rgb.size(0))
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if i % print_freq == 0:
                print('[{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'validateLoss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(val_loader),
                                                                              batch_time=batch_time,
                                                                              loss=losses))

    print('\n * LOSS - {loss.avg:.3f}\n'.format(loss=losses))

    return losses.avg


# root="/home/fereshteh/codergb"

data_folder = '/home/fereshteh/code_513/fusion'

# keep_difficult = True  # use objects considered difficult to detect?
keep_difficult = True
voc_labels = ('nonperson' ,'person')
# voc_labels = ('person', 'cyclist')
label_map = {k: v for v, k in enumerate(voc_labels)}
# label_map['background'] = 0
# label_map['person'] = 1
rev_label_map = {v: k for k, v in label_map.items()} # Inverse mapping

def main():
    # Model parameters
    # Not too many here since the SSD300 has a very specific structure
    config_file_path ="/home/fereshteh/code_513/fusion/config.json"
    with open(config_file_path, "r") as fp:
        config = json.load(fp)

    n_classes = len(label_map)  # number of different types of objects
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mobilenetv2
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])

    # Learning parameters

    checkpoint = torch.load('./mobilenet_v2.pth.tar')
    # checkpoint = torch.load('./BEST_dec_mob2_NEW_fine_checkpoint_ssd300_fusion_bn8.pth.tar')

    batch_size = 3  # batch size
    start_epoch = 0  # start at this epoch
    # start_epoch = checkpoint['epoch' ] +1
    epochs = config['n_epochs']  # number of epochs to run without early-stopping
    epochs =65
    epochs_since_improvement = 0  # number of epochs since there was an improvement in the validation metric
    best_loss = 100.  # assume a high loss at first
    workers = 4  # number of workers for loading data in the DataLoader
    # lr = config['lr']  # learning rate
    lr = 5e-5
    # lr=1e-4
    momentum = 0.9  # momentum
    weight_decay = config['weight_decay']  # weight decay
    grad_clip = None # clip if g
    backbone_network = config['backbone_network']

    model = SSD_FUSION(num_classes=n_classes, backbone_network=backbone_network)
    model.RGB_base_net.load_state_dict(checkpoint)
    model.LW_base_net.load_state_dict(checkpoint)
    # with torch.no_grad():
    #
    #     for param_name, param in model.RGB_base_net.named_parameters():
    #         param.copy_(state_dict_RGB[ "RGB_base_net."+param_name])
    #
    #     for param_name, param in model.LW_base_net.named_parameters():
    #         param.copy_(state_dict_LW[ "LW_base_net."+param_name])
    #
    #     for param_name, param in model.LW_aux_network.named_parameters():
    #         param.copy_(state_dict_LW["LW_aux_network." + param_name])
    #
    #     for param_name, param in model.RGB_aux_network.named_parameters():
    #         param.copy_(state_dict_RGB["RGB_aux_network." + param_name])

        # # for param_name, param in model.FUSION_DeConvolutions.refine2_rgb.named_parameters():
        # #     param.copy_(state_dict_RGB["RGB_DeConvolutions.refine2." + param_name])
        # # for param_name, param in model.FUSION_DeConvolutions.refine3_rgb.named_parameters():
        # #     param.copy_(state_dict_RGB["RGB_DeConvolutions.refine3." + param_name])
        # for param_name, param in model.FUSION_DeConvolutions.refine5_rgb.named_parameters():
        #     param.copy_(state_dict_RGB["RGB_DeConvolutions.refine5." + param_name])
        # for param_name, param in model.FUSION_DeConvolutions.refine9_rgb.named_parameters():
        #     param.copy_(state_dict_RGB["RGB_DeConvolutions.refine9." + param_name])
        # for param_name, param in model.FUSION_DeConvolutions.refine17_rgb.named_parameters():
        #     param.copy_(state_dict_RGB["RGB_DeConvolutions.refine17." + param_name])
        # for param_name, param in model.FUSION_DeConvolutions.refine33_rgb.named_parameters():
        #     param.copy_(state_dict_RGB["RGB_DeConvolutions.refine33." + param_name])
        #
        # # for param_name, param in model.FUSION_DeConvolutions.refine2_lw.named_parameters():
        # #     param.copy_(state_dict_LW["LW_DeConvolutions.refine2." + param_name])
        # # for param_name, param in model.FUSION_DeConvolutions.refine3_lw.named_parameters():
        # #     param.copy_(state_dict_LW["LW_DeConvolutions.refine3." + param_name])
        # for param_name, param in model.FUSION_DeConvolutions.refine5_lw.named_parameters():
        #     param.copy_(state_dict_LW["LW_DeConvolutions.refine5." + param_name])
        # for param_name, param in model.FUSION_DeConvolutions.refine9_lw.named_parameters():
        #     param.copy_(state_dict_LW["LW_DeConvolutions.refine9." + param_name])
        # for param_name, param in model.FUSION_DeConvolutions.refine17_lw.named_parameters():
        #     param.copy_(state_dict_LW["LW_DeConvolutions.refine17." + param_name])
        # for param_name, param in model.FUSION_DeConvolutions.refine33_lw.named_parameters():
        #     param.copy_(state_dict_LW["LW_DeConvolutions.refine33." + param_name])
    # global model
    # state_dict = checkpoint['model']
    # with torch.no_grad():
    #     for param_name, param in model.RGB_base_net.named_parameters():
    #         # print(param_name)
    #         # if (param_name=='RGB_prediction_network.cl_conv17_2.bias'):
    #         param.copy_(state_dict['RGB_base_net.' + param_name])
    #     for param_name, param in model.RGB_aux_network.named_parameters():
    #         # print(param_name)
    #         # if (param_name=='RGB_prediction_network.cl_conv17_2.bias'):
    #         param.copy_(state_dict['RGB_aux_network.' + param_name])
    # model.load_state_dict(state_dict_RGB,strict=False)
    # model.load_state_dict(state_dict_LW,strict=False)
    # model.load_state_dict(checkpoint['model'])  # Optimizing parameters

    model = model.to(device)
    # self.model = net
    # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
    biases = list()
    not_biases = list()
    param_names_biases = list()
    param_names_not_biases = list()
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
    # optimizer.load_state_dict(checkpoint['optimizer'])  # Optimizing parameters
    # model.load_state_dict(checkpoint['model'])
    # state_dict = torch.load("/home/fereshteh/code_513/fusion/mobilenet_v2.pth.tar", map_location=device)
    # model.RGB_base_net.load_state_dict(state_dict)
    # model.LW_base_net.load_state_dict(state_dict)
    model = model.to(device)

    criterion = MultiBoxLoss(priors_cxcy=model.priors).to(device)

    kaist_path ='/home/fereshteh/kaist'

    # create_data_lists(kaist_path, output_folder=config['data_folder'])
    #
    # data_folder = 'VOC/VOCdevkit/'
    data_folder = config['data_folder']

    test_dataset = KAISTdataset(data_folder,
                                split='sanival',
                                keep_difficult=keep_difficult)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)
    train_dataset = KAISTdataset(data_folder,
                                split='sanitrain1',
                                keep_difficult=keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                              collate_fn=train_dataset.collate_fn, num_workers=workers, pin_memory=True)

    # print (start_epoch)
    # adjust_learning_rate(optimizer,1)
    # # print(lr)
    # # optimizer.lr = 1e-5
    # adjust_learning_rate(optimizer,0.5)
    for epoch in range(start_epoch, epochs):
        # if (epoch ==1):
        #     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=3, shuffle=False,
        #                                                collate_fn=train_dataset.collate_fn, num_workers=workers,
        #                                                pin_memory=True)
        # if (epoch ==3):
        #     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=False,
        #                                                collate_fn=train_dataset.collate_fn, num_workers=workers,
        #                                                pin_memory=True)
        # if (epoch ==4):
        #     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=False,
        #                                                collate_fn=train_dataset.collate_fn, num_workers=workers,
        #                                                pin_memory=True)
        #
        # if (epoch == 6):
        #     adjust_learning_rate(optimizer, 0.1)
        # if (epoch ==8):
        #     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=False,
        #                                                collate_fn=train_dataset.collate_fn, num_workers=workers,
        #                                                pin_memory=True)
        #
        #
        # if (epoch ==4):
        #     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=False,
        #                                                collate_fn=train_dataset.collate_fn, num_workers=workers,
        #                                                pin_memory=True)
        # if (epoch ==6):
        #     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=False,
        #                                                collate_fn=train_dataset.collate_fn, num_workers=workers,
        #                                                pin_memory=True)
        # if (epoch ==4):
        #     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=False,
        #                                                collate_fn=train_dataset.collate_fn, num_workers=workers,
        #                                                pin_memory=True)
        # if (epoch ==6):
        #     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=6, shuffle=False,
        #                                                collate_fn=train_dataset.collate_fn, num_workers=workers,
        #                                                pin_memory=True)
        if (epoch ==2):

            adjust_learning_rate(optimizer, 20)
        if (epoch ==16):

            adjust_learning_rate(optimizer, 0.1)
        if (epoch == 20):
            adjust_learning_rate(optimizer, 0.1)

        if (epoch == 24):
            adjust_learning_rate(optimizer, 0.1)
        # if (epoch == 40):
        #     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=6, shuffle=False,
        #                                                collate_fn=train_dataset.collate_fn, num_workers=workers,
        #                                                pin_memory=True)

        # Paper describes decaying the learning rate at the 80000th, 100000th, 120000th 'iteration', i.e. model update or batch
        # The paper uses a batch size of 32, which means there were about 517 iterations in an epoch
        # Therefore, to find the epochs to decay at, you could do,
        # if epoch in {80000 // 517, 100000 // 517, 120000 // 517}:
        #     adjust_learning_rate(optimizer, 0.1)
        # if (epoch>0):
        # 	adjust_learning_rate(optimizer, 10)

        # In practice, I just decayed the learning rate when loss stopped improving for long periods,
        # and I would resume from the last best checkpoint with the new learning rate,
        # since there's no point in resuming at the most recent and significantly worse checkpoint.
        # So, when you're ready to decay the learning rate, just set checkpoint = 'BEST_checkpoint_ssd300.pth.tar' above
        # and have adjust_learning_rate(optimizer, 0.1) BEFORE this 'for' loop

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch,
              grad_clip= grad_clip)


        # One epoch's validation
        val_loss = validate(val_loader=test_loader,
                            model=model,
                            criterion=criterion)

        # Did validation loss improve?
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)

        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))

        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, val_loss, best_loss, is_best)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()

    # #parser.add_argument('backbone_network',help='Base model for extracting features for SSD. Must be one of ["MobileNetV1", "MobileNetV2"]')
    # parser.add_argument('config_file_path',help='configuration file')
    # args = parser.parse_args()

    main()


