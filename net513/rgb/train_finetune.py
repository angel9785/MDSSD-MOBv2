
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
from loss import MultiBoxLoss
from datasets import KAISTdataset
from utils import *
from rgb_model513_v2 import SSD_RGB
import argparse
import math
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
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
    for i, (images, boxes, labels, _) in enumerate(train_loader):

        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores= model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        predicted_locs = predicted_locs.to(device)
        predicted_scores = predicted_scores.to(device)

        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar
        if (math.isnan(loss)):
            print("Hi")
            continue











        if (loss != 0 ):#& (math.isnan(loss)==False)

            optimizer.zero_grad()

            loss.backward()


        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
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
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored


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
        for i, (images, boxes, labels, difficulties) in enumerate(val_loader):

            # Move to default device
            images = images.to(device)  # (N, 3, 300, 300)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # Forward prop.
            predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

            # Loss
            predicted_locs = predicted_locs.to(device)
            predicted_scores = predicted_scores.to(device)

            loss = criterion(predicted_locs, predicted_scores, boxes, labels)

            losses.update(loss.item(), images.size(0))
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



data_folder = '/home/fereshteh/code_513/rgbmodel'


keep_difficult = True
KAIST_labels = ('nonperson' ,'person')

label_map = {k: v for v, k in enumerate(KAIST_labels)}

rev_label_map = {v: k for k, v in label_map.items()} # Inverse mapping

def main():
    # Model parameters
    # Not too many here since the SSD300 has a very specific structure
    config_file_path ="/home/fereshteh/code_513/rgbmodel/config.json"
    with open(config_file_path, "r") as fp:
        config = json.load(fp)

    n_classes = len(label_map)  # number of different types of objects
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load("./BEST_513_mob2_fine_checkpoint_ssd_rgb_bn_new6.pth.tar", map_location=device)

    # checkpoint = None
    # checkpoint = torch.load('./BEST_513_mob2_fine_checkpoint_ssd_rgb_bn.pth.tar')
    # state_dict = 'BEST_voc_checkpoint_ssd3001_3.pth.tar'
    # batch_size = config['batch_size']  # batch size
    # batch_size = config['batch_size']  # batch size
    batch_size = 10 # batch size

    start_epoch = checkpoint['epoch']+1  # start at this epoch
    # start_epoch =0
    epochs =65
    epochs_since_improvement = 0  # number of epochs since there was an improvement in the validation metric
    best_loss = 100.  # assume a high loss at first
    workers = 4  # number of workers for loading data in the DataLoader
    # lr = config['lr']  # learning rate
    lr = 5e-5
    momentum = 0.9  # momentum
    weight_decay = config['weight_decay']  # weight decay
    grad_clip = None # clip if g

    model = SSD_RGB(num_classes=n_classes,backbone_network="MobileNetV2")
    # global model
    # state_dict=checkpoint(["model")
    # state_dict = checkpoint['model']
    # with torch.no_grad():
    #     for param_name, param in model.RGB_base_net.named_parameters():
    #
    #         param.copy_(state_dict['RGB_base_net.'+param_name])
    #     for param_name, param in model.RGB_aux_network.named_parameters():
    #
    #         param.copy_(state_dict['RGB_aux_network.'+param_name])

    # checkpoint = torch.load('./mobilenet_v2.pth.tar', map_location=device)
    # model.RGB_base_net.load_state_dict(checkpoint)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
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
    optimizer.load_state_dict(checkpoint['optimizer'])  # Optimizing parameters
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors).to(device)

    kaist_path ='/home/fereshteh/kaist'

    data_folder = config['data_folder']
    # create_data_lists(kaist_path, output_folder=data_folder)

    sanitrain_dataset = KAISTdataset(data_folder,
                                 split='sanitrain',
                                 keep_difficult=keep_difficult)

    test_dataset = KAISTdataset(data_folder,
                                split='sanival',
                                keep_difficult=keep_difficult)


    sanitrain_loader = torch.utils.data.DataLoader(sanitrain_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=sanitrain_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here

    sanitest_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                             collate_fn=test_dataset.collate_fn, num_workers=workers,
                                             pin_memory=True)

    # adjust_learning_rate(optimizer, 0.1)
    for epoch in range(start_epoch, epochs):

        if (epoch == 2):
            adjust_learning_rate(optimizer, 20)
        if (epoch == 16):
            adjust_learning_rate(optimizer, 0.1)
        if (epoch == 36):
            adjust_learning_rate(optimizer, 0.1)
        # if (epoch == 22):
        #     sanitrain_loader = torch.utils.data.DataLoader(sanitrain_dataset, batch_size=8, shuffle=True,
        #                                                    collate_fn=sanitrain_dataset.collate_fn, num_workers=workers,
        #                                                    pin_memory=True)  # note that we're passing the collate function here

        train(train_loader=sanitrain_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch,
              grad_clip= grad_clip)


        # One epoch's validation
        val_loss = validate(val_loader=sanitest_loader,
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


