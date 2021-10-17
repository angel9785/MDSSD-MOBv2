
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
from datasets_lw import PascalVOCDataset
from utils_lw import *
# from mobilev2ssd import SSD
from rgb_model2 import SSD_RGB
import argparse

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
    model.train()  # training mode enables dropout

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
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss

        # for i in range(len(boxes)):
        #  boxes[i] = boxes[i].to('cpu')
        #  labels[i] = labels[i].to('cpu')
        # print (predicted_locs, predicted_scores)
        # print (predicted_locs.shape, predicted_scores.shape)
        # print (len(boxes), len(labels))
        # print (boxes[1], labels[1])
        predicted_locs = predicted_locs.to(device)
        predicted_scores = predicted_scores.to(device)


        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
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
                  'train_Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
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
                      'val_Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(val_loader),
                                                                          batch_time=batch_time,
                                                                          loss=losses))

    print('\n * LOSS - {loss.avg:.3f}\n'.format(loss=losses))

    return losses.avg


root ="/content/drive/My Drive/data/VOC"

data_folder = '/content/drive/My Drive/data/VOC'

keep_difficult = True  # use objects considered difficult to detect?

voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()} # Inverse mapping

def main(args):
    # Model parameters
    # Not too many here since the SSD300 has a very specific structure
    with open(args.config_file_path, "r") as fp:
        config = json.load(fp)

    n_classes = len(label_map)  # number of different types of objects
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Mobilenetv2
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])

    # Learning parameters
    # checkpoint = torch.load('voc_checkpoint_ssd3001_2.pth.tar')  # path to model checkpoint, None if none
    batch_size = config['batch_size']  # batch size
    start_epoch = 0  # start at this epoch
    epochs = config['n_epochs']  # number of epochs to run without early-stopping
    epochs_since_improvement = 0  # number of epochs since there was an improvement in the validation metric
    best_loss = 100.  # assume a high loss at first
    workers = 4  # number of workers for loading data in the DataLoader
    lr = config['lr']  # learning rate
    momentum = 0.9  # momentum
    weight_decay = config['weight_decay']  # weight decay
    grad_clip = None # clip if g
    backbone_network = config['backbone_network']

    model = SSD_RGB(num_classes=n_classes, backbone_network=backbone_network)
    # model.load_state_dict(checkpoint['model'])
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
    # optimizer.load_state_dict(checkpoint['optimizer'])

    criterion = MultiBoxLoss(priors_cxcy=model.priors).to(device)

    # voc07_path = 'VOCdevkit/VOC2007'
    voc07_path = config['voc07_path']

    # voc12_path = 'VOCdevkit/VOC2012'
    voc12_path = config['voc12_path']
    # from utils import create_data_lists

    # create_data_lists(voc07_path, voc12_path, output_folder=config['data_folder'])

    # data_folder = 'VOC/VOCdevkit/'
    data_folder = config['data_folder']
    train_dataset = PascalVOCDataset(data_folder,
                                     split='train_lw',
                                     keep_difficult=keep_difficult)
    val_dataset = PascalVOCDataset(data_folder,
                                   split='test_lw',
                                   keep_difficult=keep_difficult)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                             collate_fn=val_dataset.collate_fn, num_workers=workers,
                                             pin_memory=True)

    print (start_epoch)
    for epoch in range(start_epoch, epochs):
        # Paper describes decaying the learning rate at the 80000th, 100000th, 120000th 'iteration', i.e. model update or batch
        # The paper uses a batch size of 32, which means there were about 517 iterations in an epoch
        # Therefore, to find the epochs to decay at, you could do,
        # if epoch in {80000 // 517, 100000 // 517, 120000 // 517}:
        #     adjust_learning_rate(optimizer, 0.1)

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
        val_loss = validate(val_loader=val_loader,
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
    parser = argparse.ArgumentParser()

    # parser.add_argument('backbone_network',help='Base model for extracting features for SSD. Must be one of ["MobileNetV1", "MobileNetV2"]')
    parser.add_argument('config_file_path' ,help='configuration file')
    args = parser.parse_args()

    main(args)


