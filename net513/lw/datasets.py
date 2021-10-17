
import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import transform
from utils import *

import cv2

class KAISTdataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()

        assert self.split in {'SANITEST','NSANITEST1','DSANITEST1','SANITEST1','SANIVAL','SANITRAIN'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Read data files
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)
        # with open(os.path.join(data_folder, 'SANITEST' + '_images.json'), 'r') as j:
        #     self.images1 = json.load(j)
        # with open(os.path.join(data_folder, 'SANITEST' + '_objects.json'), 'r') as j:
        #     self.objects1 = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        # Read image
        # image = Image.open(self.images[i], mode='r')
        image_lw = cv2.imread(self.images[i], 1)
        lab = cv2.cvtColor(image_lw, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        original_image_lw = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        color_coverted = cv2.cvtColor(original_image_lw, cv2.COLOR_BGR2RGB)
        original_image_lw = Image.fromarray(color_coverted)
        original_image_lw = original_image_lw.convert('RGB')
        image = original_image_lw.convert('RGB')

        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)
        # difficulties = torch.boolTensor(objects['difficulties'])  # (n_objects)
        # for i in range(0, len(boxes)):
        #     if ((image.width - boxes[i, 0]) < 0):
        #         print("")
        #     if ((image.width - boxes[i, 2]) < 0):
        #         print("")

        # Discard difficult objects, if desired
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]
        if (self.split == 'SANITRAIN'):
            if i<7601:
                image, boxes, labels, difficulties = transform1(image, boxes, labels, difficulties, split=self.split)
                # print("tr1")
                return image, boxes, labels, difficulties
            if 7600<i<15204:
                image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)
                # print("tr2")
                return image, boxes, labels, difficulties
            if 15203<i<22804:
                image, boxes, labels, difficulties = transform3(image, boxes, labels, difficulties, split=self.split)
                # print("tr3")
                return image, boxes, labels, difficulties
            if 22803<i<30404:
                image, boxes, labels, difficulties = transform4(image, boxes, labels, difficulties, split=self.split)
                # print("tr4")
                return image, boxes, labels, difficulties
        # Apply transformations
        else:
            image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)

            return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties # tensor (N, 3, 300, 300), 3 lists of N tensors each

