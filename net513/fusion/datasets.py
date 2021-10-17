
import torch
from torch.utils.data import Dataset
import json
import cv2
import os
# from PIL import Image
from utils import *
from PIL import Image, ImageEnhance

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

        assert self.split in {'NSANITEST','NNSANITEST','SANITEST1','DDSANITEST','SANITEST','SANITRAIN','SANITRAIN1','SANITRAIN2','SANIVAL',"VIS_SANITEST"}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Read data files
        with open(os.path.join(data_folder, self.split + '_images_rgb.json'), 'r') as j:
            self.images_rgb = json.load(j)
        with open(os.path.join(data_folder, self.split + '_images_lw.json'), 'r') as j:
            self.images_lw = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)
        # with open(os.path.join(data_folder, 'SANITEST' + '_images.json'), 'r') as j:
        #     self.images1 = json.load(j)
        # with open(os.path.join(data_folder, 'SANITEST' + '_objects.json'), 'r') as j:
        #     self.objects1 = json.load(j)
        # print(len(self.images_rgb) , len(self.images_lw), len(self.objects))
        assert len(self.images_rgb) == len(self.images_lw) == len(self.objects)

    def __getitem__(self, i):
        # Read image
        # print(i,"hi")
        image_rgb = Image.open(self.images_rgb[i], mode='r')
        # rgb = cv2.imread(self.images_rgb[i])
        rgb=image_rgb
        # rgb = cv2.resize(rgb, (800, 800))
        # print(rgb.shape)
        image_rgb = image_rgb.convert('RGB')
        enhancer = ImageEnhance.Brightness(image_rgb)
        factor = 1.2  # brightens the image
        image_rgb = enhancer.enhance(factor)
        # image_lw = Image.open(self.images_lw[i], mode='r')
        # image_lw = image_lw.convert('RGB')
        image_lw = cv2.imread(self.images_lw[i], 1)
        lab = cv2.cvtColor(image_lw, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        original_image_lw = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        color_coverted = cv2.cvtColor(original_image_lw, cv2.COLOR_BGR2RGB)
        original_image_lw = Image.fromarray(color_coverted)
        original_image_lw = original_image_lw.convert('RGB')
        image_lw = original_image_lw.convert('RGB')

        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        # labels=[int(x) for x in objects['labels']]
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

        # Apply transformations
        # if (boxes.__len__() != 0):
        #     boxes= torch.FloatTensor([0, 0, 1., 1.])
        #     labels= torch.LongTensor(2)
        #     difficulties=torch.ByteTensor(0)
        # rgb=image_rgb
        # m=0
        # if (i < 5098):
        #     m=1
        m=0

        if self.split in {'SANITRAIN','SANITRAIN1','SANITRAIN2'}:
            if i<7601:
                rgb1, image_rgb, image_lw, boxes, labels, difficulties = transform1(m, image_rgb, image_lw, boxes, labels,
                                                                                   difficulties, split=self.split)
                return rgb1, image_rgb, image_lw, boxes, labels, difficulties
            if 7600<i<15204:
                rgb1, image_rgb, image_lw, boxes, labels, difficulties = transform(m, image_rgb, image_lw, boxes, labels,
                                                                                   difficulties, split=self.split)
                return rgb1, image_rgb, image_lw, boxes, labels, difficulties
            if 15203<i<22804:
                rgb1, image_rgb, image_lw, boxes, labels, difficulties = transform3(m, image_rgb, image_lw, boxes, labels,
                                                                                   difficulties, split=self.split)
                return rgb1, image_rgb, image_lw, boxes, labels, difficulties
            if 22803<i<30404:
                rgb1, image_rgb, image_lw, boxes, labels, difficulties = transform4(m, image_rgb, image_lw, boxes, labels,
                                                                                   difficulties, split=self.split)
                return rgb1, image_rgb, image_lw, boxes, labels, difficulties


        else:
            rgb1,image_rgb, image_lw, boxes, labels, difficulties = transform(m,image_rgb,image_lw, boxes, labels, difficulties, split=self.split)

        return rgb1,image_rgb,image_lw, boxes, labels, difficulties

    def __len__(self):
        return len(self.images_rgb)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images_rgb = list()
        im=list()
        images_lw= list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:

            im.append(b[0])
            images_rgb.append(b[1])
            images_lw.append(b[2])
            boxes.append(b[3])

            labels.append(b[4])
            difficulties.append(b[5])
        images_rgb = torch.stack(images_rgb, dim=0)
        images_lw = torch.stack(images_lw, dim=0)
        # im=torch.stack(im, dim=0)
        return im,images_rgb,images_lw, boxes, labels, difficulties # tensor (N, 3, 300, 300), 3 lists of N tensors each

