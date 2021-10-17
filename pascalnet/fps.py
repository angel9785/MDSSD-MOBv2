#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 15:45:16 2019

@author: viswanatha
"""
from utils import *
from datasets import PascalVOCDataset
from tqdm import tqdm
from pprint import PrettyPrinter
from mobilenet_ssd_priors_512 import priors
import warnings
# from rgb_model1 import SSD_RGB
from dec512 import SSD_RGB
import time
warnings.filterwarnings("ignore", category=UserWarning)
# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()
priors_cxcy = priors
priors_cxcy = priors_cxcy.to(device)
# Parameters
config_file_path = "/home/fereshteh/code/SSD_MobileNet-master/config.json"

with open(config_file_path, "r") as fp:
    config = json.load(fp)
voc07_path = config['voc07_path']

    #voc12_path = 'VOCdevkit/VOC2012'
voc12_path = config['voc12_path']
# create_data_lists(voc07_path, voc12_path, output_folder=config['data_folder'])
data_folder = 'dataset'
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 1
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = '/home/fereshteh/code/SSD_MobileNet-master/513pascalvoc_dec_checkpoint_ssd_new10.pth.tar'

voc_labels = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()} # Inverse mapping
n_classes = len(label_map)  # number of different types of objects
backbone_network='MobileNetV2'
# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint)
model = SSD_RGB(num_classes=n_classes,backbone_network=backbone_network)
#model = checkpoint['model']
model.load_state_dict(checkpoint['model'])


# model.load_state_dict(checkpoint['model'])#model parameter
model = model.to(device)
model.eval()

# Switch to eval mode
model.eval()

# from imutils.video import FPS
# import glob



# fps=FPS().start()
test_dataset = PascalVOCDataset(data_folder,
                                split='test',
                                keep_difficult=keep_difficult)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)
global a
global t
def speed(test_loader, model):
    a = 0
    t = 0
    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores

    with torch.no_grad():
        # Batches

        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)  # (N, 3, 300, 300)

            start_time=time.time()
            predicted_locs, predicted_scores = model(images)
            stop_time=time.time()
            a=a+(stop_time-start_time)
            t=t+1
        print(t,a)
        print("[INFO] approx. FPS: {:.2f}".format(t/ a))


if __name__ == '__main__':
    speed(test_loader, model)

    # fps.stop()
    # print("[INFO] approx. FPS: {:.2f}".format(a/t))
  
