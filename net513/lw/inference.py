#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 15:45:16 2019

@author: viswanatha
"""
from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
import torch
import argparse
from mobilenet_ssd_priors1 import priors
import torch.nn.functional as F
from utils import detect_objects
# from mobilev2ssd import SSD
from lw_model import SSD_LW
import cv2

# from rgb_model import SSD_RGB
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
priors_cxcy = priors
priors_cxcy = priors_cxcy.to(device)
n_classes = 2

sanitest_rgb = '/home/fereshteh/kaist/sanitest_lw'
filename = 'set11_V000_I01539.png'


def detect(model, original_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.
    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transforms
    resize = transforms.Resize((300, 300))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    
    image = normalize(to_tensor(resize(original_image)))
    #image = (to_tensor(resize(original_image)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = detect_objects(model, priors_cxcy, predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k, n_classes=n_classes)

    print(det_scores)
    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['nonperson']:
        # Just return original image
        return original_image

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    #font = ImageFont.truetype("./calibril.ttf", 15)
    #font = ImageFont.truetype("arial.ttf", 15)
    font = ImageFont.load_default()

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

# Boxes
        #for j in range(0,i):
            #box_location = det_boxes[i].tolist()
            #box_locationj = det_boxes[j].tolist()
            #w_intsec = np.maximum (0, (np.minimum(box_location[2], box_locationj[2]) - np.maximum(box_location[0], box_locationj[0])))
            #h_intsec = np.maximum (0, (np.minimum(box_location[3], box_locationj[3]) - np.maximum(box_location[1], box_locationj[1])))
            #w1=box_location[2]- box_location[0]
            #h1=box_location[3]- box_location[1]
            #w2=box_location[2]- box_location[0]
            #h2=box_location[3]- box_location[1]
            #s_intsec = w_intsec * h_intsec
            #st=w2*h2-s_intsec
            #iou=s_intsec/st
            #print(iou)
            #if iou<0.7:
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
                    det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                                box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                    font=font)
        # print(det_scores[i])
    del draw

    
    return annotated_image
# sanitest_rgb = '/home/fereshteh/kaist/sanitest_lw'
# filename='set06_V001_I00439.png'
id=3500
# kaist_path="/home/fereshteh/kaist"
# img_path = os.path.join(kaist_path, 'sanitized_lw', 'I' + '{0:05}'.format(id) + '.png')
img_path = os.path.join(sanitest_rgb, filename)
# img_path = '/home/fereshteh/codergb/I01254.png' #I01254.png I03054.png I00421.png I00589.png I00720.png I00980.png I01149.png
checkpoint = torch.load('BEST_fine_checkpoint_ssd300_lw_3.pth.tar')#I01150.png I01168.png I01410.png I01465.png I01775.png


# A=((checkpoint['optimizer']))#I02165.png I02214.png I02658.png
print(checkpoint['epoch'])#I02165.png I02214.png I02658.png
# adjust_learning_rate(A, 1)
def main(img_path,checkpoint):
	
    #img_path = args.img_path
    
    #img_path = '/home/fereshteh/kaist/set00_V000/images/set00/V000/visible/I00000.jpg'
    original_image = Image.open(img_path, mode='r')
    # original_image = cv2.imread(img_path, 1)
    # # image = original_image.convert('RGB')
    # # image = np.zeros_like(image1)
    # # image[:, :, 0] = image1
    # # image[:, :, 1] = image1
    # # # image[:, :, 2] = image1
    # # image = cv2.imread(img_path, 1)
    # # image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # # image = np.zeros_like(image)
    # # image[:, :, 0] = image1
    # # image[:, :, 1] = image1
    # # image[:, :, 2] =image1
    # # image = ~image
    # # original_image = Image.fromarray(original_image)
    # lab = cv2.cvtColor(original_image, cv2.COLOR_BGR2LAB)
    # l, a, b = cv2.split(lab)
    # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    # cl = clahe.apply(l)
    # limg = cv2.merge((cl, a, b))
    # original_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    # color_coverted = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    # original_image = Image.fromarray(color_coverted)
    original_image = original_image.convert('RGB')
    # Load model checkpoint
    #checkpoint = args.checkpoint
    #checkpoint = torch.load('BEST_checkpoint_ssd300.pth.tar')
    #checkpoint = torch.load(checkpoint, map_location='cpu')
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['best_loss']
    print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, best_loss))
    # model = SSD(num_classes=n_classes, backbone_network="MobileNetV1")
    model = SSD_LW(num_classes=2, backbone_network="MobileNetV1")
	# global model
    model.load_state_dict(checkpoint['model'])#model parameter
    model = model.to(device)
    model.eval()
    biases = list()
    not_biases = list()
    param_names_biases = list()
    param_names_not_biases = list()
    lr = 1e-5  # learning rate
    momentum = 0.9  # momentum
    weight_decay = 1e-4  # weight decay
    grad_clip = None  # clip if g
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
    adjust_learning_rate(optimizer, 1)

    detect(model, original_image, min_score=0.1, max_overlap=0.5, top_k=200, suppress=['nonperson']).show()
    
if __name__ == '__main__':
    main(img_path,checkpoint)
  #detect(model, original_image, min_score=0.2, max_overlap=0.5, top_k=200).show()
  #  parser.add_argument('img_path',help='Image path')
   # parser.add_argument('checkpoint',help='Path for pretrained model')
    #args = parser.parse_args()
    
    #main(args)
    	
