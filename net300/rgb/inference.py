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
from mobilenet_ssd_priors2 import priors
import torch.nn.functional as F
from utils import detect_objects2
# from mobilev2ssd import SSD
from rgb_model import SSD_RGB
import time
from imutils.video import FPS

# from rgb_model import SSD_RGB
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
priors_cxcy = priors
priors_cxcy = priors_cxcy.to(device)
n_classes = 2

sanitest_rgb = '/home/fereshteh/kaist/sanitest'
filename='set10_V000_I03899.png'

fps=FPS().start()

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
    fps.update()
    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = detect_objects2(model, priors_cxcy, predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k, n_classes=n_classes)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')
    det_scores = det_scores [0].to('cpu').tolist()
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
        # det_scores[i]=det_scores[i].tolist()
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
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0]/2 +8.,
                                box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=str(round(det_scores[i],2)), fill='white',
                    font=font)
        # print(det_scores[i])
    del draw
#det_labels[i].upper()
    
    return annotated_image
# kaist_path1="/home/fereshteh/kaist"

id=3500
# fps=FPS().start()
# img_path = os.path.join(kaist_path1, 'sanitized', 'I' + '{0:05}'.format(id) + '.png')
img_path = os.path.join(sanitest_rgb, filename)
# image_rgb = Image.open(image_rgb_path, mode='r')
# img_path = '/home/fereshteh/codergb/I01254.png' #I01254.png I03054.png I00421.png I00589.png I00720.png I00980.png I01149.png
checkpoint = torch.load('fine_checkpoint_ssd300_7.pth.tar')#I01150.png I01168.png I01410.png I01465.png I01775.png
print(checkpoint['loss'])#I02165.png I02214.png I02658.png
def main(img_path,checkpoint):
	
    #img_path = args.img_path
    
    #img_path = '/home/fereshteh/kaist/set00_V000/images/set00/V000/visible/I00000.jpg'
    original_image = Image.open(img_path, mode='r')

    original_image = original_image.convert('RGB')
    # Load model checkpoint
    #checkpoint = args.checkpoint
    #checkpoint = torch.load('BEST_checkpoint_ssd300.pth.tar')
    #checkpoint = torch.load(checkpoint, map_location='cpu')
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['best_loss']
    print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, best_loss))
    # model = SSD(num_classes=n_classes, backbone_network="MobileNetV1")
    model = SSD_RGB(num_classes=2, backbone_network="MobileNetV1")
	# global model
    model.load_state_dict(checkpoint['model'])#model parameter
    model = model.to(device)
    model.eval()

    detect(model, original_image, min_score=0.25, max_overlap=0.5, top_k=200, suppress=['background','nonperson']).show()

if __name__ == '__main__':
    main(img_path,checkpoint)
  #detect(model, original_image, min_score=0.2, max_overlap=0.5, top_k=200).show()
  #  parser.add_argument('img_path',help='Image path')
   # parser.add_argument('checkpoint',help='Path for pretrained model')
    #args = parser.parse_args()
    
    #main(args)
    	
