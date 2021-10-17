#!/usr/bin/env python

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
import matplotlib 
#matplotlib.use('Agg')

import pickle
import _init_paths
from fast_rcnn.config import cfg
# from fast_rcnn.test import im_detect
from fast_rcnn.test import im_detect_2in_
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import sys
sys.path.append("/home/fereshteh/caffe-faster-rcnn1/python")
import caffe, os, sys, cv2
import argparse
from copy import copy, deepcopy
from datasets.voc_eval import voc_eval
import xml.dom.minidom as minidom
from imutils.video import FPS


# def demo_detection(net, data_dir, image_name, CLASSES, gt_roidb):
def demo_detection(u,net, data_dir,data_dir1, image_name, CLASSES):
    """Detect object classes in an image using pre-computed object proposals."""
#    print cfg.TEST.SCALES
    # Load the demo image
    # im1_file = os.path.join(data_dir,'color', image_name + '.jpg')
    # im2_file = os.path.join(data_dir,'thermal', image_name + '.jpg')
    im1_file = os.path.join(data_dir, image_name)
    im2_file = os.path.join(data_dir1, image_name )
    # print (im1_file)
    # print (im2_file)
    im1 = cv2.imread(im1_file)
    im2 = cv2.imread(im2_file)
    print(im1.shape)
    # Detect all object classes and regress object bounds
    # timer = Timer()
    # timer.tic()
    scores, boxes = im_detect_2in(net, im1, im2)
    # timer.toc()

    # print (f'Detection took {timer.total_time}s' )

    # Visualize detections for each class
    #CONF_THRESH = 0.1
    NMS_THRESH = 0.5
#    NMS_THRESH = [0.2,0.2, 0.2]
    all_dets = None
    #for cls_ind, cls in enumerate(classes):
    #    cls_ind += 1 # because we skipped background
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        cls_inds = np.ones((len(keep),)) * cls_ind
        #cls_inds = np.ones((len(cls_scores),)) * cls_ind
        #cls_inds = np.ones((len(cls_scores),)) * (-1)
        #cls_inds[keep] = cls_ind
        dets = np.hstack((cls_inds[:,np.newaxis], dets))
        if all_dets is None:
            all_dets = dets
        else:
            all_dets = np.vstack((all_dets, dets))

    visual_detection_results(u,im1, boxes, scores, CLASSES, threds=0)
    # return all_dets

    # nms again
    #y = deepcopy(all_dets[:,1:6]).astype(np.float32)
    #keep = nms(y, 0.4)
    #all_dets = all_dets[keep,:]

def visual_detection_results(u,im, boxes, scores, CLASSES, threds = 0):
    NMS_THRESH = 0.5
    im = im[:,:,(2,1,0)]



    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1
        if str(cls) != 'person':
            continue
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        cls_inds = np.ones((len(keep),)) * cls_ind

        inds = np.where(dets[:,-1])[0]
        # if len(inds) == 0:
        #     continue
        f = open("/home/fereshteh/faster/" + '{0:05}'.format(u) + '.txt', "w")
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]





                # color

            f.write("person")
            f.write(' ')

            xmin = int(bbox[0])
            f.write(str(xmin))
            f.write(' ')
            ymin = int(bbox[1])
            f.write(str(ymin))
            f.write(' ')
            w = int(bbox[2]) - int(bbox[0])
            f.write(str(w))
            f.write(' ')
            h = int(bbox[3]) - int(bbox[1])
            f.write(str(h))
            f.write(' ')
            difficult = str(score)
            f.write(str(difficult))
            f.write("\n")

def save_detection_results(det_file, classes, dets):
    with open(det_file, 'w') as fid:
        if dets is None:
            return

        for det in dets:
           class_id = int(det[0])
           if class_id < 0:
               label = 'unknown'
           else:
               label = classes[class_id]
               if label.find('car') >= 0:
                   label = 'car'
           #fmt = '%s -1 -1 -10 %9.4f %9.4f %9.4f %9.4f -1 -1 -1 -1 -1 -1 -1 %9.4f\n'
           fmt = '%s %9.4f %9.4f %9.4f %9.4f %9.8f\n'
           fid.write(fmt % (label, det[1], det[2], det[3], det[4], det[5],))

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--train_scale', dest='train_scale', help='which train scale?', default=600,type=int)
    parser.add_argument('--test_scale', dest='test_scale', help='which test scale?', default=1000,type=int)
    parser.add_argument('--proposals', dest='num_proposals', help='how many proposals?', default=300,type=int)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()
    # print (args)
    model_dir = os.path.join(cfg.ROOT_DIR, 'models/kaist_fusion/VGG16')
    prototxt = os.path.join(cfg.ROOT_DIR, 'models/kaist_fusion/VGG16', 'faster_rcnn_test1.pt')
    caffemodel = os.path.join(model_dir, 'VGG16_faster_rcnn_final_kaist_fusion.caffemodel')

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    

    CLASSES = ('__background__','person')
 
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    # print ('\n\nLoaded network {:s}'.format(caffemodel))
    # print (prototxt)
    # print det_classes

    #train_scale = model_instance.split('_')[-1]
#    train_scale = 600
    cfg.TEST.SCALES = (args.train_scale,)
    cfg.TEST.RPN_POST_NMS_TOP_N = args.num_proposals
    cfg.TEST.MAX_SIZE = args.test_scale


    # im_names = ['set07_V002_I01379', 'set08_V001_I02559', 
    #             'set09_V000_I00939', 'set10_V001_I01159']
    # data_dir = os.path.join(cfg.ROOT_DIR, 'data/demo_pedestrian')
    data_dir = '/home/fereshteh/kaist/sanitest'
    data_dir1 = '/home/fereshteh/kaist/sanitest_lw'
    # for im_name in im_names:
    u=0
    for im_name in sorted(os.listdir('/home/fereshteh/kaist/sanitest')):
#     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        # print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        # print ('Demo for data/demo/{}'.format(im_name))
        # gt = load_kitti_annotation(data_dir, im_name)
        # det = demo_detection(net, data_dir, im_name, CLASSES, gt)
        det = demo_detection(u,net, data_dir,data_dir1 ,im_name, CLASSES)
        u=u+1

    # plt.show()

