# USAGE
# python deep_learning_object_detection.py --image images/example_01.jpg \
#	--prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
import argparse
import cv2
import os
import sys
sys.path.append("/home/fereshteh/caffe-ssd/python")
import caffe
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

caffe.set_mode_gpu()
caffe.set_device(0)
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = caffe.Net(args["prototxt"], args["model"], caffe.TEST) #cv2.dnn.readNetFromCaffe()

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD
# implementation)

im = np.array(Image.open(args["image"]), dtype=np.uint8)

# Create figure and axes
fig,ax = plt.subplots(1)

# Display the image
ax.imshow(im)
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
image = cv2.resize(image, (300, 300))
image = image -127.5
image = image * 0.007843
image = image.astype(np.float32)
image = image.transpose((2,0,1))

net.blobs['data'].data[...] = image
t1=time.time()
detections = net.forward()
t2=time.time()
box = detections['detection_out'][0, 0, :, 3:7]
cls = detections['detection_out'][0, 0, :, 1]
conf = detections['detection_out'][0, 0, :, 2]
t=t2-t1
print(t)
for i in range(len(box)):
	bb = box[i] * np.array([w, h, w, h])
	(startX, startY, endX, endY) = bb.astype("int")
	
	# display the prediction
	label = "{}: {:.2f}%".format(CLASSES[int(cls[i])], conf[i] * 100)
	print("[INFO] {}".format(label))
	rect = patches.Rectangle((startX,startY),endX-startX,endY-startY,linewidth=1,edgecolor='r',facecolor='none')
	ax.add_patch(rect)
	y = startY - 15 if startY - 15 > 15 else startY + 15
	ax.text(startX, y,label,bbox=dict(facecolor='red',alpha=0.5))

	
# show the output imageax.


plt.show()
cv2.waitKey(0)

