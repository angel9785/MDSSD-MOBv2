import xml.etree.ElementTree as ET
import os
from PIL import Image
from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = None
import re
import torch
import os
n=0
id=0
voc_labels = ('nonperson','person')

label_map = {k: v for v, k in enumerate(voc_labels)}
# label_map['background'] = 0
# label_map['person'] = 1
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

# for id in range(0,17907):
# for filename in sorted(os.listdir('/home/fereshteh/kaist/sanitized_annotations')):
filename='set00_V000_I01225.txt'
if (filename=='set00_V000_I01225.txt'):
    path="/home/fereshteh/kaist"
    # address = os.path.join(path, 'panno_sani', 'I'+  '{0:05}'.format(id) + '.txt')
    # imagepath=os.path.join(path, 'sanitized', 'I'+ '{0:05}'.format(id)+ '.png')
    imagepath=os.path.join(path, 'test_sani', 'I'+ '{0:05}'.format(0)+ '.png')
    M = []
    boxes = list()  # torch.FloatTensor([0, 0, 0, 0])#.unsqueeze(0)
    labels = list()
    difficulties = list()
    images = list() 
    # filename='/home/fereshteh/kaist/sanitized_annotations/'+str(filename)
    filename='/home/fereshteh/kaist/test_sani/'+str(filename)
    with open(filename) as f:
    # with open(os.path.join(sani, filename)) as f:
        n=0
        d=0
        for line in f.readlines():
            for word in line.split():
                person = re.findall('person', word)

                if person:
                    # print(line)
                    i = 0
                    
                    for word in line.split():
                        if(d==0):

                            if (i == 0):
                                label = word.lower().strip()
                                
                                # print(label)
                            if (i == 1):
                                xmin = int(word) - 1
                                if (int(xmin)<150):
                                    xnew=xmin
                                    left=0
                                else:
                                    xnew=150
                                    left=int(xmin)-150
                                # print(x)
                            if (i == 2):
                                ymin = int(word) - 1
                                if (int(ymin)<150):
                                    ynew=ymin
                                    top=0
                                else:
                                    ynew=150
                                    top=int(ymin)-150
                                # print(y)
                            if (i == 3):
                                w = int(word) if int(word) > 0 else 1
                                xmax = xmin + w
                                # print(w)
                            if (i == 4):
                                h = int(word) if int(word) > 0 else 1
                                ymax = ymin + h
                            if (i == 5):
                                difficult = int(word)
                                d=1
                            i=i+1
                            
                        
                        if(d==1):
                            right=int(xmax)+150
                            bottom=int(ymax)+150
                            image = Image.open(imagepath)
                            image=image.crop((left,top,right,bottom))
                            # image.save('/home/fereshteh/kaist/person_sani/'+ 'I'+ '{0:05}'.format(n)+ '.png')
                            image.save('/home/fereshteh/kaist/test_sani/'+ 'I'+ '{0:05}'.format(n)+ '.png')
                            # labels.append(label)
                            labels.append(label_map[label])
                            boxes.append([xmin, ymin, xmax, ymax])
                            images.append([left,top,right,bottom])
                            difficulties.append(difficult)
                            n=n+1
                            d=0
        boxes = torch.FloatTensor(boxes)  # (n_objects, 4)
        images = torch.FloatTensor(images) 
        labels = torch.LongTensor(labels)  # (n_objects)
        difficulties = torch.ByteTensor(difficulties)                
        m=0
        for m in range(0,n):
            image = images[m]
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.
            centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (
                    bb_centers[:, 1] < bottom)
            if not centers_in_crop.any():
                continue
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]
            new_difficulties = difficulties[centers_in_crop]

            # Calculate bounding boxes' new coordinates in the crop
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], image[:2])  # crop[:2] is [left, top]
            new_boxes[:, :2] -= image[:2]
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], image[2:])  # crop[2:] is [right, bottom]
            new_boxes[:, 2:] -= image[:2]
            # f = open("/home/fereshteh/kaist/panno_sani/"+ 'I'+ '{0:05}'.format(m)+ '.txt', "w")
            f = open("/home/fereshteh/kaist/test_sani/"+ 'I'+ '{0:05}'.format(m)+ '.txt', "w")
            j=0
            o=len(new_labels)
            for word in line.split():
                if(j<o):

                    i=0
                    for word in line.split():
                        if (i == 0):
                            label = new_labels[j]
                            # label=rev_label_map(label)
                            f.write(str(label))   
                        if (i == 1):
                            xmin = new_boxes[j][0]
                            f.write(str(xmin))
                        if (i == 2):
                            ymin = new_boxes[j][1]
                            f.write(str(ymin))
                        if (i == 3):
                            w = new_boxes[j][2]
                            f.write(str(w))
                        if (i == 4):
                            h = str((new_boxes[j][2]))
                            f.write(h)
                        if (i == 5):
                            difficult = (new_difficulties[j])
                            f.write(str(difficult))
                    j=j+1

