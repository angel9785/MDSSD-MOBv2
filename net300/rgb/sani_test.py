import xml.etree.ElementTree as ET
import os
from PIL import Image
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None
import re
import torch
import os
import cv2
import numpy as np
from numpy import random
n = 0
id = 0
voc_labels = ('nonperson', 'person')

label_map = {k: v for v, k in enumerate(voc_labels)}
# label_map['background'] = 0
# label_map['person'] = 1
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping
id=0
n=0
a=0
g=0
sanitrain_rgb = '/home/fereshteh/kaist/sanitized'
sanitrain_lw = '/home/fereshteh/kaist/sanitized_lw'
# for id in range(0,17907):
o=0
for filename in sorted(os.listdir('/home/fereshteh/kaist/sanitized_lw')):
# filename = 'set00_V000_I01225.txt'
# filename ='I00328.txt'
# if (filename == filename):
    if (o>6097):

        g = g + 1
        print(g)
        # if (g < 6097):
        path = "/home/fereshteh/kaist"
        # address = os.path.join(path, 'panno_sani', 'I'+  '{0:05}'.format(id) + '.txt')
        # imagepath=os.path.join(path, 'sanitized', 'I'+ '{0:05}'.format(id)+ '.png')
        filename2 = re.sub('.png', ".txt", str(filename))
        imagepath1 =os.path.join(sanitrain_rgb, filename)
        imagepath2 = os.path.join(sanitrain_lw, filename)
        # imagepath = os.path.join(path, 'sanitized', 'I' + '{0:05}'.format(id) + '.png')
        M = []
        boxes = list()  # torch.FloatTensor([0, 0, 0, 0])#.unsqueeze(0)
        labels = list()
        difficulties = list()
        images = list()

        filename3='/home/fereshteh/kaist/sanitized_txt/'+str(filename2)
        # filename = '/home/fereshteh/kaist/sanitized_txt/' + str(filename)
        # l=0
        with open(filename3) as f:

            # with open(os.path.join(sani, filename)) as f:
            n = 0
            d = 0
            k=0
            s=0
            for l in f.readlines():
                s=s+1
            if (s<2):
                # image = Image.open(imagepath)
                # image.save('/home/fereshteh/kaist/person_sani/' + 'I' + '{0:05}'.format(a) + '.png')
                # f = open("/home/fereshteh/kaist/panno_sani/" + 'I' + '{0:05}'.format(a) + '.txt', "w")
                # f.write("blabla")
                # a=a+1
                # print ("hi")
                continue
                # print ("hi")
            if s==3:
                print("")
            # for r in range (2,s+1):
            boxes = list()  # torch.FloatTensor([0, 0, 0, 0])#.unsqueeze(0)
            labels = list()
            difficulties = list()
            # images = list()
            k = 0
            with open(filename3) as f:
                for line in f.readlines():
                    k=k+1

                    for word in line.split():
                        person = re.findall('person', word)
                        person3 = re.findall('cyclist', word)
                        person1 = re.findall('person?', word)
                        person2 = re.findall('person?a', word)
                        people=re.findall('people', word)

                        if (person or person1 or person2 or people or person3):
                            # print(line)
                            i = 0

                            for word in line.split():

                                if (d == 0):
                                    if(i>5):
                                        i=i+1
                                        continue
                                    if (i<6):


                                        if (i == 0):
                                            label = word.lower().strip()
                                            if label=="person?a":
                                                label="person"
                                            if label=="cyclist":
                                                label="person"
                                            if label=="person?":
                                                label="person"
                                            if label == "people":
                                                label = "person"

                                        # if (k==r):
                                            # print(label)
                                        if (i == 1):
                                            xmin = int(word)
                                            # left = xmin-10
                                            if (int(xmin) < 150):
                                                # xmin = xmin1
                                                left = 0

                                            else:

                                                left = int(xmin) - 150
                                                # xmin = 150

                                            # print(x)
                                        if (i == 2):
                                            ymin = int(word)
                                            if (int(ymin) < 150):
                                                # ymin = ymin1
                                                top = 0
                                            else:

                                                top = int(ymin)-150
                                                # ymin = 150

                                            # print(y)
                                        if (i == 3):
                                            w = int(word) if int(word) > 0 else 1
                                            xmax = xmin + w
                                            right=xmax+150
                                            if (right>640):
                                                right = 640
                                            # print(w)
                                        if (i == 4):
                                            h = int(word) if int(word) > 0 else 1
                                            ymax = ymin + h
                                            bottom=ymax+150
                                            if (bottom>512):
                                                bottom =512
                                        if (i == 5):
                                            difficult = int(word)
                                            d = 1
                                        i = i + 1
                                        # if (k != r):
                                        #     # print(label)
                                        #     if (i == 1):
                                        #         xmin = int(word)
                                        #
                                        #
                                        #         # print(x)
                                        #     if (i == 2):
                                        #         ymin = int(word)
                                        #
                                        #
                                        #         # print(y)
                                        #     if (i == 3):
                                        #         w = int(word) if int(word) > 0 else 1
                                        #         xmax = xmin + w
                                        #
                                        #         # print(w)
                                        #     if (i == 4):
                                        #         h = int(word) if int(word) > 0 else 1
                                        #         ymax = ymin + h
                                        #
                                        #     if (i == 5):
                                        #         difficult = int(word)
                                        #         d = 1
                                        #     i = i + 1
                                if (d == 1):
                                    # if(k==r):
                                    # right = int(xmax) + 150
                                    # image3=list()
                                    image = Image.open(imagepath1)
                                    imagel = Image.open(imagepath2)
                                    # image1 = image.crop((left, top, right, bottom))
                                    image1 = image.crop((-300, -300, image.width+300, image.height+300))
                                    image2 = imagel.crop((-300, -300, imagel.width + 300, imagel.height + 300))
                                    # image1=np.ascontiguousarray(image1)
                                    # image1 = np.array(image1)
                                    # image2 = Image.open("/home/fereshteh/network2/h.jpg")
                                    # image3[0] = image2
                                    # image3[1] = image2
                                    # image3[2] = image2
                                    # image2 =image.crop((480-40,512-40, 480, 512))
                                    # image2[0] = h
                                    # image2[1] = h
                                    # image2[2] = h
                                    # image2 = Image.fromarray(np.uint8(image2)).convert('RGB')
                                    # image2.show()
                                    # image2 =image2.crop((0, 0, 100, 100))
                                    # image.save('/home/fereshteh/kaist/person_sani/'+ 'I'+ '{0:05}'.format(a)+ '.png')
                                    images=torch.FloatTensor([left, top, right, bottom])
                                # image.save('/home/fereshteh/kaist/test_sani/' + 'I' + '{0:05}'.format(n) + '.png')
                                # labels.append(label)
                                    print(label)
                                    labels.append(label_map[label])
                                    boxes.append([xmin, ymin, xmax, ymax])
                                    # boxes.append([0, 0,w, h])
                                    difficulties.append(difficult)
                                    # if (k != r):
                                    #     # right = int(xmax) + 150
                                    #     # image3=list()
                                    #     # image = Image.open(imagepath)
                                    #     # image1 = image.crop((left, top, right, bottom))
                                    #     # image2 = Image.open("/home/fereshteh/network2/h.jpg")
                                    #     # image3[0] = image2
                                    #     # image3[1] = image2
                                    #     # image3[2] = image2
                                    #     # image2 =image.crop((480-40,512-40, 480, 512))
                                    #     # image2[0] = h
                                    #     # image2[1] = h
                                    #     # image2[2] = h
                                    #     # image2 = Image.fromarray(np.uint8(image2)).convert('RGB')
                                    #     # image2.show()
                                    #     # image2 =image2.crop((0, 0, 100, 100))
                                    #     # image.save('/home/fereshteh/kaist/person_sani/'+ 'I'+ '{0:05}'.format(a)+ '.png')
                                    #     # images = torch.FloatTensor([left, top, right, bottom])
                                    #     # image.save('/home/fereshteh/kaist/test_sani/' + 'I' + '{0:05}'.format(n) + '.png')
                                    #     # labels.append(label)
                                    #     print(label)
                                    #     labels.append(label_map[label])
                                    #     boxes.append([xmin, ymin, xmax, ymax])
                                    #     # boxes.append([0, 0,w, h])
                                    #     difficulties.append(difficult)

                                    d = 0
                        else:


                            break
                # boxes = torch.FloatTensor(boxes)  # (n_objects, 4)
                if (labels.__len__() == 0):
                    # image = Image.open(imagepath)
                    # image.save('/home/fereshteh/kaist/person_sani/' + 'I' + '{0:05}'.format(a) + '.png')
                    # f = open("/home/fereshteh/kaist/panno_sani/" + 'I' + '{0:05}'.format(a) + '.txt', "w")
                    # f.write("blabla")
                    # a = a + 1
                    continue
                boxes = torch.FloatTensor(boxes)
                labels = torch.LongTensor(labels)  # (n_objects)
                difficulties = torch.ByteTensor(difficulties)
                # m = 0
                # for m in range(0, n):
                # images = images
                # if (boxes.size()==1):
                #     print("")
                bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.
                centers_in_crop = (bb_centers[:, 0] >= left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (
                        bb_centers[:, 1] < bottom)
                if not centers_in_crop.any():
                    continue
                new_boxes = boxes[centers_in_crop, :]
                new_labels = labels[centers_in_crop]
                new_difficulties = difficulties[centers_in_crop]

                # Calculate bounding boxes' new coordinates in the crop
                # new_boxes[:, :2] = torch.max(new_boxes[:, :2], images[:2])  # crop[:2] is [left, top]
                # new_boxes[:, :2] -= images[:2]
                # new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], images[2:])  # crop[2:] is [right, bottom]
                # new_boxes[:, 2:] -= images[:2]
                new_boxes[:, :] += torch.tensor([300,300,300,300])
                # image1.save('/home/fereshteh/kaist/person_sani/' + 'I' + '{0:05}'.format(a) + '.png')
                # for i in range(0, len(new_boxes)):
                #     if ((image1.width - new_boxes[i, 0]) < 0):
                #         print("")
                #     if ((image1.width - new_boxes[i, 2]) < 0):
                #         print("")
                # boxes[:, :2] = torch.max(boxes[:, :2], images[:2])  # crop[:2] is [left, top]
                # boxes[:, :2] -= images[:2]
                # boxes[:, 2:] = torch.min(boxes[:, 2:], images[2:])  # crop[2:] is [right, bottom]
                # boxes[:, 2:] -= images[:2]
                # for i in range(new_boxes.size(0)):
                #     #
                #     box_location = new_boxes[i].tolist()
                #     image1 = cv2.rectangle(image1, (int(box_location[0]), int(box_location[1])),
                #                           (int(box_location[2]), int(box_location[3])), (0, 255, 0), 2)
                image1.save('/home/fereshteh/kaist/person_sani/' + 'I' + '{0:05}'.format(a) + '.png')
                image2.save('/home/fereshteh/kaist/person_sani_lw/' + 'I' + '{0:05}'.format(a) + '.png')
                # cv2.imwrite('/home/fereshteh/kaist/person_sani/'  + 'I' + '{0:05}'.format(a) + '.png', image1)
                # image2.save('/home/fereshteh/kaist/person_sani/' + 'I' + '{0:05}'.format(a+1) + '.png')
                # for i in range(0, len(boxes)):
                #     if ((image1.width - boxes[i, 0]) < 0):
                #         print("")
                #     if ((image1.width - boxes[i, 2]) < 0):
                #         print("")
                f = open("/home/fereshteh/kaist/panno_sani/"+ 'I'+ '{0:05}'.format(a)+ '.txt', "w")

                # print(a)
                # f = open("/home/fereshteh/kaist/test_sani/" + 'I' + '{0:05}'.format(m) + '.txt', "w")
                j = 0
                # o = len(labels)
                # for j in range(0,len(new_labels):
                #
                #
                #     label =int(new_labels[j])
                #     label=rev_label_map[label]
                #     f.write(str(label))
                #     f.write(' ')
                #
                #     xmin = int(new_boxes[j][0])
                #     f.write(str(xmin))
                #     f.write(' ')
                #     ymin = int(new_boxes[j][1])
                #     f.write(str(ymin))
                #     f.write(' ')
                #     w = int(new_boxes[j][2])-int(new_boxes[j][0])
                #     f.write(str(w))
                #     f.write(' ')
                #     h = int((new_boxes[j][3]))-int(new_boxes[j][1])
                #     f.write(str(h))
                #     f.write(' ')
                #     difficult = int(new_difficulties[j])
                #     f.write(str(difficult))
                #     f.write("\n")
                    # o = len(labels)
                for j in range(0, len(new_labels)):
                    label = int(new_labels[j])
                    label = rev_label_map[label]
                    f.write(str(label))
                    f.write(' ')

                    xmin = int(new_boxes[j][0])
                    f.write(str(xmin))
                    f.write(' ')
                    ymin = int(new_boxes[j][1])
                    f.write(str(ymin))
                    f.write(' ')
                    w = int(new_boxes[j][2]) - int(new_boxes[j][0])
                    f.write(str(w))
                    f.write(' ')
                    h = int((new_boxes[j][3])) - int(new_boxes[j][1])
                    f.write(str(h))
                    f.write(' ')
                    difficult = int(new_difficulties[j])
                    f.write(str(difficult))
                    f.write("\n")
                # f2 = open("/home/fereshteh/kaist/panno_sani/" + 'I' + '{0:05}'.format(a+1) + '.txt', "w")
                # label = int(labels[j])
                # label = rev_label_map[label]
                # f2.write("nonperson")
                # f2.write(' ')
                #
                # xmin = int(boxes[j][0])
                # f2.write('1')
                # f2.write(' ')
                # ymin = int(boxes[j][1])
                # f2.write('1')
                # f2.write(' ')
                # # w = int(boxes[j][2])-int(boxes[j][0])
                # f2.write(("1"))
                # f2.write(' ')
                # # h = int((boxes[j][3]))-int(boxes[j][1])
                # f2.write("1")
                # f2.write(' ')
                # difficult = int(difficulties[j])
                # f2.write("0")
                # f2.write("\n")
                # o = len(labels)
            a=a+1
                    # print(a)
        id=id+1
        # print(id)
    o = o + 1