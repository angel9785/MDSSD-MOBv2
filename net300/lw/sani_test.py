import xml.etree.ElementTree as ET
import os
from PIL import Image
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None
import re
import torch
import os

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

# for id in range(0,17907):
for filename in sorted(os.listdir('/home/fereshteh/kaist/sanitized_annotations')):
# filename = 'set00_V000_I01225.txt'
# filename ='I00328.txt'
# if (filename == filename):
#     path = "/home/fereshteh/kaist"
    filename2 = re.sub('.txt', ".png", str(filename))
    imagepath = '/home/fereshteh/kaist/sanitized_lw/'+str(filename2)
    # address = os.path.join(path, 'panno_sani', 'I'+  '{0:05}'.format(id) + '.txt')
    # imagepath=os.path.join(path, 'sanitized', 'I'+ '{0:05}'.format(id)+ '.png')
    # imagepath = os.path.join(path, 'sanitized', 'I' + '{0:05}'.format(id) + '.png')
    M = []
    boxes = list()  # torch.FloatTensor([0, 0, 0, 0])#.unsqueeze(0)
    labels = list()
    difficulties = list()
    images = list()

    filename='/home/fereshteh/kaist/sanitized_annotations/'+str(filename)
    # filename = '/home/fereshteh/kaist/sanitized_txt/' + str(filename)
    with open(filename) as f:
        # with open(os.path.join(sani, filename)) as f:
        n = 0
        d = 0
        k=0
        s=0
        for l in f.readlines():
            s=s+1
        for r in range (2,s+1):
            boxes = list()  # torch.FloatTensor([0, 0, 0, 0])#.unsqueeze(0)
            labels = list()
            difficulties = list()
            # images = list()
            k = 0
            with open(filename) as f:
                for line in f.readlines():
                    k=k+1

                    for word in line.split():
                        person = re.findall('person', word)
                        person1 = re.findall('person?', word)
                        person2 = re.findall('person?a', word)
                        people=re.findall('people', word)

                        if (person or person1 or person2 or people):
                            # print(line)
                            i = 0

                            for word in line.split():

                                if (d == 0):
                                    if(i>5):
                                        i=i+1
                                    if (i<6):


                                        if (i == 0):
                                            label = word.lower().strip()
                                            if label=="person?a":
                                                label="person"
                                            if label=="person?":
                                                label="nonperson"
                                            if label == "people":
                                                label = "nonperson"

                                        if (k==r):
                                            # print(label)
                                            if (i == 1):
                                                xmin = int(word)
                                                if (int(xmin) < 150):
                                                    # xmin = xmin1
                                                    left = 0

                                                else:
                                                    # xmin = 150
                                                    left = int(xmin) - 150

                                                # print(x)
                                            if (i == 2):
                                                ymin = int(word)
                                                if (int(ymin) < 150):
                                                    # ymin = ymin1
                                                    top = 0
                                                else:
                                                    # ymin = 150
                                                    top = int(ymin) - 150

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
                                                d = 1
                                            i = i + 1
                                        if (k != r):
                                            # print(label)
                                            if (i == 1):
                                                xmin = int(word)


                                                # print(x)
                                            if (i == 2):
                                                ymin = int(word)


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
                                                d = 1
                                            i = i + 1
                                if (d == 1):
                                    if(k==r):
                                        right = int(xmax) + 150
                                        bottom = int(ymax) + 150
                                        image = Image.open(imagepath)
                                        image1 = image.crop((left, top, right, bottom))
                                        # image.save('/home/fereshteh/kaist/person_sani/'+ 'I'+ '{0:05}'.format(a)+ '.png')
                                        images=torch.FloatTensor([left, top, right, bottom])
                                    # image.save('/home/fereshteh/kaist/test_sani/' + 'I' + '{0:05}'.format(n) + '.png')
                                    # labels.append(label)
                                    labels.append(label_map[label])
                                    boxes.append([xmin, ymin, xmax, ymax])

                                    difficulties.append(difficult)

                                    d = 0
                        else:
                            break
                boxes = torch.FloatTensor(boxes)  # (n_objects, 4)
                # images = torch.FloatTensor(images)
                labels = torch.LongTensor(labels)  # (n_objects)
                difficulties = torch.ByteTensor(difficulties)
                # m = 0
                # for m in range(0, n):
                images = images
                bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.
                centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (
                        bb_centers[:, 1] < bottom)
                if not centers_in_crop.any():
                    continue
                new_boxes = boxes[centers_in_crop, :]
                new_labels = labels[centers_in_crop]
                new_difficulties = difficulties[centers_in_crop]

                # Calculate bounding boxes' new coordinates in the crop
                new_boxes[:, :2] = torch.max(new_boxes[:, :2], images[:2])  # crop[:2] is [left, top]
                new_boxes[:, :2] -= images[:2]
                new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], images[2:])  # crop[2:] is [right, bottom]
                new_boxes[:, 2:] -= images[:2]
                image1.save('/home/fereshteh/kaist/person_sani/' + 'I' + '{0:05}'.format(a) + '.png')
                for i in range(0, len(new_boxes)):
                    if ((image1.width - new_boxes[i, 0]) < 0):
                        print("")
                    if ((image1.width - new_boxes[i, 2]) < 0):
                        print("")

                f = open("/home/fereshteh/kaist/panno_sani/"+ 'I'+ '{0:05}'.format(a)+ '.txt', "w")

                # print(a)
                # f = open("/home/fereshteh/kaist/test_sani/" + 'I' + '{0:05}'.format(m) + '.txt', "w")
                j = 0
                o = len(new_labels)
                for j in range(0,len(new_labels)):


                    label =int(new_labels[j])
                    label=rev_label_map[label]
                    f.write(str(label))
                    f.write(' ')

                    xmin = int(new_boxes[j][0])
                    f.write(str(xmin))
                    f.write(' ')
                    ymin = int(new_boxes[j][1])
                    f.write(str(ymin))
                    f.write(' ')
                    w = int(new_boxes[j][2])-int(new_boxes[j][0])
                    f.write(str(w))
                    f.write(' ')
                    h = int((new_boxes[j][3]))-int(new_boxes[j][1])
                    f.write(str(h))
                    f.write(' ')
                    difficult = int(new_difficulties[j])
                    f.write(str(difficult))
                    f.write("\n")
            a=a+1
            print(a)
    id=id+1
    print(id)
