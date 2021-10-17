import json
import os
import torch
import random
import xml.etree.ElementTree as ET
import torch.nn.functional as F
import torchvision.transforms.functional as FT
import matplotlib.pyplot as plt
# from torchvision import transforms
import cv2
import math
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label map
# voc_labels = ('person', 'cyclist','person?','people','nonperson')
kaist_labels = ('nonperson','person')

label_map = {k: v for v, k in enumerate(kaist_labels)}
# label_map['background'] = 0
# label_map['person'] = 1
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

# Color map for bounding boxes of detected objects from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                   '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff', '#808080', '#FFFFFF']
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}


def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = list()#torch.FloatTensor([0, 0, 0, 0])#.unsqueeze(0)
    labels = list()
    difficulties = list()
    for object in root.iter('object'):

        difficult = int(object.find('difficult').text == '1')
        if (difficult==0):
            label = object.find('name').text.lower().strip()
        # if label not in label_map:
        #     continue

            bbox = object.find('bndbox')

            xmin = int(bbox.find('x').text)-1
            ymin = int(bbox.find('y').text)-1
            w=int(bbox.find('w').text) if int(bbox.find('w').text)>0 else 1
            xmax = xmin+w
            h=int(bbox.find('h').text) if int(bbox.find('h').text)>0 else 1
            ymax = ymin+h

            if xmin<=0 :
                xmin=0

            if ymin<=0 :
                ymin=0

            if xmax>=639:
                xmax = 639

            if ymax>=511:
                ymax =511

            if label=='cyclist':
                label='person'
            if label=='person?':
                label='nonperson'    
            if label=='people':
                label='nonperson'
        # if label=='nonperson':
        #     label='background'
            
            labels.append(label_map[label])
            boxes.append([xmin, ymin, xmax, ymax])
        
        
        # h=int(bbox.find('h').text)
        # if h<50:
        #     labels.append(label_map['nonperson'])
        # else:
           
            
            difficulties.append(difficult)
    if (boxes.__len__() == 0):
        boxes.append([0, 0, 1., 1.])
    
        labels.append(label_map['nonperson'])
        difficulties.append(0)
        
        
    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}

import re
def parse_annotation1(filename):
    M = []
    boxes = list()  # torch.FloatTensor([0, 0, 0, 0])#.unsqueeze(0)
    labels = list()
    difficulties = list()
    # filename='/home/fereshteh/kaist/annotations_KAIST_testset/annotations_KAIST_test_set/'+str(filename)
    with open(filename) as f:
    # with open(os.path.join(sani, filename)) as f:
        for line in f.readlines():
            for word in line.split():
                # person = re.findall('person', word)
                person = re.findall('person', word)
                person1 = re.findall('person?', word)
                person2 = re.findall('person?a', word)
                people = re.findall('people', word)
                cyclist = re.findall('cyclist', word)
                if (person or person1 or person2 or people or cyclist):
                # if (person):

                # if person:
                    # print(line)
                    i = 0
                    for word in line.split():

                        if (i == 0):
                            label = word.lower().strip()
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
                        i = i + 1
                            # print(dificualt)
                    # if h <= 0:
                    #     print("")
                    # if w <= 0:
                    #     print("")
                    # if xmin <= 0:
                    #     xmin = 0
                    # if ymin <= 0:
                    #     ymin = 0
                    # if xmax >= 639:
                    #     xmax = 639
                    # if ymax >= 511:
                    #     ymax = 511

                    if label == 'person?a':
                        label = 'person'
                    if label == 'person?':
                        label = 'person'

                    if label == "people":
                        label = "person"

                    if label == "cyclist":
                        label = "person"

                    # if (difficult == 2):
                    #     print("")
                    # if (difficult==0):
                    labels.append(label_map[label])
                    boxes.append([xmin, ymin, xmax, ymax])
                    difficulties.append(difficult)
        if (boxes.__len__() == 0):
            boxes.append([0, 0, 1., 1.])

            labels.append(label_map['nonperson'])
            difficulties.append(0)



    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}
import re

def parse_annotation_val(filename):
    M = []
    boxes = list()  # torch.FloatTensor([0, 0, 0, 0])#.unsqueeze(0)
    labels = list()
    difficulties = list()
    # filename='/home/fereshteh/kaist/annotations_KAIST_testset/annotations_KAIST_test_set/'+str(filename)
    with open(filename) as f:
    # with open(os.path.join(sani, filename)) as f:
        for line in f.readlines():
            for word in line.split():
                # person = re.findall('person', word)
                person = re.findall('person', word)
                person1 = re.findall('person?', word)
                person2 = re.findall('person?a', word)
                people = re.findall('people', word)
                cyclist = re.findall('cyclist', word)
                if (person or person1 or person2 or people or cyclist):
                # if (person):

                # if person:
                    # print(line)
                    i = 0
                    for word in line.split():

                        if (i == 0):
                            label = word.lower().strip()
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
                        i = i + 1
                            # print(dificualt)
                    # if h <= 0:
                    #     print("")
                    # if w <= 0:
                    #     print("")
                    if xmin <= 0:
                        xmin = 0
                    if ymin <= 0:
                        ymin = 0
                    # if xmax >= 639:
                    #     xmax = 639
                    # if ymax >= 511:
                    #     ymax = 511
                    if label == 'person':
                        if h>20 :
                            # if difficult!=2:
                            label = 'person'
                        else:
                            label = 'nonperson'
                        # else:
                        #     label = 'nonperson'
                    if label == 'person?a':
                        if h > 20:
                            # if difficult!=2:
                            label = 'person'
                        else:
                            label = 'nonperson'
                        # else:
                        #     label = 'nonperson'
                    if label == 'person?':
                        if h > 20:
                            # if difficult!=2:
                            label = 'person'
                        else:
                            label = 'nonperson'
                        # else:
                        #     label = 'nonperson'
                    if label == "people":
                        label = "nonperson"

                    if label == "cyclist":
                        if h > 20:
                            # if difficult!=2:
                            label = 'person'
                        else:
                            label = 'nonperson'
                        # else:
                        #     label = 'nonperson'ave


                    labels.append(label_map[label])
                    boxes.append([xmin, ymin, xmax, ymax])
                    difficulties.append(difficult)
        if (boxes.__len__() == 0):
            boxes.append([0, 0, 1., 1.])

            labels.append(label_map['nonperson'])
            difficulties.append(0)



    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}
def parse_annotation_train(filename):
    M = []
    boxes = list()  # torch.FloatTensor([0, 0, 0, 0])#.unsqueeze(0)
    labels = list()
    difficulties = list()
    # filename='/home/fereshteh/kaist/annotations_KAIST_testset/annotations_KAIST_test_set/'+str(filename)
    with open(filename) as f:
    # with open(os.path.join(sani, filename)) as f:
        for line in f.readlines():
            for word in line.split():
                # person = re.findall('person', word)
                person = re.findall('person', word)
                person1 = re.findall('person?', word)
                person2 = re.findall('person?a', word)
                people = re.findall('people', word)
                cyclist = re.findall('cyclist', word)
                if (person or person1 or person2 or people or cyclist):
                # if (person):

                # if person:
                    # print(line)
                    i = 0
                    for word in line.split():

                        if (i == 0):
                            label = word.lower().strip()
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
                        i = i + 1
                            # print(dificualt)
                    # if h <= 0:
                    #     print("")
                    # if w <= 0:
                    #     print("")
                    if xmin <= 0:
                        xmin = 0
                    if ymin <= 0:
                        ymin = 0
                    # if xmax >= 639:
                    #     xmax = 639
                    # if ymax >= 511:
                    #     ymax = 511
                    if label == 'person':
                        # if h>50 :
                        # if difficult != 2:
                        label = 'person'
                        # else:
                        #     label = 'nonperson'
                        # else:
                        #     label = 'nonperson'
                    if label == 'person?a':
                        # if h>50 :
                        # if difficult != 2:
                        label = 'person'
                        # else:
                        #     label = 'nonperson'
                        # else:
                        #     label = 'nonperson'
                    if label == 'person?':
                        # if h>50 :
                        # if difficult != 2:
                        label = 'person'
                        # else:
                        #     label = 'nonperson'
                        # else:
                        #     label = 'nonperson'

                    if label == "people":
                        label = "nonperson"

                    if label == "cyclist":
                        # if h>50 :
                        # if difficult != 2:
                        label = 'person'
                        # else:
                        #     label = 'nonperson'
                        # else:
                        #     label = 'nonperson'

                    labels.append(label_map[label])
                    boxes.append([xmin, ymin, xmax, ymax])
                    difficulties.append(difficult)
        if (boxes.__len__() == 0):
            boxes.append([0, 0, 1., 1.])

            labels.append(label_map['nonperson'])
            difficulties.append(0)



    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}

test_labels=('nonperson','person',"ignore")
test_labels_map={k: v for v, k in enumerate(test_labels)}
def parse_annotation2(filename):
    M = []
    boxes = list()  # torch.FloatTensor([0, 0, 0, 0])#.unsqueeze(0)
    labels = list()
    difficulties = list()
    # filename='/home/fereshteh/kaist/annotations_KAIST_testset/annotations_KAIST_test_set/'+str(filename)
    with open(filename) as f:
    # with open(os.path.join(sani, filename)) as f:
        for line in f.readlines():
            for word in line.split():
                # person = re.findall('person', word)
                person = re.findall('person', word)
                person1 = re.findall('person?', word)
                person2 = re.findall('person?a', word)
                people = re.findall('people', word)
                cyclist = re.findall('cyclist', word)
                if (person or person1 or person2 or people or cyclist):
                # if (person):

                # if person:
                    # print(line)
                    i = 0
                    for word in line.split():

                        if (i == 0):
                            label = word.lower().strip()
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
                        i = i + 1
                            # print(dificualt)
                    # if h <= 0:
                    #     print("")
                    # if w <= 0:
                    #     print("")
                    if xmin <= 0:
                        xmin = 0
                    if ymin <= 0:
                        ymin = 0
                    # if xmax >= 639:
                    #     xmax = 639
                    # if ymax >= 511:
                    #     ymax = 511

                    if label == 'person?a':
                        label = 'ignore'
                    if label == 'person?':
                        label = 'ignore'

                    if label == "people":
                        label = "ignore"

                    if label == "cyclist":
                        label = "ignore"


                    labels.append(test_labels_map[label])
                    boxes.append([xmin, ymin, xmax, ymax])
                    difficulties.append(difficult)
        if (boxes.__len__() == 0):
            boxes.append([0, 0, 1., 1.])

            labels.append(label_map['nonperson'])
            difficulties.append(0)



    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}

def create_data_lists(kaist_path, output_folder):
    """
    Create lists of images, the bounding boxes and labels of the objects in these images, and save these to file.

    :param voc07_path: path to the 'VOC2007' folder
    :param voc12_path: path to the 'VOC2012' folder
    :param output_folder: folder where the JSONs must be saved
    """
    kaist_path = os.path.abspath(kaist_path)

    kaist_path = os.path.abspath(kaist_path)
    txt_images = list()
    # txt_images_lw = list()
    txt_objects = list()
    n_objects = 0
    # voc07_path = os.path.abspath(sani)
    # sani=os.path.join(kaist_path, 'sanitest_txt')
    # for path in sani:
    sanitrain_rgb = '/home/fereshteh/kaist/sanitized_annotations1/sanitized_lw'
    # sanitrain_lw = '/home/fereshteh/kaist/person_sani_lw'
    sanitrain1 = '/home/fereshteh/kaist/sanitized_annotations1/sanitized_annotations/'
    i = 0
    for filename in sorted(os.listdir('/home/fereshteh/kaist/sanitized_annotations1/sanitized_annotations')):
        # if (i < 31910):  # 1430 31910
        if filename.endswith('.txt'):
            filename1 = os.path.join(sanitrain1, filename)

            # with open (path) as f:
            objects = parse_annotation_train(filename1)
            txt_objects.append(objects)
            n_objects += len(objects)
            # if len(objects) == 0:
            #     continue
            filename2 = re.sub('.txt', ".png", str(filename))
            txt_images.append(os.path.join(sanitrain_rgb, filename2))
            # txt_images_lw.append(os.path.join(sanitrain_lw, filename2))
            i = i + 1
    for filename in sorted(os.listdir('/home/fereshteh/kaist/sanitized_annotations1/sanitized_annotations')):
        # if (i < 31910):  # 1430 31910
        if filename.endswith('.txt'):
            filename1 = os.path.join(sanitrain1, filename)

            # with open (path) as f:
            objects = parse_annotation_train(filename1)
            txt_objects.append(objects)
            n_objects += len(objects)
            # if len(objects) == 0:
            #     continue
            filename2 = re.sub('.txt', ".png", str(filename))
            txt_images.append(os.path.join(sanitrain_rgb, filename2))
            # txt_images_lw.append(os.path.join(sanitrain_lw, filename2))
            i = i + 1
    for filename in sorted(os.listdir('/home/fereshteh/kaist/sanitized_annotations1/sanitized_annotations')):
        # if (i < 31910):  # 1430 31910
        if filename.endswith('.txt'):
            filename1 = os.path.join(sanitrain1, filename)

            # with open (path) as f:
            objects = parse_annotation_train(filename1)
            txt_objects.append(objects)
            n_objects += len(objects)
            # if len(objects) == 0:
            #     continue
            filename2 = re.sub('.txt', ".png", str(filename))
            txt_images.append(os.path.join(sanitrain_rgb, filename2))
            # txt_images_lw.append(os.path.join(sanitrain_lw, filename2))
            i = i + 1
    for filename in sorted(os.listdir('/home/fereshteh/kaist/sanitized_annotations1/sanitized_annotations')):
        # if (i < 31910):  # 1430 31910
        if filename.endswith('.txt'):
            filename1 = os.path.join(sanitrain1, filename)

            # with open (path) as f:
            objects = parse_annotation_train(filename1)
            txt_objects.append(objects)
            n_objects += len(objects)
            # if len(objects) == 0:
            #     continue
            filename2 = re.sub('.txt', ".png", str(filename))
            txt_images.append(os.path.join(sanitrain_rgb, filename2))
            # txt_images_lw.append(os.path.join(sanitrain_lw, filename2))
            i = i + 1

    assert len(txt_objects) == len(txt_images)
    #
    # Save to file
    with open(os.path.join(output_folder, 'SANITRAIN_images.json'), 'w') as j:
        json.dump(txt_images, j)
    # with open(os.path.join(output_folder, 'ALLSANITEST_P_images_lw.json'), 'w') as j:
    #     json.dump(txt_images_lw, j)
    # with open(os.path.join(output_folder, 'ALLSANITEST_images.json'), 'w') as j:

    with open(os.path.join(output_folder, 'SANITRAIN_objects.json'), 'w') as j:
        json.dump(txt_objects, j)
    #
    print('\nThere are %d sanitrain images containing a total of %d objects. Files have been saved to %s.' % (
        len(txt_images), n_objects, os.path.abspath(output_folder)))


    kaist_path = os.path.abspath(kaist_path)
    txt_images1 = list()
    # txt_images_lw = list()
    txt_objects1 = list()
    n_objects1 = 0
    # voc07_path = os.path.abspath(sani)
    # sani=os.path.join(kaist_path, 'sanitest_txt')
    # for path in sani:
    sanival_rgb = '/home/fereshteh/kaist/sanitest_lw'
    # sanitrain_lw = '/home/fereshteh/kaist/person_sani_lw'
    sanival = '/home/fereshteh/kaist/annotations_KAIST_testset1/annotations_KAIST_test_set/'
    i = 0
    for filename in sorted(os.listdir('/home/fereshteh/kaist/annotations_KAIST_testset1/annotations_KAIST_test_set')):
        # if (i < 31910):  # 1430 31910
        if filename.endswith('.txt'):
            filename1 = os.path.join(sanival, filename)

            # with open (path) as f:
            objects1 = parse_annotation_val(filename1)
            txt_objects1.append(objects1)
            n_objects1 += len(objects1)
            # if len(objects) == 0:
            #     continue
            filename2 = re.sub('.txt', ".png", str(filename))
            txt_images1.append(os.path.join(sanival_rgb, filename2))
            # txt_images_lw.append(os.path.join(sanitrain_lw, filename2))
            i = i + 1

    assert len(txt_objects1) == len(txt_images1)
    #
    # Save to file
    with open(os.path.join(output_folder, 'SANIVAL_images.json'), 'w') as j:
        json.dump(txt_images1, j)
    # with open(os.path.join(output_folder, 'ALLSANITEST_P_images_lw.json'), 'w') as j:
    #     json.dump(txt_images_lw, j)
    # with open(os.path.join(output_folder, 'ALLSANITEST_images.json'), 'w') as j:

    with open(os.path.join(output_folder, 'SANIVAL_objects.json'), 'w') as j:
        json.dump(txt_objects1, j)
    #
    print('\nThere are %d sanival images containing a total of %d objects. Files have been saved to %s.' % (
        len(txt_images1), n_objects1, os.path.abspath(output_folder)))
    #
    #



    kaist_path = os.path.abspath(kaist_path)
    txt_images2 = list()
    # txt_images_lw = list()
    txt_objects2= list()
    n_objects2 = 0
    # voc07_path = os.path.abspath(sani)
    # sani=os.path.join(kaist_path, 'sanitest_txt')
    # for path in sani:
    sanitest_rgb = '/home/fereshteh/kaist/sanitest_lw'
    # sanitrain_lw = '/home/fereshteh/kaist/person_sani_lw'
    sanitest = '/home/fereshteh/kaist/annotations_KAIST_testset1/annotations_KAIST_test_set/'
    i = 0
    for filename in sorted(os.listdir('/home/fereshteh/kaist/annotations_KAIST_testset1/annotations_KAIST_test_set')):
        if (i >= 1455):  # 1430 31910
            if filename.endswith('.txt'):
                filename1 = os.path.join(sanitest, filename)

                # with open (path) as f:
                objects2 = parse_annotation2(filename1)
                txt_objects2.append(objects2)
                n_objects2 += len(objects2)
                # if len(objects) == 0:
                #     continue
                filename2 = re.sub('.txt', ".png", str(filename))
                txt_images2.append(os.path.join(sanitest_rgb, filename2))
                # txt_images_lw.append(os.path.join(sanitrain_lw, filename2))
        i = i + 1

    assert len(txt_objects2) == len(txt_images2)
    #
    # Save to file
    with open(os.path.join(output_folder, 'nSANITEST1_images.json'), 'w') as j:
        json.dump(txt_images2, j)
    # with open(os.path.join(output_folder, 'ALLSANITEST_P_images_lw.json'), 'w') as j:
    #     json.dump(txt_images_lw, j)
    # with open(os.path.join(output_folder, 'ALLSANITEST_images.json'), 'w') as j:

    with open(os.path.join(output_folder, 'nSANITEST1_objects.json'), 'w') as j:
        json.dump(txt_objects2, j)
    #
    print('\nThere are %d sanitest images containing a total of %d objects. Files have been saved to %s.' % (
        len(txt_images2), n_objects2, os.path.abspath(output_folder)))

    '''
    # txt_images = list()
    # # txt_images_lw = list()
    # txt_objects = list()
    # n_objects = 0
    # # voc07_path = os.path.abspath(sani)
    # # sani=os.path.join(kaist_path, 'sanitest_txt')
    # # for path in sani:
    # sanitrain_rgb = '/home/fereshteh/kaist/sanitized_lw'
    # # sanitrain_lw = '/home/fereshteh/kaist/person_sani_lw'
    # sanitrain1 = '/home/fereshteh/kaist/sanitized_txt/'
    # i = 0
    # for filename in sorted(os.listdir('/home/fereshteh/kaist/sanitized_txt')):
    #     # if (i < 31910):  # 1430 31910
    #     if filename.endswith('.txt'):
    #         filename1 = os.path.join(sanitrain1, filename)
    #
    #         # with open (path) as f:
    #         objects = parse_annotation1(filename1)
    #         txt_objects.append(objects)
    #         n_objects += len(objects)
    #         # if len(objects) == 0:
    #         #     continue
    #         filename2 = re.sub('.txt', ".png", str(filename))
    #         txt_images.append(os.path.join(sanitrain_rgb, filename2))
    #         # txt_images_lw.append(os.path.join(sanitrain_lw, filename2))
    #         i = i + 1
    #
    # assert len(txt_objects) == len(txt_images)
    # #
    # # Save to file
    # with open(os.path.join(output_folder, 'SANITRAIN_images.json'), 'w') as j:
    #     json.dump(txt_images, j)
    # # with open(os.path.join(output_folder, 'ALLSANITEST_P_images_lw.json'), 'w') as j:
    # #     json.dump(txt_images_lw, j)
    # # with open(os.path.join(output_folder, 'ALLSANITEST_images.json'), 'w') as j:
    #
    # with open(os.path.join(output_folder, 'SANITRAIN_objects.json'), 'w') as j:
    #     json.dump(txt_objects, j)
    # #
    # print('\nThere are %d sanitrain images containing a total of %d objects. Files have been saved to %s.' % (
    #     len(txt_images), n_objects, os.path.abspath(output_folder)))
    #
    # kaist_path = os.path.abspath(kaist_path)
    # txt_images1 = list()
    # # txt_images_lw = list()
    # txt_objects1 = list()
    # n_objects1 = 0
    # # voc07_path = os.path.abspath(sani)
    # # sani=os.path.join(kaist_path, 'sanitest_txt')
    # # for path in sani:
    # sanitval_rgb = '/home/fereshteh/kaist/eval_lw'
    # # sanitrain_lw = '/home/fereshteh/kaist/person_sani_lw'
    # sanival = '/home/fereshteh/kaist/eval_txt/'
    # i = 0
    # for filename in sorted(os.listdir('/home/fereshteh/kaist/eval_txt')):
    #     # if (i < 31910):  # 1430 31910
    #     if filename.endswith('.txt'):
    #         filename1 = os.path.join(sanival, filename)
    #
    #         # with open (path) as f:
    #         objects1 = parse_annotation1(filename1)
    #         txt_objects1.append(objects1)
    #         n_objects1 += len(objects1)
    #         # if len(objects) == 0:
    #         #     continue
    #         filename2 = re.sub('.txt', ".png", str(filename))
    #         txt_images1.append(os.path.join(sanitval_rgb, filename2))
    #         # txt_images_lw.append(os.path.join(sanitrain_lw, filename2))
    #         i = i + 1
    #
    # assert len(txt_objects1) == len(txt_images1)
    # #
    # # Save to file
    # with open(os.path.join(output_folder, 'SANIVAL_images.json'), 'w') as j:
    #     json.dump(txt_images1, j)
    # # with open(os.path.join(output_folder, 'ALLSANITEST_P_images_lw.json'), 'w') as j:
    # #     json.dump(txt_images_lw, j)
    # # with open(os.path.join(output_folder, 'ALLSANITEST_images.json'), 'w') as j:
    #
    # with open(os.path.join(output_folder, 'SANIVAL_objects.json'), 'w') as j:
    #     json.dump(txt_objects1, j)
    # #
    # print('\nThere are %d sanival images containing a total of %d objects. Files have been saved to %s.' % (
    #     len(txt_images1), n_objects1, os.path.abspath(output_folder)))

    kaist_path = os.path.abspath(kaist_path)
    txt_images2 = list()
    # txt_images_lw = list()
    txt_objects2 = list()
    n_objects2 = 0
    # voc07_path = os.path.abspath(sani)
    # sani=os.path.join(kaist_path, 'sanitest_txt')
    # for path in sani:
    sanitest_rgb = '/home/fereshteh/kaist/sanitest_lw'
    # sanitrain_lw = '/home/fereshteh/kaist/person_sani_lw'
    sanitest = '/home/fereshteh/kaist/annotations_KAIST_testset/annotations_KAIST_test_set/'
    i = 0
    for filename in sorted(os.listdir('/home/fereshteh/kaist/annotations_KAIST_testset/annotations_KAIST_test_set')):
        # if (i < 31910):  # 1430 31910
        if filename.endswith('.txt'):
            filename1 = os.path.join(sanitest, filename)

            # with open (path) as f:
            objects2 = parse_annotation1(filename1)
            txt_objects2.append(objects2)
            n_objects2 += len(objects2)
            # if len(objects) == 0:
            #     continue
            filename2 = re.sub('.txt', ".png", str(filename))
            txt_images2.append(os.path.join(sanitest_rgb, filename2))
            # txt_images_lw.append(os.path.join(sanitrain_lw, filename2))
            i = i + 1

    assert len(txt_objects2) == len(txt_images2)
    #
    # Save to file
    with open(os.path.join(output_folder, 'SANITEST_images.json'), 'w') as j:
        json.dump(txt_images2, j)
    # with open(os.path.join(output_folder, 'ALLSANITEST_P_images_lw.json'), 'w') as j:
    #     json.dump(txt_images_lw, j)
    # with open(os.path.join(output_folder, 'ALLSANITEST_images.json'), 'w') as j:

    with open(os.path.join(output_folder, 'SANITEST_objects.json'), 'w') as j:
        json.dump(txt_objects2, j)
    #
    print('\nThere are %d sanitest images containing a total of %d objects. Files have been saved to %s.' % (
        len(txt_images2), n_objects2, os.path.abspath(output_folder)))
    
    sanitrain_images = list()
    sanitrain_objects = list()
    n_objects = 0
    for id in range(0, 31910):
        # for id in range(19, 29178,20):  # (15001,17913):
        # Parse annotation's XML file

        objects = parse_annotation1(os.path.join(kaist_path, 'sanitized_txt', 'I' + '{0:05}'.format(id) + '.txt'))
        # if len(objects) == 0:
        #     continue
        sanitrain_objects.append(objects)
        n_objects += len(objects)
        sanitrain_images.append(os.path.join(kaist_path, 'sanitized_lw', 'I' + '{0:05}'.format(id) + '.png'))

    assert len(sanitrain_objects) == len(sanitrain_images)
    #
    # Save to file
    with open(os.path.join(output_folder, 'SANITRAIN_images.json'), 'w') as j:
        json.dump(sanitrain_images, j)
    with open(os.path.join(output_folder, 'SANITRAIN_objects.json'), 'w') as j:
        json.dump(sanitrain_objects, j)

    print('\nThere are %d sanitrain images containing a total of %d objects. Files have been saved to %s.' % (
        len(sanitrain_images), n_objects, os.path.abspath(output_folder)))
    # txt_images = list()
    # txt_objects = list()
    # n_objects = 0
    # # # # voc07_path = os.path.abspath(sani)
    # # # # sani=os.path.join(kaist_path, 'sanitest_txt')
    # # # # for path in sani:
    # sanitest='/home/fereshteh/kaist/sanitest_lw'
    # sanitest1='/home/fereshteh/kaist/annotations_KAIST_testset/annotations_KAIST_test_set/'
    # i=0
    # for filename in sorted(os.listdir('/home/fereshteh/kaist/annotations_KAIST_testset/annotations_KAIST_test_set')):
    #     if(i<2254):#1430
    #         if filename.endswith('.txt'):
    #             filename1=os.path.join(sanitest1,filename)
    #
    #             # with open (path) as f:
    #             objects = parse_annotation1(filename1)
    #             txt_objects.append(objects)
    #             n_objects += len(objects)
    #         # if len(objects) == 0:
    #         #     continue
    #             filename2 = re.sub('.txt', ".png", str(filename))
    #             txt_images.append(os.path.join(sanitest,filename2))
    #             i = i + 1
    #
    # # for id in range(0, 2252):  # (15001,17913):
    #     # Parse annotation's XML file
    #
    #     objects = parse_annotation1(os.path.join(kaist_path, 'sanitest_txt', 'I' + '{0:05}'.format(id) + '.txt'))
    #     if len(objects) == 0:
    #         continue
    #
    #     txt_images.append(os.path.join(kaist_path, 'sanitest', 'I' + '{0:05}'.format(id) + '.png'))
    #
    # assert len(txt_objects) == len(txt_images)
    # #
    # # Save to file
    # with open(os.path.join(output_folder, 'ALLSANITEST_images.json'), 'w') as j:
    #     json.dump(txt_images, j)
    # with open(os.path.join(output_folder, 'ALLSANITEST_objects.json'), 'w') as j:
    #     json.dump(txt_objects, j)
    # #
    # print('\nThere are %d sani images containing a total of %d objects. Files have been saved to %s.' % (
    #     len(txt_images), n_objects, os.path.abspath(output_folder)))

    #
    txt_images1 = list()
    txt_objects1 = list()
    n_objects = 0
    sanitest = '/home/fereshteh/kaist/sanitest_lw'
    sanitest1 = '/home/fereshteh/kaist/annotations_KAIST_testset/annotations_KAIST_test_set/'
    i = 0
    for filename in sorted(os.listdir('/home/fereshteh/kaist/annotations_KAIST_testset/annotations_KAIST_test_set')):
        if (i < 1430):  # 1430
            if filename.endswith('.txt'):
                filename1 = os.path.join(sanitest1, filename)

                # with open (path) as f:
                objects = parse_annotation1(filename1)
                txt_objects1.append(objects)
                n_objects += len(objects)
                # if len(objects) == 0:
                #     continue
                filename2 = re.sub('.txt', ".png", str(filename))
                txt_images1.append(os.path.join(sanitest, filename2))
                i = i + 1

    # for id in range(0, 2252):  # (15001,17913):
    # Parse annotation's XML file

    # objects = parse_annotation1(os.path.join(kaist_path, 'sanitest_txt', 'I' + '{0:05}'.format(id) + '.txt'))
    # if len(objects) == 0:
    #     continue

    # txt_images.append(os.path.join(kaist_path, 'sanitest', 'I' + '{0:05}'.format(id) + '.png'))

    assert len(txt_objects1) == len(txt_images1)
    #
    # Save to file
    with open(os.path.join(output_folder, 'SANITEST_images.json'), 'w') as j:
        json.dump(txt_images1, j)
    with open(os.path.join(output_folder, 'SANITEST_objects.json'), 'w') as j:
        json.dump(txt_objects1, j)
    #
    print('\nThere are %d day sani images containing a total of %d objects. Files have been saved to %s.' % (
        len(txt_images1), n_objects, os.path.abspath(output_folder)))
    #
    
    # sanitrain_rgb = '/home/fereshteh/kaist/sanitized_lw'
    # # sanitrain_lw = '/home/fereshteh/kaist/sanitized_lw'
    # sanitrain1 = '/home/fereshteh/kaist/sanitized_txt/'
    # sanitrain_images = list()
    # sanitrain_objects = list()
    # n_objects = 0
    # i = 0
    # # for id in range(0, 30062):
    # for filename in sorted(os.listdir('/home/fereshteh/kaist/sanitized_txt')):
    #     # for id in range(19, 29178,20):  # (15001,17913):
    #     # Parse annotation's XML file
    #
    #     if filename.endswith('.txt'):
    #         filename1 = os.path.join(sanitrain1, filename)
    #
    #         # with open (path) as f:
    #         objects = parse_annotation1(filename1)
    #         sanitrain_objects.append(objects)
    #         n_objects += len(objects)
    #         # if len(objects) == 0:
    #         #     continue
    #         filename2 = re.sub('.txt', ".png", str(filename))
    #         sanitrain_images.append(os.path.join(sanitrain_rgb, filename2))
    #
    #         i = i + 1
    #
    # assert len(sanitrain_objects) == len(sanitrain_images)
    # #
    # # Save to file
    # with open(os.path.join(output_folder, 'SANITRAIN1_images.json'), 'w') as j:
    #     json.dump(sanitrain_images, j)
    # with open(os.path.join(output_folder, 'SANITRAIN1_objects.json'), 'w') as j:
    #     json.dump(sanitrain_objects, j)
    #
    # print('\nThere are %d sanitrain images containing a total of %d objects. Files have been saved to %s.' % (
    #     len(sanitrain_images), n_objects, os.path.abspath(output_folder)))
    txt_images = list()
    txt_objects = list()
    n_objects = 0
    # # voc07_path = os.path.abspath(sani)
    # # sani=os.path.join(kaist_path, 'sanitest_txt')
    # # for path in sani:
    sanitest = '/home/fereshteh/kaist/eval_lw'

    sanitest1 = '/home/fereshteh/kaist/eval_txt/'
    i = 0
    for filename in sorted(os.listdir('/home/fereshteh/kaist/eval_txt')):
        # if(i<2254):#1430
        if filename.endswith('.txt'):
            filename1 = os.path.join(sanitest1, filename)

            # with open (path) as f:
            objects = parse_annotation1(filename1)
            txt_objects.append(objects)
            n_objects += len(objects)
            # if len(objects) == 0:
            #     continue
            filename2 = re.sub('.txt', ".png", str(filename))
            txt_images.append(os.path.join(sanitest, filename2))

            i = i + 1

    # for id in range(0, 2252):  # (15001,17913):
    #     # Parse annotation's XML file
    #
    #     objects = parse_annotation1(os.path.join(kaist_path, 'sanitest_txt', 'I' + '{0:05}'.format(id) + '.txt'))
    #     if len(objects) == 0:
    #         continue
    #
    #     txt_images.append(os.path.join(kaist_path, 'sanitest', 'I' + '{0:05}'.format(id) + '.png'))

    assert len(txt_objects) == len(txt_images)
    # Save to file
    with open(os.path.join(output_folder, 'ALLSANIVAL1_images.json'), 'w') as j:
        json.dump(txt_images, j)

    # with open(os.path.join(output_folder, 'ALLSANITEST_images.json'), 'w') as j:

    with open(os.path.join(output_folder, 'ALLSANIVAL1_objects.json'), 'w') as j:
        json.dump(txt_objects, j)
    #
    print('\nThere are %d sani images containing a total of %d objects. Files have been saved to %s.' % (
        len(txt_images), n_objects, os.path.abspath(output_folder)))
    # txt_images = list()
    # # txt_images_lw = list()
    # txt_objects = list()
    # n_objects = 0
    # # voc07_path = os.path.abspath(sani)
    # # sani=os.path.join(kaist_path, 'sanitest_txt')
    # # for path in sani:
    # sanitrain_rgb = '/home/fereshteh/kaist/person_sani_lw'
    # # sanitrain_lw = '/home/fereshteh/kaist/person_sani_lw'
    # sanitrain1 = '/home/fereshteh/kaist/panno_sani/'
    # i = 0
    # for filename in sorted(os.listdir('/home/fereshteh/kaist/panno_sani')):
    #     # if (i < 31910):  # 1430 31910
    #     if filename.endswith('.txt'):
    #         filename1 = os.path.join(sanitrain1, filename)
    #
    #         # with open (path) as f:
    #         objects = parse_annotation1(filename1)
    #         txt_objects.append(objects)
    #         n_objects += len(objects)
    #         # if len(objects) == 0:
    #         #     continue
    #         filename2 = re.sub('.txt', ".png", str(filename))
    #         txt_images.append(os.path.join(sanitrain_rgb, filename2))
    #         # txt_images_lw.append(os.path.join(sanitrain_lw, filename2))
    #         i = i + 1
    #
    # # for id in range(0, 2252):  # (15001,17913):
    # #     # Parse annotation's XML file
    # #
    # #     objects = parse_annotation1(os.path.join(kaist_path, 'sanitest_txt', 'I' + '{0:05}'.format(id) + '.txt'))
    # #     if len(objects) == 0:
    # #         continue
    # #
    # #     txt_images.append(os.path.join(kaist_path, 'sanitest', 'I' + '{0:05}'.format(id) + '.png'))
    #
    # assert len(txt_objects) == len(txt_images)
    # #
    # # Save to file
    # with open(os.path.join(output_folder, 'ALLSANITEST_P_images.json'), 'w') as j:
    #     json.dump(txt_images, j)
    # # with open(os.path.join(output_folder, 'ALLSANITEST_P_images_lw.json'), 'w') as j:
    # #     json.dump(txt_images_lw, j)
    # # with open(os.path.join(output_folder, 'ALLSANITEST_images.json'), 'w') as j:
    #
    # with open(os.path.join(output_folder, 'ALLSANITEST_P_objects.json'), 'w') as j:
    #     json.dump(txt_objects, j)
    # #
    # print('\nThere are %d sani images containing a total of %d objects. Files have been saved to %s.' % (
    #     len(txt_images), n_objects, os.path.abspath(output_folder)))
    #
'''
def decimate(tensor, m):
    """
    Decimate a tensor by a factor 'm', i.e. downsample by keeping every 'm'th value.

    This is used when we convert FC layers to equivalent Convolutional layers, BUT of a smaller size.

    :param tensor: tensor to be decimated
    :param m: list of decimation factors for each dimension of the tensor; None if not to be decimated along a dimension
    :return: decimated tensor
    """
    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d,
                                         index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())

    return tensor

def calculate_mAP1(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties):
    """
    Calculate the Mean Average Precision (mAP) of detected objects.

    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation

    :param det_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
    :param det_labels: list of tensors, one tensor for each image containing detected objects' labels
    :param det_scores: list of tensors, one tensor for each image containing detected objects' labels' scores
    :param true_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
    :param true_labels: list of tensors, one tensor for each image containing actual objects' labels
    :param true_difficulties: list of tensors, one tensor for each image containing actual objects' difficulty (0 or 1)
    :return: list of average precisions for all classes, mean average precision (mAP)
    """
    global n_easy_class_objects
    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(
        true_labels) == len(
        true_difficulties)  # these are all lists of tensors of the same length, i.e. number of images
    n_classes = len(label_map)

    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(
        device)  # (n_objects), n_objects is the total no. of objects across all images
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
    true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    # Calculate APs for each class (except background)
    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    precision = torch.zeros((n_classes - 1), dtype=torch.float)
    recall = torch.zeros((n_classes - 1), dtype=torch.float)
    true_positives = torch.zeros((n_classes - 1), dtype=torch.float)
    false_positives = torch.zeros((n_classes - 1), dtype=torch.float)
    # pre=torch.zeros((n_classes - 1), dtype=torch.float)
    f = plt.figure()

    for c in range(1, n_classes):
        # Extract only objects with this class
        true_class_images1 = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes1 = true_boxes[true_labels == c]  # (n_class_objects, 4)
        true_class_difficulties1 = true_difficulties[true_labels == c]  # (n_class_objects)
        n_class_objects1 = true_class_boxes1.size(0)
        # wt=true_class_boxes1[:][1]-true_class_boxes1[:][0]
        # ht=true_class_boxes1[:][3]-true_class_boxes1[:][2]
        true_class_images = list()
        true_class_boxes = list()
        true_class_difficulties = list()
        true_class_images1 = true_class_images1.unsqueeze(1)
        # true_class_boxes1=true_class_boxes1.unsqueeze(1)
        true_class_difficulties1 = true_class_difficulties1.unsqueeze(1)
        scale = 100 / 512
        for i in range(0, n_class_objects1):
            if ((true_class_boxes1[i][3] - true_class_boxes1[i][1]) > scale):
                true_class_images.append(true_class_images1[i])  # (n_class_detections)
                true_class_boxes.append(true_class_boxes1[i].unsqueeze(0))  # (n_class_detections, 4)
                true_class_difficulties.append(true_class_difficulties1[i].unsqueeze(0))
        true_class_images = torch.LongTensor(true_class_images).to(device)
        # true_class_images = torch.cat(true_class_images, dim=0)  # (n_detections)
        if true_class_images.size(0) == 0:
            # continue torch.Tensor()
            true_class_boxes = torch.Tensor()  # (n_detections, 4)
            true_class_difficulties = torch.Tensor()  # (n_detections)
        else:
            true_class_boxes = torch.cat(true_class_boxes, dim=0)  # (n_detections, 4)
            true_class_difficulties = torch.cat(true_class_difficulties, dim=0)  # (n_detections)

        # (n_class_detections)

        n_easy_class_objects = (true_class_difficulties==0).sum().item()  # ignore difficult objects

        # Keep track of which true objects with this class have already been 'detected'
        # So far, none
        true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).to(
            device)  # (n_class_objects)

        # Extract only detections with this class
        det_class_images1 = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes1 = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores1 = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections1 = det_class_boxes1.size(0)
        # wd=det_class_boxes1[:][1]-det_class_boxes1[:][0]
        # hd=det_class_boxes1[:][3]-det_class_boxes1[:][2]
        det_class_images = list()
        det_class_boxes = list()
        det_class_scores = list()
        det_class_images1 = det_class_images1.unsqueeze(1)
        # det_class_boxes1=det_class_boxes1.unsqueeze(1)
        # det_class_scores1=det_class_scores1.unsqueeze(1)
        for i in range(0, n_class_detections1):
            if ((det_class_boxes1[i][3] - det_class_boxes1[i][1] > scale)):
                det_class_images.append(det_class_images1[i])  # (n_class_detections)
                det_class_boxes.append(det_class_boxes1[i].unsqueeze(0))  # (n_class_detections, 4)
                det_class_scores.append(det_class_scores1[i].unsqueeze(0))
        det_class_images = torch.LongTensor(det_class_images).to(device)  # (n_detections)
        n_class_detections = det_class_images.size(0)
        if n_class_detections == 0:
            continue
        det_class_boxes = torch.cat(det_class_boxes, dim=0)  # (n_detections, 4)
        det_class_scores = torch.cat(det_class_scores, dim=0)  # (n_detections)

        # Sort detections in decreasing order of confidence/scores
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # In the order of decreasing scores, check if true or false positive
        true_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        false_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            # Find objects in the same image with this class, their difficulties, and whether they have been detected before
            object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
            object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
            # If no such object in this image, then the detection is a false positive
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            # Find maximum overlap of this detection with objects in this image of this class
            overlaps,overlaps1 = find_jaccard_overlap(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

            # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
            original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]
            # We need 'original_ind' to update 'true_class_boxes_detected'

            # If the maximum overlap is greater than the threshold of 0.5, it's a match
            if max_overlap.item() > 0.5:
                # If the object it matched with is 'difficult', ignore it
                if object_difficulties[ind] == 0:
                    # If this object has already not been detected, it's a true positive
                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                    # Otherwise, it's a false positive (since this object is already accounted for)
                    else:
                        false_positives[d] = 1
            # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
            else:
                false_positives[d] = 1

        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (
                cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)

        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        # pre[c-1]=precisions.cpu()
        average_precisions[c-1] = precisions.mean()  # c is in [1, n_classes - 1]
        precision[c-1] = cumul_precision[n_class_detections - 1]
        recall[c-1] = cumul_recall[n_class_detections - 1]
        true_positives[c-1]=cumul_true_positives[n_class_detections - 1]
        false_positives[c-1]=cumul_false_positives[n_class_detections - 1]

        ##me
        # if (c==1):
        # precision=cumul_precision.max()
        # recall=cumul_recall.max()
        # axarr = f.add_subplot(1, 1, 1)
        plt.plot(recall_thresholds, precisions.cpu(), color='gold', lw=2)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall' + str(c))
        plt.ylabel('Precision' + str(c))
        plt.show()

    # Calculate Mean Average Precision (mAP)
    mean_average_precision = average_precisions.mean().item()

    # Keep class-wise average precisions in a dictionary
    average_precisions = {rev_label_map[c+1]: v for c, v in enumerate(average_precisions.tolist())}
    precision = {rev_label_map[c+1]: v for c, v in enumerate(precision.tolist())}
    recall = {rev_label_map[c+1]: v for c, v in enumerate(recall.tolist())}

    return average_precisions, mean_average_precision, precision, recall, f ,n_easy_class_objects,true_positives,false_positives

def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties):
    """
    Calculate the Mean Average Precision (mAP) of detected objects.

    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation

    :param det_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
    :param det_labels: list of tensors, one tensor for each image containing detected objects' labels
    :param det_scores: list of tensors, one tensor for each image containing detected objects' labels' scores
    :param true_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
    :param true_labels: list of tensors, one tensor for each image containing actual objects' labels
    :param true_difficulties: list of tensors, one tensor for each image containing actual objects' difficulty (0 or 1)
    :return: list of average precisions for all classes, mean average precision (mAP)
    """
    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(
        true_labels) == len(
        true_difficulties)  # these are all lists of tensors of the same length, i.e. number of images
    n_classes = len(label_map)

    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(
        device)  # (n_objects), n_objects is the total no. of objects across all images
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
    true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    # Calculate APs for each class (except background)
    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    for c in range(1, n_classes):
        # Extract only objects with this class
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
        true_class_difficulties = true_difficulties[true_labels == c]  # (n_class_objects)
        n_easy_class_objects = (1 - true_class_difficulties).sum().item()  # ignore difficult objects

        # Keep track of which true objects with this class have already been 'detected'
        # So far, none
        true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).to(
            device)  # (n_class_objects)

        # Extract only detections with this class
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            continue

        # Sort detections in decreasing order of confidence/scores
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # In the order of decreasing scores, check if true or false positive
        true_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        false_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            # Find objects in the same image with this class, their difficulties, and whether they have been detected before
            object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
            object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
            # If no such object in this image, then the detection is a false positive
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            # Find maximum overlap of this detection with objects in this image of this class
            overlaps = find_jaccard_overlap(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

            # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
            original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]
            # We need 'original_ind' to update 'true_class_boxes_detected'

            # If the maximum overlap is greater than the threshold of 0.5, it's a match
            if max_overlap.item() > 0.15:
                # If the object it matched with is 'difficult', ignore it
                if object_difficulties[ind] == 0:
                    # If this object has already not been detected, it's a true positive
                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                    # Otherwise, it's a false positive (since this object is already accounted for)
                    else:
                        false_positives[d] = 1
            # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
            else:
                false_positives[d] = 1

        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (
                cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)

        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]
        ##me
        if (c==1):
            f = plt.figure()
            plt.plot(recall_thresholds, precisions, color='gold', lw=2)
            
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            # plt.show()

    # Calculate Mean Average Precision (mAP)
    mean_average_precision = average_precisions.mean().item()

    # Keep class-wise average precisions in a dictionary
    average_precisions = {rev_label_map[c]: v for c, v in enumerate(average_precisions.tolist())}

    return average_precisions, mean_average_precision, f


def xy_to_cxcy(xy):
    """
    Convert bounding boxes from boundary coordinates (x_min, y_min, x_max, y_max) to center-size coordinates (c_x, c_y, w, h).

    :param xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h


def cxcy_to_xy(cxcy):
    """
    Convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    Encode bounding boxes (that are in center-size form) w.r.t. the corresponding prior boxes (that are in center-size form).

    For the center coordinates, find the offset with respect to the prior box, and scale by the size of the prior box.
    For the size coordinates, scale by the size of the prior box, and convert to the log-space.

    In the model, we are predicting bounding box coordinates in this encoded form.

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding must be performed, a tensor of size (n_priors, 4)
    :return: encoded bounding boxes, a tensor of size (n_priors, 4)
    """

    # The 10 and 5 below are referred to as 'variances' in the original Caffe repo, completely empirical
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log((cxcy[:, 2:]) / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    Decode bounding box coordinates predicted by the model, since they are encoded in the form mentioned above.

    They are decoded into center-size coordinates.

    This is the inverse of the function above.

    :param gcxgcy: encoded bounding boxes, i.e. output of the model, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding is defined, a tensor of size (n_priors, 4)
    :return: decoded bounding boxes in center-size form, a tensor of size (n_priors, 4)
    """

    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h


def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union ,  intersection / areas_set_1# (n1, n2)


# Some augmentation functions below have been adapted from
# From https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

def expand(image, boxes, filler):
    """
    Perform a zooming out operation by placing the image in a larger canvas of filler material.

    Helps to learn to detect smaller objects.

    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param filler: RBG values of the filler material, a list like [R, G, B]
    :return: expanded image, updated bounding box coordinates
    """
    # Calculate dimensions of proposed expanded (zoomed-out) image
    original_h = image.size(1)
    original_w = image.size(2)
    max_scale = 4
    scale = random.uniform(1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)

    # Create such an image with the filler
    filler = torch.FloatTensor(filler)  # (3)
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)  # (3, new_h, new_w)
    # Note - do not use expand() like new_image = filler.unsqueeze(1).unsqueeze(1).expand(3, new_h, new_w)
    # because all expanded values will share the same memory, so changing one pixel will change all

    # Place the original image at random coordinates in this new image (origin at top-left of image)
    left = random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h
    new_image[:, top:bottom, left:right] = image

    # Adjust bounding boxes' coordinates accordingly

    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(
        0).float()  # (n_objects, 4), n_objects is the no. of objects in this image

    return new_image, new_boxes


def random_crop(image, boxes, labels, difficulties):
    """
    Performs a random crop in the manner stated in the paper. Helps to learn to detect larger and partial objects.

    Note that some objects may be cut out entirely.

    Adapted from https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
    :return: cropped image, updated bounding box coordinates, updated labels, updated difficulties
    """
    original_h = image.size(1)
    original_w = image.size(2)
    # Keep choosing a minimum overlap until a successful crop is made
    while True:
        # Randomly draw the value for minimum overlap
        min_overlap = random.choice([0., .1, .3, .5, .7, .9, None])  # 'None' refers to no cropping

        # If not cropping
        if min_overlap is None:
            return image, boxes, labels, difficulties

        # Try up to 50 times for this choice of minimum overlap
        # This isn't mentioned in the paper, of course, but 50 is chosen in paper authors' original Caffe repo
        max_trials = 50
        for _ in range(max_trials):
            # Crop dimensions must be in [0.3, 1] of original dimensions
            # Note - it's [0.1, 1] in the paper, but actually [0.3, 1] in the authors' repo
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)

            # Aspect ratio has to be in [0.5, 2]
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue

            # Crop coordinates (origin at top-left of image)
            left = random.randint(0, original_w - new_w)
            right = left + new_w
            top = random.randint(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([left, top, right, bottom])  # (4)

            # Calculate Jaccard overlap between the crop and the bounding boxes
            overlap,overlap1 = find_jaccard_overlap(crop.unsqueeze(0),
                                           boxes)  # (1, n_objects), n_objects is the no. of objects in this image
            overlap = overlap.squeeze(0)  # (n_objects)

            # If not a single bounding box has a Jaccard overlap of greater than the minimum, try again
            if overlap.max().item() < min_overlap:
                continue

            # Crop image
            new_image = image[:, top:bottom, left:right]  # (3, new_h, new_w)

            # Find centers of original bounding boxes
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.  # (n_objects, 2)

            # Find bounding boxes whose centers are in the crop
            centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (
                    bb_centers[:, 1] < bottom)  # (n_objects), a Torch uInt8/Byte tensor, can be used as a boolean index

            # If not a single bounding box has its center in the crop, try again
            if not centers_in_crop.any():
                continue

            # Discard bounding boxes that don't meet this criterion
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]
            new_difficulties = difficulties[centers_in_crop]

            # Calculate bounding boxes' new coordinates in the crop
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])  # crop[:2] is [left, top]
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])  # crop[2:] is [right, bottom]
            new_boxes[:, 2:] -= crop[:2]

            return new_image, new_boxes, new_labels, new_difficulties


def flip(image, boxes):
    """
    Flip image horizontally.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: flipped image, updated bounding box coordinates
    """
    # Flip image
    new_image = FT.hflip(image)

    # Flip boxes
    # new_boxes=torch.tensor()
    new_boxes = boxes[:, [0, 1, 2, 3]]
    new_boxes[:, 1] = boxes[:, 1]
    new_boxes[:, 3] = boxes[:, 3]
    new_boxes[:, 0] = image.width - boxes[:, 0]

    new_boxes[:, 2] = image.width - boxes[:, 2]
    # for i in range (0,len(boxes)):
    #     if ((image.width - boxes[i, 0] ) < 0):
    #         print("")
    #     if ((image.width - boxes[i, 2] ) < 0):
    #         print("")

    new_boxes = new_boxes[:, [2, 1, 0, 3]]
    # for i in range(0, len(new_boxes)):
    #     if ((new_boxes[i, 3] - new_boxes[i, 1]) <= 0):
    #         new_boxes[i, 3] = new_boxes[i, 1]+1
    #     if ((new_boxes[i, 2] - new_boxes[i, 0]) <= 0):
    #         new_boxes[i, 2] = new_boxes[i, 0]+1

    return new_image, new_boxes


def resize(image, boxes, dims=(513, 513), return_percent_coords=True):
    """
    Resize image. For the SSD300, resize to (300, 300).

    Since percent/fractional coordinates are calculated for the bounding boxes (w.r.t image dimensions) in this process,
    you may choose to retain them.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: resized image, updated bounding box coordinates (or fractional coordinates, in which case they remain the same)
    """
    # Resize image
    new_image = FT.resize(image, dims)

    # Resize bounding boxes
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims  # percent coordinates

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes


def photometric_distort(image):
    """
    Distort brightness, contrast, saturation, and hue, each with a 50% chance, in random order.

    :param image: image, a PIL Image
    :return: distorted image
    """
    new_image = image

    distortions = [FT.adjust_brightness,
                   FT.adjust_contrast,
                   FT.adjust_saturation,
                   FT.adjust_hue]

    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ is 'adjust_hue':
                # Caffe repo uses a 'hue_delta' of 18 - we divide by 255 because PyTorch needs a normalized value
                adjust_factor = random.uniform(-18 / 255., 18 / 255.)
            else:
                # Caffe repo uses 'lower' and 'upper' values of 0.5 and 1.5 for brightness, contrast, and saturation
                adjust_factor = random.uniform(0.5, 1.5)

            # Apply this distortion
            new_image = d(new_image, adjust_factor)

    return new_image


def transform(image, boxes, labels, difficulties, split):
    """
    Apply the transformations above.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
    :param split: one of 'TRAIN' or 'TEST', since different sets of transformations are applied
    :return: transformed image, transformed bounding box coordinates, transformed labels, transformed difficulties
    """
    assert split in {'SANITEST',"NSANITEST1","DSANITEST1","SANITEST1",'SANITRAIN','SANIVAL'}

    # Mean and standard deviation of ImageNet data that our base VGG from torchvision was trained on
    # see: https://pytorch.org/docs/stable/torchvision/models.html
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_image = image

    new_boxes = boxes
    new_labels = labels
    new_difficulties = difficulties
    # Skip the following operations if validation/evaluation
    if ( split =='SANITRAIN'):
        # A series of photometric distortions in random order, each with 50% chance of occurrence, as in Caffe repo
        new_image = photometric_distort(new_image)

        # Convert PIL image to Torch tensor
        new_image = FT.to_tensor(new_image)


        # Convert PIL image to Torch tensor


        # Expand image (zoom out) with a 50% chance - helpful for training detection of small objects
        # Fill surrounding space with the mean of ImageNet data that our base VGG was trained on
        if random.random() < 0.5:
            new_image, new_boxes = expand(new_image, boxes, filler=mean)
            # if ((new_boxes[:, 0] < 0).sum()) > 0:
            #     print("jj")
            # for i in range(0, len(new_boxes)):
            #     if (new_boxes[i, 2] == 0):
            #         new_boxes[i, 2] = 0
            #     if (new_boxes[i, 2] == 0):
            #         new_boxes[i, 3] = 0
        # Randomly crop image (zoom in)
        new_image, new_boxes, new_labels, new_difficulties = random_crop(new_image, new_boxes, new_labels,
                                                                         new_difficulties)
        new_boxes1 = new_boxes
        # if ((new_boxes[:, 0] < 0).sum()) > 0:
        #     print("jj")
        # Convert Torch tensor to PIL image
        new_image = FT.to_pil_image(new_image)

        # Flip image with a 50% chance
        if random.random() < 0.5:
            new_image, new_boxes = flip(new_image, new_boxes)
            # if ((new_boxes[:, 0] < 0).sum()) > 0:
            #     print("jj")
            # for i in range(0, len(new_boxes)):
            #     if (new_boxes[i, 2] == 0):
            #         new_boxes[i, 2] = 0
            #     if (new_boxes[i, 2] == 0):
            #         new_boxes[i, 3] = 0
    # Resize image to (300, 300) - this also converts absolute boundary coordinates to their fractional form
    new_image1, new_boxes = resize(new_image, new_boxes, dims=(513, 513))
    # if ((new_boxes[:,0]<0).sum())>0:
    #     print("jj")
    # for i in range(0, len(new_boxes)):
    #     if (new_boxes[i, 2] == 0):
    #         new_boxes[i, 2] = 0
    #     if (new_boxes[i, 2] == 0):
    #         new_boxes[i, 3] = 0
    # Convert PIL image to Torch tensor
    new_image = FT.to_tensor(new_image1)

    # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
    new_image = FT.normalize(new_image, mean=mean, std=std)
    # if ((new_boxes[:,0]<0).sum())>0:
    #     print("jj")
    # for i in range(0, len(new_boxes)):
    #     if (new_boxes[i, 2] == 0):
    #         new_boxes[i, 2] = 0
    #     if (new_boxes[i, 2] == 0):
    #         new_boxes[i, 3] = 0
    return new_image, new_boxes, new_labels, new_difficulties
def transform1(image, boxes, labels, difficulties, split):
    """
    Apply the transformations above.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
    :param split: one of 'TRAIN' or 'TEST', since different sets of transformations are applied
    :return: transformed image, transformed bounding box coordinates, transformed labels, transformed difficulties
    """
    assert split in {'SANITEST', "NSANITEST1", "DSANITEST1", "SANITEST1", 'SANITRAIN', 'SANIVAL'}



    # Mean and standard deviation of ImageNet data that our base VGG from torchvision was trained on
    # see: https://pytorch.org/docs/stable/torchvision/models.html
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_image = image
    # new_boxes = boxes* torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(
    #         0)
    new_boxes = boxes
    new_labels = labels
    new_difficulties = difficulties
    # Skip the following operations if validation/evaluation
    # if(split == 'TRAIN' or split=='COCOTRAIN'):
    #     # A series of photometric distortions in random order, each with 50% chance of occurrence, as in Caffe repo
    #     new_image = photometric_distort(new_image)
    #
    #     # Convert PIL image to Torch tensor
    #     new_image = FT.to_tensor(new_image)
    #
    #     # Expand image (zoom out) with a 50% chance - helpful for training detection of small objects
    #     # Fill surrounding space with the mean of ImageNet data that our base VGG was trained on
    #     # if len(boxes)!=0:
    #     if random.random() < 0.5:
    #         new_image, new_boxes = expand(new_image, boxes, filler=mean)
    #
    #     # Randomly crop image (zoom in)
    #     new_image, new_boxes, new_labels, new_difficulties = random_crop(new_image, new_boxes, new_labels,
    #                                                                      new_difficulties)
    #
    #     # Convert Torch tensor to PIL image
    #     new_image = FT.to_pil_image(new_image)
    #
    #     # Flip image with a 50% chance
    #     if random.random() < 0.5:
    #         new_image, new_boxes = flip(new_image, new_boxes)

    # Resize image to (300, 300) - this also converts absolute boundary coordinates to their fractional form
    new_image, new_boxes = resize(new_image, new_boxes)
    # new_image, new_boxes = resize512(new_image, new_boxes, dims=(513, 513))
    # Convert PIL image to Torch tensor
    new_image = FT.to_tensor(new_image)

    # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
    new_image = FT.normalize(new_image, mean=mean, std=std)

    return new_image, new_boxes, new_labels, new_difficulties

def transform2(image, boxes, labels, difficulties, split):
    """
    Apply the transformations above.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
    :param split: one of 'TRAIN' or 'TEST', since different sets of transformations are applied
    :return: transformed image, transformed bounding box coordinates, transformed labels, transformed difficulties
    """
    assert split in {'SANITEST', "NSANITEST1", "DSANITEST1", "SANITEST1", 'SANITRAIN', 'SANIVAL'}

    # Mean and standard deviation of ImageNet data that our base VGG from torchvision was trained on
    # see: https://pytorch.org/docs/stable/torchvision/models.html
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_image = image

    new_boxes = boxes
    new_labels = labels
    new_difficulties = difficulties
    # Skip the following operations if validation/evaluation
    if (split == 'SANITRAIN'):
        # A series of photometric distortions in random order, each with 50% chance of occurrence, as in Caffe repo
        new_image = photometric_distort(new_image)
        new_image = FT.to_tensor(new_image)
        if random.random() < 0.5:
            new_image, new_boxes = expand(new_image, boxes, filler=mean)

        # Randomly crop image (zoom in)
        new_image, new_boxes, new_labels, new_difficulties = random_crop(new_image, new_boxes, new_labels,
                                                                         new_difficulties)
        # Convert PIL image to Torch tensor
        new_image = FT.to_tensor(new_image)
        new_image = FT.to_pil_image(new_image)
        if random.random() < 0.5:
            new_image, new_boxes = flip(new_image, new_boxes)

        # # Expand image (zoom out) with a 50% chance - helpful for training detection of small objects
        # # Fill surrounding space with the mean of ImageNet data that our base VGG was trained on
        # # if len(boxes)!=0:
        # if random.random() < 0.5:
        #     new_image, new_boxes = expand(new_image, boxes, filler=mean)
        #
        # # Randomly crop image (zoom in)
        # new_image, new_boxes, new_labels, new_difficulties = random_crop(new_image, new_boxes, new_labels,
        #                                                                  new_difficulties)
        #
        # # Convert Torch tensor to PIL image
        # new_image = FT.to_pil_image(new_image)
        #
        # # Flip image with a 50% chance
        # if random.random() < 0.5:
        #     new_image, new_boxes = flip(new_image, new_boxes)

    # Resize image to (300, 300) - this also converts absolute boundary coordinates to their fractional form
    new_image, new_boxes = resize(new_image, new_boxes)
    # new_image, new_boxes = resize512(new_image, new_boxes, dims=(513, 513))
    # Convert PIL image to Torch tensor
    new_image = FT.to_tensor(new_image)

    # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
    new_image = FT.normalize(new_image, mean=mean, std=std)

    return new_image, new_boxes, new_labels, new_difficulties
def transform3(image, boxes, labels, difficulties, split):
    """
    Apply the transformations above.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
    :param split: one of 'TRAIN' or 'TEST', since different sets of transformations are applied
    :return: transformed image, transformed bounding box coordinates, transformed labels, transformed difficulties
    """
    assert split in {'SANITEST', "NSANITEST1", "DSANITEST1", "SANITEST1", 'SANITRAIN', 'SANIVAL'}

    # Mean and standard deviation of ImageNet data that our base VGG from torchvision was trained on
    # see: https://pytorch.org/docs/stable/torchvision/models.html
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_image = image

    new_boxes = boxes
    new_labels = labels
    new_difficulties = difficulties
    # Skip the following operations if validation/evaluation
    if (split == 'SANITRAIN'):
        # A series of photometric distortions in random order, each with 50% chance of occurrence, as in Caffe repo
        # new_image = photometric_distort(new_image)

        # Convert PIL image to Torch tensor
        new_image = FT.to_tensor(new_image)

        # Expand image (zoom out) with a 50% chance - helpful for training detection of small objects
        # Fill surrounding space with the mean of ImageNet data that our base VGG was trained on
        # if len(boxes)!=0:
        # if random.random() < 0.5:
        new_image, new_boxes = expand(new_image, boxes, filler=mean)

        # Randomly crop image (zoom in)
        # new_image, new_boxes, new_labels, new_difficulties = random_crop(new_image, new_boxes, new_labels,
        #                                                                  new_difficulties)

        # Convert Torch tensor to PIL image
        new_image = FT.to_pil_image(new_image)

        # Flip image with a 50% chance
        if random.random() < 0.5:
            new_image, new_boxes = flip(new_image, new_boxes)

    # Resize image to (300, 300) - this also converts absolute boundary coordinates to their fractional form
    new_image, new_boxes = resize(new_image, new_boxes)
    # new_image, new_boxes = resize512(new_image, new_boxes, dims=(513, 513))
    # Convert PIL image to Torch tensor
    new_image = FT.to_tensor(new_image)

    # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
    new_image = FT.normalize(new_image, mean=mean, std=std)

    return new_image, new_boxes, new_labels, new_difficulties
def transform4(image, boxes, labels, difficulties, split):
    """
    Apply the transformations above.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
    :param split: one of 'TRAIN' or 'TEST', since different sets of transformations are applied
    :return: transformed image, transformed bounding box coordinates, transformed labels, transformed difficulties
    """
    assert split in {'SANITEST', "NSANITEST1", "DSANITEST1", "SANITEST1", 'SANITRAIN', 'SANIVAL'}

    # Mean and standard deviation of ImageNet data that our base VGG from torchvision was trained on
    # see: https://pytorch.org/docs/stable/torchvision/models.html
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_image = image

    new_boxes = boxes
    new_labels = labels
    new_difficulties = difficulties
    # Skip the following operations if validation/evaluation
    if (split == 'SANITRAIN'):
        # A series of photometric distortions in random order, each with 50% chance of occurrence, as in Caffe repo
        # new_image = photometric_distort(new_image)

        # Convert PIL image to Torch tensor
        new_image = FT.to_tensor(new_image)

        # Expand image (zoom out) with a 50% chance - helpful for training detection of small objects
        # Fill surrounding space with the mean of ImageNet data that our base VGG was trained on
        # if len(boxes)!=0:
        # if random.random() < 0.5:
        #     new_image, new_boxes = expand(new_image, boxes, filler=mean)

        # Randomly crop image (zoom in)
        new_image, new_boxes, new_labels, new_difficulties = random_crop(new_image, new_boxes, new_labels,
                                                                         new_difficulties)

        # Convert Torch tensor to PIL image
        new_image = FT.to_pil_image(new_image)

        # Flip image with a 50% chance
        if random.random() < 0.5:
            new_image, new_boxes = flip(new_image, new_boxes)

    # Resize image to (300, 300) - this also converts absolute boundary coordinates to their fractional form
    new_image, new_boxes = resize(new_image, new_boxes)
    # new_image, new_boxes = resize512(new_image, new_boxes, dims=(513, 513))
    # Convert PIL image to Torch tensor
    new_image = FT.to_tensor(new_image)

    # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
    new_image = FT.normalize(new_image, mean=mean, std=std)

    return new_image, new_boxes, new_labels, new_difficulties


def adjust_learning_rate(optimizer, scale):
    """
    Scale learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param scale: factor to multiply learning rate with.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale
    print("DECAYING learning rate.\n The new LR is %f\n" % (optimizer.param_groups[1]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)

def calculate_mAP9(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties):
    """
    Calculate the Mean Average Precision (mAP) of detected objects.

    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation

    :param det_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
    :param det_labels: list of tensors, one tensor for each image containing detected objects' labels
    :param det_scores: list of tensors, one tensor for each image containing detected objects' labels' scores
    :param true_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
    :param true_labels: list of tensors, one tensor for each image containing actual objects' labels
    :param true_difficulties: list of tensors, one tensor for each image containing actual objects' difficulty (0 or 1)
    :return: list of average precisions for all classes, mean average precision (mAP)
    """
    global n_easy_class_objects
    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(
        true_labels) == len(
        true_difficulties)  # these are all lists of tensors of the same length, i.e. number of images
    n_classes = len(label_map)
    bir1=list()
    # for t in range(bir.__len__()):
    #     b=np.expand_dims(np.ascontiguousarray(bir[t]),axis=0)
    #     if t==0:
    #         bir1=b
    #     else:
    #         bir1=np.append(bir1,b,axis=0)
    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(
        device)  # (n_objects), n_objects is the total no. of objects across all images
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
    true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    # Calculate APs for each class (except background)
    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    precision = torch.zeros((n_classes - 1), dtype=torch.float)
    recall = torch.zeros((n_classes - 1), dtype=torch.float)
    true_positives = torch.zeros((n_classes - 1), dtype=torch.float)
    false_positives = torch.zeros((n_classes - 1), dtype=torch.float)
    # pre=torch.zeros((n_classes - 1), dtype=torch.float)
    f = plt.figure()

    for c in range(1, n_classes):
        # Extract only objects with this class
        true_class_images1 = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes1 = true_boxes[true_labels == c]  # (n_class_objects, 4)
        true_class_difficulties1 = true_difficulties[true_labels == c]  # (n_class_objects)
        n_class_objects1 = true_class_boxes1.size(0)

        true_class_images5 = list()
        true_class_boxes5 = list()
        true_class_difficulties5 = list()
        true_class_images1 = true_class_images1.unsqueeze(1)
        # true_class_boxes1=true_class_boxes1.unsqueeze(1)
        true_class_difficulties1 = true_class_difficulties1.unsqueeze(1)
        scale1 = 0/512
        scale = 100 / 512
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(0, n_class_objects1):
            if ((true_class_boxes1[i][3] - true_class_boxes1[i][1]) < scale):
                true_class_images5.append(true_class_images1[i])  # (n_class_detections)
                true_class_boxes5.append(true_class_boxes1[i].unsqueeze(0))  # (n_class_detections, 4)
                true_class_difficulties5.append(true_class_difficulties1[i].unsqueeze(0))
        true_class_images5 = torch.LongTensor(true_class_images5).to(device)
        # true_class_images = torch.cat(true_class_images, dim=0)  # (n_detections)
        if true_class_images5.size(0) == 0:
            # continue torch.Tensor()
            true_class_boxes5 = torch.Tensor()  # (n_detections, 4)
            true_class_difficulties5 = torch.Tensor()  # (n_detections)
        else:
            true_class_boxes5 = torch.cat(true_class_boxes5, dim=0)  # (n_detections, 4)
            true_class_difficulties5 = torch.cat(true_class_difficulties5, dim=0)  # (n_detections)
        # wt=true_class_boxes1[:][1]-true_class_boxes1[:][0]
        # ht=true_class_boxes1[:][3]-true_class_boxes1[:][2]
        true_class_images = list()
        true_class_boxes = list()
        true_class_difficulties = list()
        # true_class_images1 = true_class_images1.unsqueeze(1)
        # # true_class_boxes1=true_class_boxes1.unsqueeze(1)
        # true_class_difficulties1 = true_class_difficulties1.unsqueeze(1)

        for i in range(0, n_class_objects1):
            if ((true_class_boxes1[i][3] - true_class_boxes1[i][1]) >= scale):
                true_class_images.append(true_class_images1[i])  # (n_class_detections)
                true_class_boxes.append(true_class_boxes1[i].unsqueeze(0))  # (n_class_detections, 4)
                true_class_difficulties.append(true_class_difficulties1[i].unsqueeze(0))
        true_class_images = torch.LongTensor(true_class_images).to(device)
        # true_class_images = torch.cat(true_class_images, dim=0)  # (n_detections)
        if true_class_images.size(0) == 0:
            # continue torch.Tensor()
            true_class_boxes = torch.Tensor()  # (n_detections, 4)
            true_class_difficulties = torch.Tensor()  # (n_detections)
        else:
            true_class_boxes = torch.cat(true_class_boxes, dim=0)  # (n_detections, 4)
            true_class_difficulties = torch.cat(true_class_difficulties, dim=0)  # (n_detections)

        # (n_class_detections)

        n_easy_class_objects = (true_class_difficulties==0).sum().item()  # ignore difficult objects
        # n_easy_class_objects = (len(true_class_difficulties))
        # Keep track of which true objects with this class have already been 'detected'
        # So far, none
        true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).to(
            device)  # (n_class_objects)
        true_class_boxes_detected5 = torch.zeros((true_class_difficulties5.size(0)), dtype=torch.uint8).to(
            device)
        # Extract only detections with this class
        det_class_images1 = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes1 = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores1 = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections1 = det_class_boxes1.size(0)
        # wd=det_class_boxes1[:][1]-det_class_boxes1[:][0]
        # hd=det_class_boxes1[:][3]-det_class_boxes1[:][2]

        det_class_images = list()
        det_class_boxes = list()
        det_class_scores = list()
        det_class_images1 = det_class_images1.unsqueeze(1)
        # det_class_boxes1=det_class_boxes1.unsqueeze(1)
        # det_class_scores1=det_class_scores1.unsqueeze(1)
        for i in range(0, n_class_detections1):
            if ((det_class_boxes1[i][3] - det_class_boxes1[i][1] >= scale)):
                det_class_images.append(det_class_images1[i])  # (n_class_detections)
                det_class_boxes.append(det_class_boxes1[i].unsqueeze(0))  # (n_class_detections, 4)
                det_class_scores.append(det_class_scores1[i].unsqueeze(0))
        det_class_images = torch.LongTensor(det_class_images).to(device)  # (n_detections)
        n_class_detections = det_class_images.size(0)

        det_class_images2 = list()
        det_class_boxes2 = list()
        det_class_scores2 = list()
        # det_class_images1 = det_class_images1.unsqueeze(1)
        # det_class_boxes1=det_class_boxes1.unsqueeze(1)
        # det_class_scores1=det_class_scores1.unsqueeze(1)
        for i in range(0, n_class_detections1):
            if ((det_class_boxes1[i][3] - det_class_boxes1[i][1] < scale)):
                det_class_images2.append(det_class_images1[i])  # (n_class_detections)
                det_class_boxes2.append(det_class_boxes1[i].unsqueeze(0))  # (n_class_detections, 4)
                det_class_scores2.append(det_class_scores1[i].unsqueeze(0))
        det_class_images2 = torch.LongTensor(det_class_images2).to(device)  # (n_detections)
        n_class_detections2 = det_class_images2.size(0)
        true_positives = torch.zeros((n_class_detections1), dtype=torch.float).to(device)  # (n_class_detections)
        false_positives = torch.zeros((n_class_detections1), dtype=torch.float).to(device)  # (n_class_detections)
        if n_class_detections == 0 & n_class_detections2 == 0:
            continue



        det_class_boxes2 = torch.cat(det_class_boxes2, dim=0)  # (n_detections, 4)
        det_class_scores2 = torch.cat(det_class_scores2, dim=0)  # (n_detections)

        # Sort detections in decreasing order of confidence/scores
        det_class_scores2, sort_ind2 = torch.sort(det_class_scores2, dim=0, descending=True)  # (n_class_detections)
        det_class_images2 = det_class_images2[sort_ind2]  # (n_class_detections)
        det_class_boxes2 = det_class_boxes2[sort_ind2]  # (n_class_detections, 4)
        # In the order of decreasing scores, check if true or false positive
        if n_class_detections == 0 & n_class_detections2 != 0:
            det_scores2 = str(round(det_class_scores2[d - n_class_detections].to('cpu').tolist(), 2))
            this_detection_box6 = det_class_boxes2[d - n_class_detections].unsqueeze(0)  # (1, 4)
            this_image6 = det_class_images2[d - n_class_detections]  # (), scalar

            # Find objects in the same image with this class, their difficulties, and whether they have been detected before
            object_boxes6 = true_class_boxes[true_class_images == this_image6]  # (n_class_objects_in_img)
            object_difficulties = true_class_difficulties[
                true_class_images == this_image6]  # (n_class_objects_in_img)
            # If no such object in this image, then the detection is a false positive

            if object_boxes6.size(0) == 0:
                continue

            overlaps6, overlaps61 = find_jaccard_overlap(this_detection_box6,
                                                         object_boxes6)  # (1, n_class_objects_in_img)
            max_overlap6, ind6 = torch.max(overlaps6.squeeze(0), dim=0)  # (), () - scalars
            original_ind6 = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image6][
                ind6]

            # if object_boxes3.size(0) != 0:
            # Find maximum overlap of this detection with objects in this image of this class

            # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
            # We need 'original_ind' to update 'true_class_boxes_detected'

            # If the maximum overlap is greater than the threshold of 0.5, it's a match
            if ((max_overlap6.item() > 0.5)):
                # If the object it matched with is 'difficult', ignore it
                if object_difficulties[ind6] == 0:
                    # If this object has already not been detected, it's a true positive

                    if true_class_boxes_detected[original_ind6] == 0:
                        true_positives[d] = 1
                        # box_location1 = this_detection_box6[0].tolist()
                        # bir1[this_image6] = cv2.rectangle(np.squeeze(bir1[this_image6], axis=0),
                        #                                  (int(box_location1[0]), int(box_location1[1])),
                        #                                  (int(box_location1[2]), int(box_location1[3])), (255, 0, 0),
                        #                                  2)
                        # # bir1[this_image6] = cv2.putText(bir1[this_image6], det_scores,
                        #                                org=(int(box_location1[0]) + 2, int(box_location1[1]) - 4),
                        #                                fontFace=font, fontScale=1 / 2, color=(255, 0, 0), thickness=2,
                        #                                lineType=cv2.LINE_AA)
                        # this_detection_box1 = this_detection_box.cpu() * original_dims.cpu()

                        true_class_boxes_detected[original_ind6] = 1
            else:
                if ((object_boxes6[ind6][2] - object_boxes6[ind6][0]) / (
                        object_boxes6[ind6][3] - object_boxes6[ind6][1]) >= 0.7):
                    if (max_overlap6 >= 0.1):
                        # if object_difficulties2[ind3] == 0:
                        if object_difficulties[ind6] == 0:
                            true_positives[d] = 1
        #
        #
        #     # Sort detections in decreasing order of confidence/scores

            # for d in range(n_class_detections,n_class_detections2):
            #     det_scores2 = str(round(det_class_scores2[d-n_class_detections].to('cpu').tolist(), 2))
            #     this_detection_box2 = det_class_boxes2[d-n_class_detections].unsqueeze(0)  # (1, 4)
            #     this_image2 = det_class_images2[d-n_class_detections]  # (), scalar
            #
            #     # Find objects in the same image with this class, their difficulties, and whether they have been detected before
            #     object_boxes2 = true_class_boxes[true_class_images == this_image2]  # (n_class_objects_in_img)
            #     # object_difficulties2 = true_class_difficulties[
            #     #     true_class_images == this_image2]  # (n_class_objects_in_img)
            #     # If no such object in this image, then the detection is a false positive
            #
            #     if object_boxes2.size(0) == 0:
            #         continue
            #
            #     overlaps3, overlaps31 = find_jaccard_overlap(this_detection_box2,
            #                                                  object_boxes2)  # (1, n_class_objects_in_img)
            #     max_overlap3, ind3 = torch.max(overlaps3.squeeze(0), dim=0)  # (), () - scalars
            #     original_ind3 = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image2][
            #         ind3]
            #
            #     # if object_boxes3.size(0) != 0:
            #         # Find maximum overlap of this detection with objects in this image of this class
            #
            #         # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            #         # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
            #         # We need 'original_ind' to update 'true_class_boxes_detected'
            #
            #         # If the maximum overlap is greater than the threshold of 0.5, it's a match
            #     if ((max_overlap3.item() > 0.5)):
            #         # If the object it matched with is 'difficult', ignore it
            #         # if object_difficulties2[ind3] == 0:
            #         # If this object has already not been detected, it's a true positive
            #         if true_class_boxes_detected[original_ind3] == 0:
            #             true_positives[d] = 1
            #             # this_detection_box1 = this_detection_box.cpu() * original_dims.cpu()
            #
            #             true_class_boxes_detected[original_ind3] = 1
            #     else:
            #         if ((object_boxes2[ind3][2] - object_boxes2[ind3][0]) / (
            #             object_boxes2[ind3][3] - object_boxes2[ind3][1]) >= 0.7):
            #             if (max_overlap3 >= 0.1):
            #                 # if object_difficulties2[ind3] == 0:
            #                 true_positives[d] = 1
            # continue
        s1=int(55/512)
        det_class_boxes = torch.cat(det_class_boxes, dim=0)  # (n_detections, 4)
        det_class_scores = torch.cat(det_class_scores, dim=0)  # (n_detections)

        # Sort detections in decreasing order of confidence/scores
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)
        for d in range(n_class_detections1):
            if d<n_class_detections:
                this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
                this_image = det_class_images[d]  # (), scalar

                # Find objects in the same image with this class, their difficulties, and whether they have been detected before
                object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
                object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)

                object_boxes5 = true_class_boxes5[true_class_images5 == this_image]  # (n_class_objects_in_img)
                object_difficulties5 = true_class_difficulties5[true_class_images5 == this_image]
                # If no such object in this image, then the detection is a false positive
                if object_boxes.size(0) == 0:
                    if object_boxes5.size(0) == 0:
                        false_positives[d] = 1
                        continue

                # Find maximum overlap of this detection with objects in this image of this class
                if object_boxes5.size(0) != 0:
                    overlaps5, overlaps51 = find_jaccard_overlap(this_detection_box, object_boxes5)  # (1, n_class_objects_in_img)
                    max_overlap5, ind5 = torch.max(overlaps5.squeeze(0), dim=0)  # (), () - scalars
                    # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
                    # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
                    # We need 'original_ind' to update 'true_class_boxes_detected'
                    original_ind5 = torch.LongTensor(range(true_class_boxes5.size(0)))[true_class_images5== this_image][ind5]
                if object_boxes.size(0) != 0:
                    overlaps, overlaps1 = find_jaccard_overlap(this_detection_box,
                                                               object_boxes)  # (1, n_class_objects_in_img)
                    max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars
                    original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]
                    if ((max_overlap.item() > 0.5) or ((object_boxes[ind][3]-object_boxes[ind][1])<s1 & (max_overlap.item() > 0.1))):
                        # If the object it matched with is 'difficult', ignore it
                        if object_difficulties[ind] == 0:
                            # If this object has already not been detected, it's a true positive
                            if true_class_boxes_detected[original_ind] == 0:
                                true_positives[d] = 1
                                # box_location1 = this_detection_box[0].tolist()
                                # bir1[this_image] = cv2.rectangle(np.squeeze(bir1[this_image], axis=0), (int(box_location1[0]), int(box_location1[1])),
                                #                       (int(box_location1[2]), int(box_location1[3])), (255, 0, 0),
                                #                       2)
                                # bir1[this_image] = cv2.putText(bir1[this_image], det_scores,
                                #                     org=(int(box_location1[0]) + 2, int(box_location1[1]) - 4),
                                #                     fontFace=font, fontScale=1 / 2, color=(255, 0, 0), thickness=2,
                                #                     lineType=cv2.LINE_AA)
                                true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                            # Otherwise, it's a false positive (since this object is already accounted for)
                            else:
                                false_positives[d] = 1
                    # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
                    else:
                        if((object_boxes[ind][2]-object_boxes[ind][0])/(object_boxes[ind][3]-object_boxes[ind][1])>=0.7) :
                            if (max_overlap>=0.1):
                                if object_difficulties[ind] == 0:
                                    true_positives[d] = 1
                                # box_location1 = this_detection_box[0].tolist()
                                # bir1[this_image] = cv2.rectangle(np.squeeze(bir1[this_image], axis=0),
                                #                                  (int(box_location1[0]), int(box_location1[1])),
                                #                                  (int(box_location1[2]), int(box_location1[3])),
                                #                                  (255, 0, 0),
                                #                                  2)
                                # bir1[this_image] = cv2.putText(bir1[this_image], det_scores,
                                #                                org=(
                                #                                int(box_location1[0]) + 2, int(box_location1[1]) - 4),
                                #                                fontFace=font, fontScale=1 / 2, color=(255, 0, 0),
                                #                                thickness=2,
                                #                                lineType=cv2.LINE_AA)
                            else:
                                if object_boxes5.size(0) != 0:
                                    if ((max_overlap5.item() > 0.5) or (
                                            (object_boxes5[ind5][3] - object_boxes5[ind5][1]) < s1 & (
                                            max_overlap5.item() > 0.1))):
                                        # If the object it matched with is 'difficult', ignore it
                                        # if object_difficulties5[ind5] == 0:
                                        # If this object has already not been detected, it's a true positive
                                        if object_difficulties5[ind5] == 0:
                                            if true_class_boxes_detected5[original_ind5] == 0:
                                                # true_positives[d] = 1
                                                true_class_boxes_detected5[
                                                    original_ind5] = 1  # this object has now been detected/accounted for
                                            # Otherwise, it's a false positive (since this object is already accounted for)
                                            else:
                                                false_positives[d] = 1
                                    else:
                                        if ((object_boxes5[ind5][2] - object_boxes5[ind5][0]) / (
                                                object_boxes5[ind5][3] - object_boxes5[ind5][1]) >= 0.7):
                                            if object_difficulties5[ind5] == 0:
                                                if (max_overlap5 >= 0.1):

                                                    print("")
                                                else:
                                                    false_positives[d] = 1
                                        else:
                                            false_positives[d] = 1
                                else:
                                    false_positives[d] = 1
                        else:
                            if object_boxes5.size(0) != 0:
                                if ((max_overlap5.item() > 0.5) or ((object_boxes5[ind5][3]-object_boxes5[ind5][1])<s1 & (max_overlap5.item() > 0.1))):
                                    # If the object it matched with is 'difficult', ignore it
                                    if object_difficulties5[ind5] == 0:
                                        # If this object has already not been detected, it's a true positive
                                        if true_class_boxes_detected5[original_ind5] == 0:
                                            # true_positives[d] = 1
                                            true_class_boxes_detected5[
                                                original_ind5] = 1  # this object has now been detected/accounted for
                                        # Otherwise, it's a false positive (since this object is already accounted for)
                                        else:
                                            false_positives[d] = 1
                                else:
                                    if ((object_boxes5[ind5][2] - object_boxes5[ind5][0]) / (object_boxes5[ind5][3] - object_boxes5[ind5][1])>=0.7):
                                        if object_difficulties5[ind5] == 0:
                                            if (max_overlap5 >= 0.1):
                                                print("")
                                            else:
                                                false_positives[d] = 1
                                    else:
                                        false_positives[d] = 1
                            else:
                                false_positives[d] = 1
                else:
                    if object_boxes5.size(0) != 0:
                        if ((max_overlap5.item() > 0.5) or (
                                (object_boxes5[ind5][3] - object_boxes5[ind5][1]) < s1 & (max_overlap5.item() > 0.1))):
                            # If the object it matched with is 'difficult', ignore it
                            if object_difficulties5[ind5] == 0:
                            # If this object has already not been detected, it's a true positive
                                if true_class_boxes_detected5[original_ind5] == 0:
                                    # true_positives[d] = 1
                                    true_class_boxes_detected5[
                                        original_ind5] = 1  # this object has now been detected/accounted for
                                # Otherwise, it's a false positive (since this object is already accounted for)
                                else:
                                    false_positives[d] = 1
                        else:
                            if ((object_boxes5[ind5][2] - object_boxes5[ind5][0]) / (
                                    object_boxes5[ind5][3] - object_boxes5[ind5][1]) >= 0.7):
                                if object_difficulties5[ind5] == 0:
                                    if (max_overlap5 >= 0.1):
                                        print("")
                                    else:
                                        false_positives[d] = 1
                            else:
                                false_positives[d] = 1
                    else:
                        false_positives[d] = 1
            else:
                # for d in range(n_class_detections2):
                det_scores2 = str(round(det_class_scores2[d-n_class_detections].to('cpu').tolist(), 2))
                this_detection_box6 = det_class_boxes2[d-n_class_detections].unsqueeze(0)  # (1, 4)
                this_image6 = det_class_images2[d-n_class_detections]  # (), scalar

                # Find objects in the same image with this class, their difficulties, and whether they have been detected before
                object_boxes6 = true_class_boxes[true_class_images == this_image6]  # (n_class_objects_in_img)
                object_difficulties = true_class_difficulties[
                    true_class_images == this_image6]  # (n_class_objects_in_img)
                # If no such object in this image, then the detection is a false positive

                if object_boxes6.size(0) == 0:
                    continue

                overlaps6, overlaps61 = find_jaccard_overlap(this_detection_box6,
                                                             object_boxes6)  # (1, n_class_objects_in_img)
                max_overlap6, ind6 = torch.max(overlaps6.squeeze(0), dim=0)  # (), () - scalars
                original_ind6 = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image6][
                    ind6]

                # if object_boxes3.size(0) != 0:
                # Find maximum overlap of this detection with objects in this image of this class

                # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
                # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
                # We need 'original_ind' to update 'true_class_boxes_detected'

                # If the maximum overlap is greater than the threshold of 0.5, it's a match
                if ((max_overlap6.item() > 0.5)):
                    # If the object it matched with is 'difficult', ignore it
                    if object_difficulties[ind6] == 0:
                    # If this object has already not been detected, it's a true positive

                        if true_class_boxes_detected[original_ind6] == 0:
                            true_positives[d] = 1
                            # box_location1 = this_detection_box6[0].tolist()
                            # bir1[this_image6] = cv2.rectangle(np.squeeze(bir1[this_image6], axis=0),
                            #                                  (int(box_location1[0]), int(box_location1[1])),
                            #                                  (int(box_location1[2]), int(box_location1[3])), (255, 0, 0),
                            #                                  2)
                            # # bir1[this_image6] = cv2.putText(bir1[this_image6], det_scores,
                            #                                org=(int(box_location1[0]) + 2, int(box_location1[1]) - 4),
                            #                                fontFace=font, fontScale=1 / 2, color=(255, 0, 0), thickness=2,
                            #                                lineType=cv2.LINE_AA)
                            # this_detection_box1 = this_detection_box.cpu() * original_dims.cpu()

                            true_class_boxes_detected[original_ind6] = 1
                else:
                    if ((object_boxes6[ind6][2] - object_boxes6[ind6][0]) / (
                            object_boxes6[ind6][3] - object_boxes6[ind6][1]) >= 0.7):
                        if (max_overlap6 >= 0.1):
                            # if object_difficulties2[ind3] == 0:
                            if object_difficulties[ind6] == 0:
                                true_positives[d] = 1
                            # box_location1 = this_detection_box6[0].tolist()
                            # bir1[this_image6] = cv2.rectangle(np.squeeze(bir1[this_image6], axis=0),
                            #                                   (int(box_location1[0]), int(box_location1[1])),
                            #                                   (int(box_location1[2]), int(box_location1[3])),
                            #                                   (255, 0, 0),
                            #                                   2)
                            # bir1[this_image6] = cv2.putText(bir1[this_image6], det_scores,
                            #                                 org=(int(box_location1[0]) + 2, int(box_location1[1]) - 4),
                            #                                 fontFace=font, fontScale=1 / 2, color=(255, 0, 0),
                            #                                 thickness=2,
                            #                                 lineType=cv2.LINE_AA)
            # If the maximum overlap is greater than the threshold of 0.5, it's a match
        # if n_class_detections2 !=0 :
        # for d in range(true_class_images.size(0)):
        #     # det_scores2 = str(round(det_class_scores2[d].to('cpu').tolist(), 2))
        #     # this_detection_box6 = det_class_boxes2[d].unsqueeze(0)  # (1, 4)
        #     # this_image6 = det_class_images2[d]  # (), scalar
        #     # det_class_boxes2=torch.stack(det_class_boxes2,dim=0)
        #     # Find objects in the same image with this class, their difficulties, and whether they have been detected before
        #     object_boxes6 = det_class_boxes2[true_class_images[d] == det_class_images2]  # (n_class_objects_in_img)
        #     # object_difficulties2 = true_class_difficulties[
        #     #     true_class_images == this_image2]  # (n_class_objects_in_img)
        #     # If no such object in this image, then the detection is a false positive
        #
        #     if object_boxes6.size(0) == 0:
        #         continue
        #
        #     overlaps6, overlaps61 = find_jaccard_overlap(true_class_boxes[d].unsqueeze(0),
        #                                                  object_boxes6)  # (1, n_class_objects_in_img)
        #     max_overlap6, ind6 = torch.max(overlaps6.squeeze(0), dim=0)  # (), () - scalars
        #     original_ind6 = torch.LongTensor(range(true_class_boxes.size(0)))[d]
        #
        #         # if object_boxes3.size(0) != 0:
        #         # Find maximum overlap of this detection with objects in this image of this class
        #
        #         # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
        #         # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
        #         # We need 'original_ind' to update 'true_class_boxes_detected'
        #
        #         # If the maximum overlap is greater than the threshold of 0.5, it's a match
        #     if ((max_overlap6.item() > 0.5)):
        #         # If the object it matched with is 'difficult', ignore it
        #         # if object_difficulties2[ind3] == 0:
        #         # If this object has already not been detected, it's a true positive
        #         if true_class_boxes_detected[original_ind6] == 0:
        #             true_positives[d+n_class_detections] = 1
        #             # this_detection_box1 = this_detection_box.cpu() * original_dims.cpu()
        #
        #             true_class_boxes_detected[original_ind6] = 1
        #     else:
        #         if ((true_class_boxes[d][2] - true_class_boxes[d][0]) / (
        #                 true_class_boxes[d][3] - true_class_boxes[d][1]) >= 0.7):
        #             if (max_overlap6 >= 0.1):
        #                 # if object_difficulties2[ind3] == 0:
        #                 true_positives[d+n_class_detections] = 1


                # false_positives[d] = 1
        # for y in range(bir.__len__()):
        #     cv2.imwrite('/home/fereshteh/result2/' + str(y) + '.png', bir1[y])
        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (
                cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)
        print(true_positives)
        print(cumul_true_positives)
        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        # pre[c-1]=precisions.cpu()
        average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]
        precision[c - 1] =torch.sum(true_positives, dim=0)/ (
                torch.sum(true_positives, dim=0) + torch.sum(false_positives, dim=0) + 1e-10)
        recall[c - 1] = torch.sum(true_positives, dim=0) / n_easy_class_objects
        true_positives[c - 1] = torch.sum(true_positives, dim=0)
        false_positives[c - 1] =torch.sum(false_positives, dim=0)

        ##me
        # if (c==1):
        # precision=cumul_precision.max()
        # recall=cumul_recall.max()
        # axarr = f.add_subplot(1, 1, 1)
        plt.plot(recall_thresholds, precisions.cpu(), color='gold', lw=2)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall' + str(c))
        plt.ylabel('Precision' + str(c))
        plt.show()

    # Calculate Mean Average Precision (mAP)
    mean_average_precision = average_precisions.mean().item()

    # Keep class-wise average precisions in a dictionary
    average_precisions = {rev_label_map[c + 1]: v for c, v in enumerate(average_precisions.tolist())}
    precision = {rev_label_map[c + 1]: v for c, v in enumerate(precision.tolist())}
    recall = {rev_label_map[c + 1]: v for c, v in enumerate(recall.tolist())}

    return average_precisions, mean_average_precision, precision, recall, f, n_easy_class_objects, true_positives, false_positives


def calculate_mAP10(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties):
    """
    Calculate the Mean Average Precision (mAP) of detected objects.

    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation

    :param det_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
    :param det_labels: list of tensors, one tensor for each image containing detected objects' labels
    :param det_scores: list of tensors, one tensor for each image containing detected objects' labels' scores
    :param true_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
    :param true_labels: list of tensors, one tensor for each image containing actual objects' labels
    :param true_difficulties: list of tensors, one tensor for each image containing actual objects' difficulty (0 or 1)
    :return: list of average precisions for all classes, mean average precision (mAP)
    """
    global n_easy_class_objects
    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(
        true_labels) == len(
        true_difficulties)  # these are all lists of tensors of the same length, i.e. number of images
    n_classes = len(label_map)
    bir1=list()
    # for t in range(bir.__len__()):
    #     b=np.expand_dims(np.ascontiguousarray(bir[t]),axis=0)
    #     if t==0:
    #         bir1=b
    #     else:
    #         bir1=np.append(bir1,b,axis=0)
    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(
        device)  # (n_objects), n_objects is the total no. of objects across all images
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
    true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    # Calculate APs for each class (except background)
    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    precision = torch.zeros((n_classes - 1), dtype=torch.float)
    recall = torch.zeros((n_classes - 1), dtype=torch.float)
    true_positives = torch.zeros((n_classes - 1), dtype=torch.float)
    false_positives = torch.zeros((n_classes - 1), dtype=torch.float)
    # pre=torch.zeros((n_classes - 1), dtype=torch.float)
    f = plt.figure()

    for c in range(1, n_classes):
        # Extract only objects with this class
        true_class_images1 = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes1 = true_boxes[true_labels == c]  # (n_class_objects, 4)
        true_class_difficulties1 = true_difficulties[true_labels == c]  # (n_class_objects)
        n_class_objects1 = true_class_boxes1.size(0)

        true_class_images5 = list()
        true_class_boxes5 = list()
        true_class_difficulties5 = list()
        true_class_images1 = true_class_images1.unsqueeze(1)
        # true_class_boxes1=true_class_boxes1.unsqueeze(1)
        true_class_difficulties1 = true_class_difficulties1.unsqueeze(1)
        scale1 = 0/512
        scale = 100 / 512
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(0, n_class_objects1):
            if ((true_class_boxes1[i][3] - true_class_boxes1[i][1]) < scale):
                true_class_images5.append(true_class_images1[i])  # (n_class_detections)
                true_class_boxes5.append(true_class_boxes1[i].unsqueeze(0))  # (n_class_detections, 4)
                true_class_difficulties5.append(true_class_difficulties1[i].unsqueeze(0))
        true_class_images5 = torch.LongTensor(true_class_images5).to(device)
        # true_class_images = torch.cat(true_class_images, dim=0)  # (n_detections)
        if true_class_images5.size(0) == 0:
            # continue torch.Tensor()
            true_class_boxes5 = torch.Tensor()  # (n_detections, 4)
            true_class_difficulties5 = torch.Tensor()  # (n_detections)
        else:
            true_class_boxes5 = torch.cat(true_class_boxes5, dim=0)  # (n_detections, 4)
            true_class_difficulties5 = torch.cat(true_class_difficulties5, dim=0)  # (n_detections)
        # wt=true_class_boxes1[:][1]-true_class_boxes1[:][0]
        # ht=true_class_boxes1[:][3]-true_class_boxes1[:][2]
        true_class_images = list()
        true_class_boxes = list()
        true_class_difficulties = list()
        # true_class_images1 = true_class_images1.unsqueeze(1)
        # # true_class_boxes1=true_class_boxes1.unsqueeze(1)
        # true_class_difficulties1 = true_class_difficulties1.unsqueeze(1)

        for i in range(0, n_class_objects1):
            if ((true_class_boxes1[i][3] - true_class_boxes1[i][1]) >= scale):
                true_class_images.append(true_class_images1[i])  # (n_class_detections)
                true_class_boxes.append(true_class_boxes1[i].unsqueeze(0))  # (n_class_detections, 4)
                true_class_difficulties.append(true_class_difficulties1[i].unsqueeze(0))
        true_class_images = torch.LongTensor(true_class_images).to(device)
        # true_class_images = torch.cat(true_class_images, dim=0)  # (n_detections)
        if true_class_images.size(0) == 0:
            # continue torch.Tensor()
            true_class_boxes = torch.Tensor()  # (n_detections, 4)
            true_class_difficulties = torch.Tensor()  # (n_detections)
        else:
            true_class_boxes = torch.cat(true_class_boxes, dim=0)  # (n_detections, 4)
            true_class_difficulties = torch.cat(true_class_difficulties, dim=0)  # (n_detections)

        # (n_class_detections)

        # n_easy_class_objects = (true_class_difficulties==0).sum().item()  # ignore difficult objects
        n_easy_class_objects = (len(true_class_difficulties))
        # Keep track of which true objects with this class have already been 'detected'
        # So far, none
        true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).to(
            device)  # (n_class_objects)
        true_class_boxes_detected5 = torch.zeros((true_class_difficulties5.size(0)), dtype=torch.uint8).to(
            device)
        # Extract only detections with this class
        det_class_images1 = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes1 = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores1 = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections1 = det_class_boxes1.size(0)
        # wd=det_class_boxes1[:][1]-det_class_boxes1[:][0]
        # hd=det_class_boxes1[:][3]-det_class_boxes1[:][2]

        det_class_images = list()
        det_class_boxes = list()
        det_class_scores = list()
        det_class_images1 = det_class_images1.unsqueeze(1)
        # det_class_boxes1=det_class_boxes1.unsqueeze(1)
        # det_class_scores1=det_class_scores1.unsqueeze(1)
        for i in range(0, n_class_detections1):
            if ((det_class_boxes1[i][3] - det_class_boxes1[i][1] >= scale)):
                det_class_images.append(det_class_images1[i])  # (n_class_detections)
                det_class_boxes.append(det_class_boxes1[i].unsqueeze(0))  # (n_class_detections, 4)
                det_class_scores.append(det_class_scores1[i].unsqueeze(0))
        det_class_images = torch.LongTensor(det_class_images).to(device)  # (n_detections)
        n_class_detections = det_class_images.size(0)

        det_class_images2 = list()
        det_class_boxes2 = list()
        det_class_scores2 = list()
        # det_class_images1 = det_class_images1.unsqueeze(1)
        # det_class_boxes1=det_class_boxes1.unsqueeze(1)
        # det_class_scores1=det_class_scores1.unsqueeze(1)
        for i in range(0, n_class_detections1):
            if ((det_class_boxes1[i][3] - det_class_boxes1[i][1] < scale)):
                det_class_images2.append(det_class_images1[i])  # (n_class_detections)
                det_class_boxes2.append(det_class_boxes1[i].unsqueeze(0))  # (n_class_detections, 4)
                det_class_scores2.append(det_class_scores1[i].unsqueeze(0))
        det_class_images2 = torch.LongTensor(det_class_images2).to(device)  # (n_detections)
        n_class_detections2 = det_class_images2.size(0)
        true_positives = torch.zeros((n_class_detections1), dtype=torch.float).to(device)  # (n_class_detections)
        false_positives = torch.zeros((n_class_detections1), dtype=torch.float).to(device)  # (n_class_detections)
        if n_class_detections == 0 & n_class_detections2 == 0:
            continue



        det_class_boxes2 = torch.cat(det_class_boxes2, dim=0)  # (n_detections, 4)
        det_class_scores2 = torch.cat(det_class_scores2, dim=0)  # (n_detections)

        # Sort detections in decreasing order of confidence/scores
        det_class_scores2, sort_ind2 = torch.sort(det_class_scores2, dim=0, descending=True)  # (n_class_detections)
        det_class_images2 = det_class_images2[sort_ind2]  # (n_class_detections)
        det_class_boxes2 = det_class_boxes2[sort_ind2]  # (n_class_detections, 4)
        # In the order of decreasing scores, check if true or false positive
        if n_class_detections == 0 & n_class_detections2 != 0:
            det_scores2 = str(round(det_class_scores2[d - n_class_detections].to('cpu').tolist(), 2))
            this_detection_box6 = det_class_boxes2[d - n_class_detections].unsqueeze(0)  # (1, 4)
            this_image6 = det_class_images2[d - n_class_detections]  # (), scalar

            # Find objects in the same image with this class, their difficulties, and whether they have been detected before
            object_boxes6 = true_class_boxes[true_class_images == this_image6]  # (n_class_objects_in_img)
            # object_difficulties = true_class_difficulties[
            #     true_class_images == this_image6]  # (n_class_objects_in_img)
            # If no such object in this image, then the detection is a false positive

            if object_boxes6.size(0) == 0:
                continue

            overlaps6, overlaps61 = find_jaccard_overlap(this_detection_box6,
                                                         object_boxes6)  # (1, n_class_objects_in_img)
            max_overlap6, ind6 = torch.max(overlaps6.squeeze(0), dim=0)  # (), () - scalars
            original_ind6 = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image6][
                ind6]

            # if object_boxes3.size(0) != 0:
            # Find maximum overlap of this detection with objects in this image of this class

            # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
            # We need 'original_ind' to update 'true_class_boxes_detected'

            # If the maximum overlap is greater than the threshold of 0.5, it's a match
            if ((max_overlap6.item() > 0.5)):
                # If the object it matched with is 'difficult', ignore it
                # if object_difficulties[ind6] == 0:
                    # If this object has already not been detected, it's a true positive

                if true_class_boxes_detected[original_ind6] == 0:
                    true_positives[d] = 1
                    # box_location1 = this_detection_box6[0].tolist()
                    # bir1[this_image6] = cv2.rectangle(np.squeeze(bir1[this_image6], axis=0),
                    #                                  (int(box_location1[0]), int(box_location1[1])),
                    #                                  (int(box_location1[2]), int(box_location1[3])), (255, 0, 0),
                    #                                  2)
                    # # bir1[this_image6] = cv2.putText(bir1[this_image6], det_scores,
                    #                                org=(int(box_location1[0]) + 2, int(box_location1[1]) - 4),
                    #                                fontFace=font, fontScale=1 / 2, color=(255, 0, 0), thickness=2,
                    #                                lineType=cv2.LINE_AA)
                    # this_detection_box1 = this_detection_box.cpu() * original_dims.cpu()

                    true_class_boxes_detected[original_ind6] = 1
            else:
                if ((object_boxes6[ind6][2] - object_boxes6[ind6][0]) / (
                        object_boxes6[ind6][3] - object_boxes6[ind6][1]) >= 0.7):
                    if (max_overlap6 >= 0.1):
                        # if object_difficulties2[ind3] == 0:
                        # if object_difficulties[ind6] == 0:
                        true_positives[d] = 1
        #
        #
        #     # Sort detections in decreasing order of confidence/scores

            # for d in range(n_class_detections,n_class_detections2):
            #     det_scores2 = str(round(det_class_scores2[d-n_class_detections].to('cpu').tolist(), 2))
            #     this_detection_box2 = det_class_boxes2[d-n_class_detections].unsqueeze(0)  # (1, 4)
            #     this_image2 = det_class_images2[d-n_class_detections]  # (), scalar
            #
            #     # Find objects in the same image with this class, their difficulties, and whether they have been detected before
            #     object_boxes2 = true_class_boxes[true_class_images == this_image2]  # (n_class_objects_in_img)
            #     # object_difficulties2 = true_class_difficulties[
            #     #     true_class_images == this_image2]  # (n_class_objects_in_img)
            #     # If no such object in this image, then the detection is a false positive
            #
            #     if object_boxes2.size(0) == 0:
            #         continue
            #
            #     overlaps3, overlaps31 = find_jaccard_overlap(this_detection_box2,
            #                                                  object_boxes2)  # (1, n_class_objects_in_img)
            #     max_overlap3, ind3 = torch.max(overlaps3.squeeze(0), dim=0)  # (), () - scalars
            #     original_ind3 = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image2][
            #         ind3]
            #
            #     # if object_boxes3.size(0) != 0:
            #         # Find maximum overlap of this detection with objects in this image of this class
            #
            #         # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            #         # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
            #         # We need 'original_ind' to update 'true_class_boxes_detected'
            #
            #         # If the maximum overlap is greater than the threshold of 0.5, it's a match
            #     if ((max_overlap3.item() > 0.5)):
            #         # If the object it matched with is 'difficult', ignore it
            #         # if object_difficulties2[ind3] == 0:
            #         # If this object has already not been detected, it's a true positive
            #         if true_class_boxes_detected[original_ind3] == 0:
            #             true_positives[d] = 1
            #             # this_detection_box1 = this_detection_box.cpu() * original_dims.cpu()
            #
            #             true_class_boxes_detected[original_ind3] = 1
            #     else:
            #         if ((object_boxes2[ind3][2] - object_boxes2[ind3][0]) / (
            #             object_boxes2[ind3][3] - object_boxes2[ind3][1]) >= 0.7):
            #             if (max_overlap3 >= 0.1):
            #                 # if object_difficulties2[ind3] == 0:
            #                 true_positives[d] = 1
            # continue
        s1=int(75/512)
        det_class_boxes = torch.cat(det_class_boxes, dim=0)  # (n_detections, 4)
        det_class_scores = torch.cat(det_class_scores, dim=0)  # (n_detections)

        # Sort detections in decreasing order of confidence/scores
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)
        for d in range(n_class_detections1):
            if d<n_class_detections:
                this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
                this_image = det_class_images[d]  # (), scalar

                # Find objects in the same image with this class, their difficulties, and whether they have been detected before
                object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
                # object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)

                object_boxes5 = true_class_boxes5[true_class_images5 == this_image]  # (n_class_objects_in_img)
                # object_difficulties5 = true_class_difficulties5[true_class_images5 == this_image]
                # If no such object in this image, then the detection is a false positive
                if object_boxes.size(0) == 0:
                    if object_boxes5.size(0) == 0:
                        false_positives[d] = 1
                        continue

                # Find maximum overlap of this detection with objects in this image of this class
                if object_boxes5.size(0) != 0:
                    overlaps5, overlaps51 = find_jaccard_overlap(this_detection_box, object_boxes5)  # (1, n_class_objects_in_img)
                    max_overlap5, ind5 = torch.max(overlaps5.squeeze(0), dim=0)  # (), () - scalars
                    # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
                    # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
                    # We need 'original_ind' to update 'true_class_boxes_detected'
                    original_ind5 = torch.LongTensor(range(true_class_boxes5.size(0)))[true_class_images5== this_image][ind5]
                if object_boxes.size(0) != 0:
                    overlaps, overlaps1 = find_jaccard_overlap(this_detection_box,
                                                               object_boxes)  # (1, n_class_objects_in_img)
                    max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars
                    original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]
                    if ((max_overlap.item() > 0.5) or ((object_boxes[ind][3]-object_boxes[ind][1])<s1 & (max_overlap.item() > 0.1))):
                        # If the object it matched with is 'difficult', ignore it
                        # if object_difficulties[ind] == 0:
                            # If this object has already not been detected, it's a true positive
                        if true_class_boxes_detected[original_ind] == 0:
                            true_positives[d] = 1
                            # box_location1 = this_detection_box[0].tolist()
                            # bir1[this_image] = cv2.rectangle(np.squeeze(bir1[this_image], axis=0), (int(box_location1[0]), int(box_location1[1])),
                            #                       (int(box_location1[2]), int(box_location1[3])), (255, 0, 0),
                            #                       2)
                            # bir1[this_image] = cv2.putText(bir1[this_image], det_scores,
                            #                     org=(int(box_location1[0]) + 2, int(box_location1[1]) - 4),
                            #                     fontFace=font, fontScale=1 / 2, color=(255, 0, 0), thickness=2,
                            #                     lineType=cv2.LINE_AA)
                            true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                        # Otherwise, it's a false positive (since this object is already accounted for)
                        else:
                            false_positives[d] = 1
                    # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
                    else:
                        if((object_boxes[ind][2]-object_boxes[ind][0])/(object_boxes[ind][3]-object_boxes[ind][1])>=0.7) :
                            if (max_overlap>=0.1):
                                # if object_difficulties[ind] == 0:
                                true_positives[d] = 1
                                # box_location1 = this_detection_box[0].tolist()
                                # bir1[this_image] = cv2.rectangle(np.squeeze(bir1[this_image], axis=0),
                                #                                  (int(box_location1[0]), int(box_location1[1])),
                                #                                  (int(box_location1[2]), int(box_location1[3])),
                                #                                  (255, 0, 0),
                                #                                  2)
                                # bir1[this_image] = cv2.putText(bir1[this_image], det_scores,
                                #                                org=(
                                #                                int(box_location1[0]) + 2, int(box_location1[1]) - 4),
                                #                                fontFace=font, fontScale=1 / 2, color=(255, 0, 0),
                                #                                thickness=2,
                                #                                lineType=cv2.LINE_AA)
                            else:
                                if object_boxes5.size(0) != 0:
                                    if ((max_overlap5.item() > 0.5) or (
                                            (object_boxes5[ind5][3] - object_boxes5[ind5][1]) < s1 & (
                                            max_overlap5.item() > 0.1))):
                                        # If the object it matched with is 'difficult', ignore it
                                        # if object_difficulties5[ind5] == 0:
                                        # If this object has already not been detected, it's a true positive
                                        # if object_difficulties5[ind5] == 0:
                                        if true_class_boxes_detected5[original_ind5] == 0:
                                            # true_positives[d] = 1
                                            true_class_boxes_detected5[
                                                original_ind5] = 1  # this object has now been detected/accounted for
                                        # Otherwise, it's a false positive (since this object is already accounted for)
                                        else:
                                            false_positives[d] = 1
                                    else:
                                        if ((object_boxes5[ind5][2] - object_boxes5[ind5][0]) / (
                                                object_boxes5[ind5][3] - object_boxes5[ind5][1]) >= 0.7):
                                            # if object_difficulties5[ind5] == 0:
                                            if (max_overlap5 >= 0.1):

                                                print("")
                                            else:
                                                false_positives[d] = 1
                                        else:
                                            false_positives[d] = 1
                                else:
                                    false_positives[d] = 1
                        else:
                            if object_boxes5.size(0) != 0:
                                if ((max_overlap5.item() > 0.5) or ((object_boxes5[ind5][3]-object_boxes5[ind5][1])<s1 & (max_overlap5.item() > 0.1))):
                                    # If the object it matched with is 'difficult', ignore it
                                    # if object_difficulties5[ind5] == 0:
                                        # If this object has already not been detected, it's a true positive
                                    if true_class_boxes_detected5[original_ind5] == 0:
                                        # true_positives[d] = 1
                                        true_class_boxes_detected5[
                                            original_ind5] = 1  # this object has now been detected/accounted for
                                    # Otherwise, it's a false positive (since this object is already accounted for)
                                    else:
                                        false_positives[d] = 1
                                else:
                                    if ((object_boxes5[ind5][2] - object_boxes5[ind5][0]) / (object_boxes5[ind5][3] - object_boxes5[ind5][1])>=0.7):
                                        # if object_difficulties5[ind5] == 0:
                                        if (max_overlap5 >= 0.1):
                                            print("")
                                        else:
                                            false_positives[d] = 1
                                    else:
                                        false_positives[d] = 1
                            else:
                                false_positives[d] = 1
                else:
                    if object_boxes5.size(0) != 0:
                        if ((max_overlap5.item() > 0.5) or (
                                (object_boxes5[ind5][3] - object_boxes5[ind5][1]) < s1 & (max_overlap5.item() > 0.1))):
                            # If the object it matched with is 'difficult', ignore it
                            # if object_difficulties5[ind5] == 0:
                            # If this object has already not been detected, it's a true positive
                            if true_class_boxes_detected5[original_ind5] == 0:
                                # true_positives[d] = 1
                                true_class_boxes_detected5[
                                    original_ind5] = 1  # this object has now been detected/accounted for
                            # Otherwise, it's a false positive (since this object is already accounted for)
                            else:
                                false_positives[d] = 1
                        else:
                            if ((object_boxes5[ind5][2] - object_boxes5[ind5][0]) / (
                                    object_boxes5[ind5][3] - object_boxes5[ind5][1]) >= 0.7):
                                # if object_difficulties5[ind5] == 0:
                                if (max_overlap5 >= 0.1):
                                    print("")
                                else:
                                    false_positives[d] = 1
                            else:
                                false_positives[d] = 1
                    else:
                        false_positives[d] = 1
            else:
                # for d in range(n_class_detections2):
                det_scores2 = str(round(det_class_scores2[d-n_class_detections].to('cpu').tolist(), 2))
                this_detection_box6 = det_class_boxes2[d-n_class_detections].unsqueeze(0)  # (1, 4)
                this_image6 = det_class_images2[d-n_class_detections]  # (), scalar

                # Find objects in the same image with this class, their difficulties, and whether they have been detected before
                object_boxes6 = true_class_boxes[true_class_images == this_image6]  # (n_class_objects_in_img)
                # object_difficulties = true_class_difficulties[
                #     true_class_images == this_image6]  # (n_class_objects_in_img)
                # If no such object in this image, then the detection is a false positive

                if object_boxes6.size(0) == 0:
                    continue

                overlaps6, overlaps61 = find_jaccard_overlap(this_detection_box6,
                                                             object_boxes6)  # (1, n_class_objects_in_img)
                max_overlap6, ind6 = torch.max(overlaps6.squeeze(0), dim=0)  # (), () - scalars
                original_ind6 = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image6][
                    ind6]

                # if object_boxes3.size(0) != 0:
                # Find maximum overlap of this detection with objects in this image of this class

                # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
                # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
                # We need 'original_ind' to update 'true_class_boxes_detected'

                # If the maximum overlap is greater than the threshold of 0.5, it's a match
                if ((max_overlap6.item() > 0.5)):
                    # If the object it matched with is 'difficult', ignore it
                    # if object_difficulties[ind6] == 0:
                    # If this object has already not been detected, it's a true positive

                    if true_class_boxes_detected[original_ind6] == 0:
                        true_positives[d] = 1
                        # box_location1 = this_detection_box6[0].tolist()
                        # bir1[this_image6] = cv2.rectangle(np.squeeze(bir1[this_image6], axis=0),
                        #                                  (int(box_location1[0]), int(box_location1[1])),
                        #                                  (int(box_location1[2]), int(box_location1[3])), (255, 0, 0),
                        #                                  2)
                        # # bir1[this_image6] = cv2.putText(bir1[this_image6], det_scores,
                        #                                org=(int(box_location1[0]) + 2, int(box_location1[1]) - 4),
                        #                                fontFace=font, fontScale=1 / 2, color=(255, 0, 0), thickness=2,
                        #                                lineType=cv2.LINE_AA)
                        # this_detection_box1 = this_detection_box.cpu() * original_dims.cpu()

                        true_class_boxes_detected[original_ind6] = 1
                else:
                    if ((object_boxes6[ind6][2] - object_boxes6[ind6][0]) / (
                            object_boxes6[ind6][3] - object_boxes6[ind6][1]) >= 0.7):
                        if (max_overlap6 >= 0.1):
                            # if object_difficulties2[ind3] == 0:
                            # if object_difficulties[ind6] == 0:
                            true_positives[d] = 1
                            # box_location1 = this_detection_box6[0].tolist()
                            # bir1[this_image6] = cv2.rectangle(np.squeeze(bir1[this_image6], axis=0),
                            #                                   (int(box_location1[0]), int(box_location1[1])),
                            #                                   (int(box_location1[2]), int(box_location1[3])),
                            #                                   (255, 0, 0),
                            #                                   2)
                            # bir1[this_image6] = cv2.putText(bir1[this_image6], det_scores,
                            #                                 org=(int(box_location1[0]) + 2, int(box_location1[1]) - 4),
                            #                                 fontFace=font, fontScale=1 / 2, color=(255, 0, 0),
                            #                                 thickness=2,
                            #                                 lineType=cv2.LINE_AA)
            # If the maximum overlap is greater than the threshold of 0.5, it's a match
        # if n_class_detections2 !=0 :
        # for d in range(true_class_images.size(0)):
        #     # det_scores2 = str(round(det_class_scores2[d].to('cpu').tolist(), 2))
        #     # this_detection_box6 = det_class_boxes2[d].unsqueeze(0)  # (1, 4)
        #     # this_image6 = det_class_images2[d]  # (), scalar
        #     # det_class_boxes2=torch.stack(det_class_boxes2,dim=0)
        #     # Find objects in the same image with this class, their difficulties, and whether they have been detected before
        #     object_boxes6 = det_class_boxes2[true_class_images[d] == det_class_images2]  # (n_class_objects_in_img)
        #     # object_difficulties2 = true_class_difficulties[
        #     #     true_class_images == this_image2]  # (n_class_objects_in_img)
        #     # If no such object in this image, then the detection is a false positive
        #
        #     if object_boxes6.size(0) == 0:
        #         continue
        #
        #     overlaps6, overlaps61 = find_jaccard_overlap(true_class_boxes[d].unsqueeze(0),
        #                                                  object_boxes6)  # (1, n_class_objects_in_img)
        #     max_overlap6, ind6 = torch.max(overlaps6.squeeze(0), dim=0)  # (), () - scalars
        #     original_ind6 = torch.LongTensor(range(true_class_boxes.size(0)))[d]
        #
        #         # if object_boxes3.size(0) != 0:
        #         # Find maximum overlap of this detection with objects in this image of this class
        #
        #         # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
        #         # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
        #         # We need 'original_ind' to update 'true_class_boxes_detected'
        #
        #         # If the maximum overlap is greater than the threshold of 0.5, it's a match
        #     if ((max_overlap6.item() > 0.5)):
        #         # If the object it matched with is 'difficult', ignore it
        #         # if object_difficulties2[ind3] == 0:
        #         # If this object has already not been detected, it's a true positive
        #         if true_class_boxes_detected[original_ind6] == 0:
        #             true_positives[d+n_class_detections] = 1
        #             # this_detection_box1 = this_detection_box.cpu() * original_dims.cpu()
        #
        #             true_class_boxes_detected[original_ind6] = 1
        #     else:
        #         if ((true_class_boxes[d][2] - true_class_boxes[d][0]) / (
        #                 true_class_boxes[d][3] - true_class_boxes[d][1]) >= 0.7):
        #             if (max_overlap6 >= 0.1):
        #                 # if object_difficulties2[ind3] == 0:
        #                 true_positives[d+n_class_detections] = 1


                # false_positives[d] = 1
        # for y in range(bir.__len__()):
        #     cv2.imwrite('/home/fereshteh/result2/' + str(y) + '.png', bir1[y])
        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (
                cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)
        print(true_positives)
        print(cumul_true_positives)
        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        # pre[c-1]=precisions.cpu()
        average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]
        precision[c - 1] =torch.sum(true_positives, dim=0)/ (
                torch.sum(true_positives, dim=0) + torch.sum(false_positives, dim=0) + 1e-10)
        recall[c - 1] = torch.sum(true_positives, dim=0) / n_easy_class_objects
        true_positives[c - 1] = torch.sum(true_positives, dim=0)
        false_positives[c - 1] =torch.sum(false_positives, dim=0)

        ##me
        # if (c==1):
        # precision=cumul_precision.max()
        # recall=cumul_recall.max()
        # axarr = f.add_subplot(1, 1, 1)
        plt.plot(recall_thresholds, precisions.cpu(), color='gold', lw=2)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall' + str(c))
        plt.ylabel('Precision' + str(c))
        plt.show()

    # Calculate Mean Average Precision (mAP)
    mean_average_precision = average_precisions.mean().item()

    # Keep class-wise average precisions in a dictionary
    average_precisions = {rev_label_map[c + 1]: v for c, v in enumerate(average_precisions.tolist())}
    precision = {rev_label_map[c + 1]: v for c, v in enumerate(precision.tolist())}
    recall = {rev_label_map[c + 1]: v for c, v in enumerate(recall.tolist())}

    return average_precisions, mean_average_precision, precision, recall, f, n_easy_class_objects, true_positives, false_positives

def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, loss, best_loss, is_best):
    """
    Save model checkpoint.

    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement
    :param model: model
    :param optimizer: optimizer
    :param loss: validation loss in this epoch
    :param best_loss: best validation loss achieved so far (not necessarily in this checkpoint)
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'loss': loss,
             'best_loss': best_loss,
             'model': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    filename = '513mob2_fine_checkpoint_ssd_lw_bn_new7.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)

def save_checkpoint2(epoch, epochs_since_improvement,model1,model2,jumper,optimizer2,  loss, best_loss, is_best):
    """
    Save model checkpoint.

    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement
    :param model: model
    :param optimizer: optimizer
    :param loss: validation loss in this epoch
    :param best_loss: best validation loss achieved so far (not necessarily in this checkpoint)
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'loss': loss,
             'best_loss': best_loss,
             'model1': model1.state_dict(),
             'model2': model2.state_dict(),
             'jumper': jumper.state_dict(),

             'optimizer2': optimizer2.state_dict()}
    filename = 'refine_fine_checkpoint_ssd300_8.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def detect_objects(model, priors_cxcy, predicted_locs, predicted_scores, min_score, max_overlap, top_k, n_classes):
        """
        Decipher the 8732 locations and class scores (output of ths SSD300) to detect objects.
        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.
        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = predicted_locs.size(0)
        n_priors = priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()
        # print(n_priors , predicted_locs.size(1),predicted_scores.size(1))
        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], priors_cxcy))  # (8732, 4), these are fractional pt. coordinates

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)

            # Check for each class
            for c in range(1, n_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (8732)
                score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                # Find the overlap between predicted boxes
                overlap, overlap1 = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # Non-Maximum Suppression (NMS)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score), dtype=torch.bool).to(device)  # (n_qualified)
                
                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue

                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    #suppress = torch.max(suppress, overlap[box] > max_overlap)
                    condition = overlap[box] > max_overlap
                    condition = torch.tensor(condition, dtype=torch.bool).to(device)
      
                    suppress = torch.max(suppress, condition)  
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0

                # Store only unsuppressed boxes for this class
                image_boxes.append(class_decoded_locs[~ suppress])
                image_labels.append(torch.LongTensor((~ suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[~ suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size
        
        
def calculate_mAP_kaist(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties):
    """
    Calculate the Mean Average Precision (mAP) of detected objects.

    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation

    :param det_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
    :param det_labels: list of tensors, one tensor for each image containing detected objects' labels
    :param det_scores: list of tensors, one tensor for each image containing detected objects' labels' scores
    :param true_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
    :param true_labels: list of tensors, one tensor for each image containing actual objects' labels
    :param true_difficulties: list of tensors, one tensor for each image containing actual objects' difficulty (0 or 1)
    :return: list of average precisions for all classes, mean average precision (mAP)
    """
    global n_easy_class_objects
    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(
        true_labels) == len(
        true_difficulties)  # these are all lists of tensors of the same length, i.e. number of images
    n_classes = len(label_map)

    # for t in range(bir.__len__()):
    #     b=np.expand_dims(np.ascontiguousarray(bir[t]),axis=0)
    #     if t==0:
    #         bir1=b
    #     else:
    #         bir1=np.append(bir1,b,axis=0)
    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(
        device)  # (n_objects), n_objects is the total no. of objects across all images
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
    true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    # Calculate APs for each class (except background)
    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    precision = torch.zeros((n_classes - 1), dtype=torch.float)
    recall = torch.zeros((n_classes - 1), dtype=torch.float)
    true_positives = torch.zeros((n_classes - 1), dtype=torch.float)
    false_positives = torch.zeros((n_classes - 1), dtype=torch.float)
    # pre=torch.zeros((n_classes - 1), dtype=torch.float)
    f = plt.figure()

    for c in range(1, n_classes):
        # Extract only objects with this class
        true_class_labels1 = true_labels[true_labels != 0 ]
        true_class_images1 = true_images[true_labels != 0]  # (n_class_objects)
        true_class_boxes1 = true_boxes[true_labels != 0]  # (n_class_objects, 4)
        true_class_difficulties1 = true_difficulties[true_labels != 0]  # (n_class_objects)
        n_class_objects1 = true_class_boxes1.size(0)

        true_class_labels5 = list()
        true_class_images5 = list()
        true_class_boxes5 = list()
        true_class_difficulties5 = list()
        true_class_images1 = true_class_images1.unsqueeze(1)
        true_class_labels1 = true_class_labels1.unsqueeze(1)
        # true_class_boxes1=true_class_boxes1.unsqueeze(1)
        true_class_difficulties1 = true_class_difficulties1.unsqueeze(1)

        scale = 56 / 512

        for i in range(0, n_class_objects1):
            if ((true_class_boxes1[i][3] - true_class_boxes1[i][1]) < scale):
                true_class_labels5.append(true_class_labels1[i])
                true_class_images5.append(true_class_images1[i])  # (n_class_detections)
                true_class_boxes5.append(true_class_boxes1[i].unsqueeze(0))  # (n_class_detections, 4)
                true_class_difficulties5.append(true_class_difficulties1[i].unsqueeze(0))
        true_class_images5 = torch.LongTensor(true_class_images5).to(device)
        # true_class_images = torch.cat(true_class_images, dim=0)  # (n_detections)
        if true_class_images5.size(0) == 0:
            # continue torch.Tensor()
            true_class_labels5 = torch.Tensor()
            true_class_boxes5 = torch.Tensor()  # (n_detections, 4)
            true_class_difficulties5 = torch.Tensor()  # (n_detections)
        else:
            true_class_labels5 =torch.cat(true_class_labels5, dim=0)
            true_class_boxes5 = torch.cat(true_class_boxes5, dim=0)  # (n_detections, 4)
            true_class_difficulties5 = torch.cat(true_class_difficulties5, dim=0)  # (n_detections)
        # wt=true_class_boxes1[:][1]-true_class_boxes1[:][0]
        # ht=true_class_boxes1[:][3]-true_class_boxes1[:][2]
        true_class_images = list()
        true_class_boxes = list()
        true_class_difficulties = list()
        true_class_labels = list()
        # true_class_images1 = true_class_images1.unsqueeze(1)
        # # true_class_boxes1=true_class_boxes1.unsqueeze(1)
        # true_class_difficulties1 = true_class_difficulties1.unsqueeze(1)

        for i in range(0, n_class_objects1):
            if ((true_class_boxes1[i][3] - true_class_boxes1[i][1]) >= scale):
                true_class_labels.append(true_class_labels1[i])
                true_class_images.append(true_class_images1[i])  # (n_class_detections)
                true_class_boxes.append(true_class_boxes1[i].unsqueeze(0))  # (n_class_detections, 4)
                true_class_difficulties.append(true_class_difficulties1[i].unsqueeze(0))
        true_class_images = torch.LongTensor(true_class_images).to(device)
        # true_class_images = torch.cat(true_class_images, dim=0)  # (n_detections)
        if true_class_images.size(0) == 0:
            # continue torch.Tensor()
            true_class_labels = torch.Tensor()
            true_class_boxes = torch.Tensor()  # (n_detections, 4)
            true_class_difficulties = torch.Tensor()  # (n_detections)
        else:
            true_class_labels = torch.cat(true_class_labels, dim=0)
            true_class_boxes = torch.cat(true_class_boxes, dim=0)  # (n_detections, 4)
            true_class_difficulties = torch.cat(true_class_difficulties, dim=0)  # (n_detections)

        # (n_class_detections)

        n_easy_class_objects =((true_class_difficulties[(true_class_labels)==1]!=2)).sum().item()  # ignore difficult objects
        # n_easy_class_objects = (len(true_class_difficulties))
        # Keep track of which true objects with this class have already been 'detected'
        # So far, none
        true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).to(
            device)  # (n_class_objects)
        true_class_boxes_detected5 = torch.zeros((true_class_difficulties5.size(0)), dtype=torch.uint8).to(
            device)
        # Extract only detections with this class
        det_class_images1 = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes1 = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores1 = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections1 = det_class_boxes1.size(0)
        # wd=det_class_boxes1[:][1]-det_class_boxes1[:][0]
        # hd=det_class_boxes1[:][3]-det_class_boxes1[:][2]

        det_class_images = list()
        det_class_boxes = list()
        det_class_scores = list()
        det_class_images1 = det_class_images1.unsqueeze(1)
        # det_class_boxes1=det_class_boxes1.unsqueeze(1)
        # det_class_scores1=det_class_scores1.unsqueeze(1)
        for i in range(0, n_class_detections1):
            if ((det_class_boxes1[i][3] - det_class_boxes1[i][1] >= scale)):
                det_class_images.append(det_class_images1[i])  # (n_class_detections)
                det_class_boxes.append(det_class_boxes1[i].unsqueeze(0))  # (n_class_detections, 4)
                det_class_scores.append(det_class_scores1[i].unsqueeze(0))
        det_class_images = torch.LongTensor(det_class_images).to(device)  # (n_detections)
        n_class_detections = det_class_images.size(0)

        det_class_images2 = list()
        det_class_boxes2 = list()
        det_class_scores2 = list()
        # det_class_images1 = det_class_images1.unsqueeze(1)
        # det_class_boxes1=det_class_boxes1.unsqueeze(1)
        # det_class_scores1=det_class_scores1.unsqueeze(1)
        for i in range(0, n_class_detections1):
            if ((det_class_boxes1[i][3] - det_class_boxes1[i][1] < scale)):
                det_class_images2.append(det_class_images1[i])  # (n_class_detections)
                det_class_boxes2.append(det_class_boxes1[i].unsqueeze(0))  # (n_class_detections, 4)
                det_class_scores2.append(det_class_scores1[i].unsqueeze(0))
        det_class_images2 = torch.LongTensor(det_class_images2).to(device)  # (n_detections)
        n_class_detections2 = det_class_images2.size(0)
        true_positives = torch.zeros((n_class_detections1), dtype=torch.float).to(device)  # (n_class_detections)
        false_positives = torch.zeros((n_class_detections1), dtype=torch.float).to(device)  # (n_class_detections)
        if n_class_detections == 0 & n_class_detections2 == 0:
            continue

        if n_class_detections2 != 0:
            det_class_boxes2 = torch.cat(det_class_boxes2, dim=0)  # (n_detections, 4)
            det_class_scores2 = torch.cat(det_class_scores2, dim=0)  # (n_detections)
        else:
            det_class_boxes2 =torch.Tensor()  # (n_detections, 4)
            det_class_scores2 =torch.Tensor()


        # Sort detections in decreasing order of confidence/scores
        det_class_scores2, sort_ind2 = torch.sort(det_class_scores2, dim=0, descending=True)  # (n_class_detections)
        det_class_images2 = det_class_images2[sort_ind2]  # (n_class_detections)
        det_class_boxes2 = det_class_boxes2[sort_ind2]  # (n_class_detections, 4)
        # In the order of decreasing scores, check if true or false positive
        if n_class_detections == 0 & n_class_detections2 != 0:
            det_scores2 = str(round(det_class_scores2[d - n_class_detections].to('cpu').tolist(), 2))
            this_detection_box6 = det_class_boxes2[d - n_class_detections].unsqueeze(0)  # (1, 4)
            this_image6 = det_class_images2[d - n_class_detections]  # (), scalar

            # Find objects in the same image with this class, their difficulties, and whether they have been detected before
            object_boxes6 = true_class_boxes[true_class_images == this_image6 & true_class_labels ==1]  # (n_class_objects_in_img)
            object_difficulties = true_class_difficulties[
                true_class_images == this_image6]  # (n_class_objects_in_img)
            # If no such object in this image, then the detection is a false positive

            if object_boxes6.size(0) == 0:
                continue

            overlaps6, overlaps61 = find_jaccard_overlap(this_detection_box6,
                                                         object_boxes6)  # (1, n_class_objects_in_img)
            max_overlap6, ind6 = torch.max(overlaps6.squeeze(0), dim=0)  # (), () - scalars
            max_overlap61, ind61 = torch.max(overlaps61.squeeze(0), dim=0)  # (), () - scalars
            original_ind6 = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image6][
                ind6]

            # if object_boxes3.size(0) != 0:
            # Find maximum overlap of this detection with objects in this image of this class

            # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
            # We need 'original_ind' to update 'true_class_boxes_detected'

            # If the maximum overlap is greater than the threshold of 0.5, it's a match
            if ((max_overlap6.item() > 0.5)):
                # If the object it matched with is 'difficult', ignore it
                if object_difficulties[ind6] != 2:
                    # If this object has already not been detected, it's a true positive

                    if true_class_boxes_detected[original_ind6] == 0:
                        true_positives[d] = 1
                        #
                        true_class_boxes_detected[original_ind6] = 1

        #
        #
        det_class_boxes = torch.cat(det_class_boxes, dim=0)  # (n_detections, 4)
        det_class_scores = torch.cat(det_class_scores, dim=0)  # (n_detections)

        # Sort detections in decreasing order of confidence/scores
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)
        for d in range(n_class_detections1):
            if d<n_class_detections:
                this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
                this_image = det_class_images[d]  # (), scalar

                # Find objects in the same image with this class, their difficulties, and whether they have been detected before
                object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
                object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
                object_labels = true_class_labels[true_class_images == this_image]

                object_boxes5 = true_class_boxes5[true_class_images5 == this_image]  # (n_class_objects_in_img)
                object_difficulties5 = true_class_difficulties5[true_class_images5 == this_image]
                object_labels5 = true_class_labels5[true_class_images5 == this_image]
                # If no such object in this image, then the detection is a false positive
                if object_boxes.size(0) == 0:
                    if object_boxes5.size(0) == 0:
                        false_positives[d] = 1
                        continue

                # Find maximum overlap of this detection with objects in this image of this class
                if object_boxes5.size(0) != 0:
                    overlaps5, overlaps51 = find_jaccard_overlap(this_detection_box, object_boxes5)  # (1, n_class_objects_in_img)
                    max_overlap5, ind5 = torch.max(overlaps5.squeeze(0), dim=0)  # (), () - scalars
                    max_overlap51, ind51 = torch.max(overlaps51.squeeze(0), dim=0)
                    # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
                    # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
                    # We need 'original_ind' to update 'true_class_boxes_detected'
                    original_ind5 = torch.LongTensor(range(true_class_boxes5.size(0)))[true_class_images5== this_image][ind5]
                if object_boxes.size(0) != 0:
                    overlaps, overlaps1 = find_jaccard_overlap(this_detection_box,
                                                               object_boxes)  # (1, n_class_objects_in_img)
                    max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars
                    max_overlap1, ind1 = torch.max(overlaps1.squeeze(0), dim=0)
                    original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]
                    if ((max_overlap.item() > 0.5) & bool((object_labels[ind] ==1)) ):
                        # If the object it matched with is 'difficult', ignore it
                        if object_difficulties[ind] != 2:
                            # If this object has already not been detected, it's a true positive
                            if true_class_boxes_detected[original_ind] == 0 :
                                true_positives[d] = 1
                                # box_location1 = this_detection_box[0].tolist()
                                # bir1[this_image] = cv2.rectangle(np.squeeze(bir1[this_image], axis=0), (int(box_location1[0]), int(box_location1[1])),
                                #                       (int(box_location1[2]), int(box_location1[3])), (255, 0, 0),
                                #                       2)
                                # bir1[this_image] = cv2.putText(bir1[this_image], det_scores,
                                #                     org=(int(box_location1[0]) + 2, int(box_location1[1]) - 4),
                                #                     fontFace=font, fontScale=1 / 2, color=(255, 0, 0), thickness=2,
                                #                     lineType=cv2.LINE_AA)
                                true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                            # Otherwise, it's a false positive (since this object is already accounted for)
                            else:
                                false_positives[d] = 1
                    # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
                    elif(not((max_overlap1.item() > 0.5) & bool(object_labels[ind1] ==2))):

                        if object_boxes5.size(0) != 0:
                            if ((max_overlap5.item() > 0.5) or ((object_labels5[ind5] ==1))):
                                # If the object it matched with is 'difficult', ignore it

                                    # If this object has already not been detected, it's a true positive
                                if true_class_boxes_detected5[original_ind5] == 0:
                                    # true_positives[d] = 1
                                    true_class_boxes_detected5[
                                        original_ind5] = 1  # this object has now been detected/accounted for
                                    # Otherwise, it's a false positive (since this object is already accounted for)
                                else:
                                    false_positives[d] = 1
                            elif (not ((max_overlap51.item() > 0.5) & bool((object_labels5[ind51] == 2)))):

                                false_positives[d] = 1
                        else:
                            false_positives[d] = 1
                else:
                    if object_boxes5.size(0) != 0:
                        if ((max_overlap5.item() > 0.5) or ((object_labels5[ind5] == 1))):
                            # If the object it matched with is 'difficult', ignore it

                            # If this object has already not been detected, it's a true positive
                            if true_class_boxes_detected5[original_ind5] == 0:
                                # true_positives[d] = 1
                                true_class_boxes_detected5[
                                    original_ind5] = 1  # this object has now been detected/accounted for
                                # Otherwise, it's a false positive (since this object is already accounted for)
                            else:
                                false_positives[d] = 1
                        elif (not ((max_overlap51.item() > 0.5) & bool(object_labels5[ind51] == 2))):

                            false_positives[d] = 1
                    else:
                        false_positives[d] = 1
            else:
                # for d in range(n_class_detections2):

                this_detection_box6 = det_class_boxes2[d-n_class_detections].unsqueeze(0)  # (1, 4)
                this_image6 = det_class_images2[d-n_class_detections]  # (), scalar

                # Find objects in the same image with this class, their difficulties, and whether they have been detected before
                object_boxes6 = true_class_boxes[true_class_images == this_image6]  # (n_class_objects_in_img)
                object_difficulties6 = true_class_difficulties[
                    true_class_images == this_image6]  # (n_class_objects_in_img)
                object_labels6 = true_class_labels[true_class_images == this_image6]
                # If no such object in this image, then the detection is a false positive

                if object_boxes6.size(0) == 0:
                    continue

                overlaps6, overlaps61 = find_jaccard_overlap(this_detection_box6,
                                                             object_boxes6)  # (1, n_class_objects_in_img)
                max_overlap6, ind6 = torch.max(overlaps6.squeeze(0), dim=0)  # (), () - scalars
                max_overlap61, ind61 = torch.max(overlaps61.squeeze(0), dim=0)  # (), () - scalars
                original_ind6 = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image6][
                    ind6]


                # if object_boxes3.size(0) != 0:
                # Find maximum overlap of this detection with objects in this image of this class

                # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
                # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
                # We need 'original_ind' to update 'true_class_boxes_detected'

                # If the maximum overlap is greater than the threshold of 0.5, it's a match
                if ((max_overlap6.item() > 0.5) & bool(object_labels6[ind6]==1)):
                    # If the object it matched with is 'difficult', ignore it
                    if object_difficulties6[ind6] != 2:
                    # If this object has already not been detected, it's a true positive

                        if true_class_boxes_detected[original_ind6] == 0:
                            true_positives[d] = 1
                            # box_location1 = this_detection_box6[0].tolist()
                            # bir1[this_image6] = cv2.rectangle(np.squeeze(bir1[this_image6], axis=0),
                            #                                  (int(box_location1[0]), int(box_location1[1])),
                            #                                  (int(box_location1[2]), int(box_location1[3])), (255, 0, 0),
                            #                                  2)
                            # # bir1[this_image6] = cv2.putText(bir1[this_image6], det_scores,
                            #                                org=(int(box_location1[0]) + 2, int(box_location1[1]) - 4),
                            #                                fontFace=font, fontScale=1 / 2, color=(255, 0, 0), thickness=2,
                            #                                lineType=cv2.LINE_AA)
                            # this_detection_box1 = this_detection_box.cpu() * original_dims.cpu()

                            true_class_boxes_detected[original_ind6] = 1
                elif not((max_overlap61.item() > 0.5) & bool(object_labels6[ind61]==2)):
                    None
        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (
                cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)
        print(true_positives)
        print(cumul_true_positives)
        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        # pre[c-1]=precisions.cpu()
        average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]
        precision[c - 1] =torch.sum(true_positives, dim=0)/ (
                torch.sum(true_positives, dim=0) + torch.sum(false_positives, dim=0) + 1e-10)
        recall[c - 1] = torch.sum(true_positives, dim=0) / n_easy_class_objects
        true_positives[c - 1] = torch.sum(true_positives, dim=0)
        false_positives[c - 1] =torch.sum(false_positives, dim=0)
        fppi = (1 - precision[c - 1])
        mr = (1 - recall[c - 1])

        fppi_tmp = np.insert(fppi, 0, -1.0)
        mr_tmp = np.insert(mr, 0, 1.0)
        ref = np.logspace(-2.0, 1, num=9)
        # for i,ref_i in enumerate(ref):
        #     j=np.where(fppi_tmp<= ref_i)[-1][-1]
        #     ref[i]=mr_tmp[j]
        for i, ref_i in enumerate(ref):
            j = np.where(fppi_tmp <= ref_i)[-1][-1]
            if ref_i == 0.1:
                x = mr_tmp[i]
            ref[i] = mr_tmp[j]

        lamr = math.exp(np.mean(np.log(np.maximum(1e-1, ref))))
        ##me
        # if (c==1):
        # precision=cumul_precision.max()
        # recall=cumul_recall.max()
        # axarr = f.add_subplot(1, 1, 1)
        plt.plot(recall_thresholds, precisions.cpu(), color='gold', lw=2)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall' + str(c))
        plt.ylabel('Precision' + str(c))
        plt.show()

    # Calculate Mean Average Precision (mAP)
    mean_average_precision = average_precisions.mean().item()

    # Keep class-wise average precisions in a dictionary
    average_precisions = {rev_label_map[c + 1]: v for c, v in enumerate(average_precisions.tolist())}
    precision = {rev_label_map[c + 1]: v for c, v in enumerate(precision.tolist())}
    recall = {rev_label_map[c + 1]: v for c, v in enumerate(recall.tolist())}

    return average_precisions, mean_average_precision, precision, recall, f, n_easy_class_objects, true_positives, false_positives,lamr





