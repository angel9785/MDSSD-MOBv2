# # voc_labels = ('nonperson','person')
# #
# # label_map = {k: v for v, k in enumerate(voc_labels)}
# # # label_map['background'] = 0
# # # label_map['person'] = 1
# # rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping
# #
# # import re
# # def parse_annotation1(filename):
# #     M = []
# #     boxes = list()  # torch.FloatTensor([0, 0, 0, 0])#.unsqueeze(0)
# #     labels = list()
# #     difficulties = list()
# #     with open(filename) as f:
# #         for line in f.readlines():
# #             for word in line.split():
# #                 person = re.findall('person', word)
# #
# #                 if person:
# #                     # print(line)
# #                     i = 0
# #                     for word in line.split():
# #
# #                         if (i == 0):
# #                             label = word.lower().strip()
# #                             # print(label)
# #                         if (i == 1):
# #                             xmin = int(word) - 1
# #                             # print(x)
# #                         if (i == 2):
# #                             ymin = int(word) - 1
# #                             # print(y)
# #                         if (i == 3):
# #                             w = int(word) if int(word) > 0 else 1
# #                             xmax = xmin + w
# #                             # print(w)
# #                         if (i == 4):
# #                             h = int(word) if int(word) > 0 else 1
# #                             ymax = ymin + h
# #                         if (i == 5):
# #                             difficult = int(word)
# #                         i = i + 1
# #                             # print(dificualt)
# #
# #                     if xmin <= 0:
# #                         xmin = 0
# #                     if ymin <= 0:
# #                         ymin = 0
# #                     if xmax >= 639:
# #                         xmax = 639
# #                     if ymax >= 511:
# #                         ymax = 511
# #
# #                     if label == 'person?a':
# #                         label = 'person'
# #                     if label == 'person?':
# #                         label = 'person'
# #
# #                     labels.append(label_map[label])
# #                     boxes.append([xmin, ymin, xmax, ymax])
# #                     difficulties.append(difficult)
# #         if (boxes.__len__() == 0):
# #             boxes.append([0, 0, 1., 1.])
# #
# #             labels.append(label_map['nonperson'])
# #             difficulties.append(0)
# #     return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}
# # filename='/home/fereshteh/kaist/sanitized_annotations1/set00_V000_I01225.txt'
# # print(parse_annotation1(filename))
# # # print(boxes)
# #
# from utils import *
# data_folder = "/home/fereshteh/codergb_new"
# kaist_path='/home/fereshteh/kaist/day'
# create_data_lists(kaist_path, output_folder=data_folder)
# # '/home/fereshteh/kaist/day/rgb/I00000.png'
# # /home/fereshteh/kaist/day/rgb/I00000.png
import os

for filename in sorted(os.listdir('/home/fereshteh/kaist/sanitized_annotations')):
    print('')
