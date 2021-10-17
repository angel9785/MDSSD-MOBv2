import torch
import time
from lw_model import SSD_LW
from rgb_model import SSD_RGB
from utils import *
from torchvision import transforms
from mobilenet_ssd_priors import priors
from PIL import Image, ImageDraw, ImageFont
import cv2
from datasets import KAISTdataset
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import time
from imutils.video import FPS

s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()
# Initialise cuda tensors here. E.g.:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kaist_path="/home/fereshteh/kaist"



priors_cxcy = priors
priors_cxcy = priors_cxcy.to(device)
suppress=['nonperson']
checkpoint_rgb = torch.load('/home/fereshteh/codergb_new/fine_checkpoint_ssd300_7.pth.tar')
checkpoint_lw = torch.load('/home/fereshteh/codelw_new/BEST_fine_checkpoint_ssd300_lw_3.pth.tar')
model_rgb = SSD_RGB(num_classes=2, backbone_network="MobileNetV1")
model_rgb = model_rgb.to(device)
model_rgb.load_state_dict(checkpoint_rgb['model'])
model_lw = SSD_LW(num_classes=2, backbone_network="MobileNetV1")
model_lw = model_lw.to(device)
model_lw.load_state_dict(checkpoint_lw['model'])
model_rgb.eval()
model_lw.eval()
def rgb(image_rgb):



    # image_rgb = image_rgb.to(device)
    predicted_locs_rgb, predicted_scores_rgb = model_rgb(image_rgb)
    return predicted_locs_rgb, predicted_scores_rgb

def lw(image_lw):



    # image_lw = image_lw.to(device)
    predicted_locs_lw, predicted_scores_lw = model_lw(image_lw)
    return predicted_locs_lw, predicted_scores_lw

kaist_path='/home/fereshteh/kaist'
data_folder = "/home/fereshteh/code_fusion"
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 1
workers = 2
# create_data_lists(kaist_path, output_folder=data_folder)
# Load test data
test_dataset = KAISTdataset(data_folder,
			                           split='allsanitest',
			                             keep_difficult=keep_difficult)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)

det_boxes = list()
det_labels = list()
det_scores = list()
true_boxes = list()
true_labels = list()
true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py
# fps = FPS().start()
fps = FPS().start()
with torch.no_grad():
    # Batches
    for i, (images_rgb,images_lw, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
        main_stream = torch.cuda.current_stream(device)
        torch.cuda.synchronize()
        with torch.cuda.stream(s1):
            images_rgb = images_rgb.to(device)

            predicted_locs_rgb, predicted_scores_rgb = rgb(images_rgb)


        with torch.cuda.stream(s2):
            images_lw = images_lw.to(device)

            predicted_locs_lw, predicted_scores_lw = lw(images_lw)


        torch.cuda.synchronize()
        fps.update()
        # predicted_scores=torch.add(predicted_scores_lw,predicted_scores_rgb)

        det_boxes1, det_labels1, det_scores1 = detect_objects1(priors_cxcy, predicted_locs_lw, predicted_locs_rgb, predicted_scores_lw, predicted_scores_rgb, min_score=0.2,
                                                                 max_overlap=0.5, top_k=200, n_classes=2)
fps.stop()
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
#         boxes = [b.to(device) for b in boxes]
#         labels = [l.to(device) for l in labels]
#         difficulties = [d.to(device) for d in difficulties]
#
#         det_boxes.extend(det_boxes1)
#         det_labels.extend(det_labels1)
#         det_scores.extend(det_scores1)
#         true_boxes.extend(boxes)
#         true_labels.extend(labels)
#         true_difficulties.extend(difficulties)
#
#     APs, mAP, precision,recall,f ,n_easy_class_objects,true_positives,false_positives= calculate_mAP1(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)
#
#
# print('AP', APs)
# print("precision", precision)
# print("recall", recall)
# print("n_easy_class_objects", n_easy_class_objects)
# print("true_positives", true_positives)
# print("false_positives", false_positives)
# f
