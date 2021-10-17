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


# Initialise cuda tensors here. E.g.:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kaist_path = "/home/fereshteh/kaist"

priors_cxcy = priors
priors_cxcy = priors_cxcy.to(device)
suppress = ['nonperson']
checkpoint_rgb = torch.load('/home/fereshteh/codergb_new/fine_checkpoint_ssd300_7.pth.tar')
checkpoint_lw = torch.load('/home/fereshteh/codelw_new/BEST_fine_checkpoint_ssd300_lw_3.pth.tar')
model_rgb = SSD_RGB(num_classes=2, backbone_network="MobileNetV1")
model_rgb = model_rgb.to(device)
# model_rgb = model_rgb.to(torch.s1)

model_rgb.load_state_dict(checkpoint_rgb['model'])
model_lw = SSD_LW(num_classes=2, backbone_network="MobileNetV1")
model_lw = model_lw.to(device)
# model_lw = model_lw.to(torch.s2)

model_lw.load_state_dict(checkpoint_lw['model'])
model_rgb.eval()
model_lw.eval()


# def rgb(image_rgb):
#     # image_rgb = image_rgb.to(device)
#     predicted_locs_rgb, predicted_scores_rgb = model_rgb(image_rgb)
#     return predicted_locs_rgb, predicted_scores_rgb
#
#
# def lw(image_lw):
#     # image_lw = image_lw.to(device)
#     predicted_locs_lw, predicted_scores_lw = model_lw(image_lw)
#     return predicted_locs_lw, predicted_scores_lw


kaist_path = '/home/fereshteh/kaist'
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
# test_dataset_rgb = KAISTdataset(data_folder,
#                             split='allsanitest',
#                             keep_difficult=keep_difficult)
# test_loader_rgb = torch.utils.data.DataLoader(test_dataset_rgb, batch_size=batch_size, shuffle=False,
#                                           collate_fn=test_dataset_rgb.collate_fn, num_workers=workers, pin_memory=True)
# test_dataset_lw = KAISTdataset(data_folder,
#                             split='allsanitest',
#                             keep_difficult=keep_difficult)
# test_loader_lw = torch.utils.data.DataLoader(test_dataset_lw, batch_size=batch_size, shuffle=False,
#                                           collate_fn=test_dataset_lw.collate_fn, num_workers=workers, pin_memory=True)
det_boxes = list()
det_labels = list()
det_scores = list()
true_boxes = list()
true_labels = list()
true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py
# fps = FPS().start()

def main(images_rgb, images_lw):

        # Batches


    # main_stream = torch.cuda.current_stream(device)

    # torch.cuda.synchronize(device)
    with torch.cuda.stream(s1):
        # s1.wait_stream(main_stream)
        # images_rgb = images_rgb.to(s1)
        images_rgb = images_rgb.to(device)
        predicted_locs_rgb, predicted_scores_rgb = model_rgb(images_rgb)
        # predicted_locs_rgb, predicted_scores_rgb = rgb(images_rgb)

    with torch.cuda.stream(s2):
        # images_lw = images_lw.to(s2)
        images_lw = images_lw.to(device)
        predicted_locs_lw, predicted_scores_lw = model_lw(images_lw)

    torch.cuda.synchronize(device)

    return predicted_locs_rgb, predicted_scores_rgb,predicted_locs_lw, predicted_scores_lw

s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()
with torch.no_grad():


    fps = FPS().start()
    for i, (images_rgb, images_lw, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):

        # images_rgb = images_rgb.to(device)
        # images_lw = images_lw.to(device)

        predicted_locs_rgb, predicted_scores_rgb, predicted_locs_lw, predicted_scores_lw = main(images_rgb, images_lw)

        fps.update()
        det_boxes1, det_labels1, det_scores1 = detect_objects1(priors_cxcy, predicted_locs_lw, predicted_locs_rgb,
                                                               predicted_scores_lw, predicted_scores_rgb,
                                                               min_score=0.28,
                                                               max_overlap=0.5, top_k=200, n_classes=2)





        # predicted_scores=torch.add(predicted_scores_lw,predicted_scores_rgb)


fps.stop()
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
