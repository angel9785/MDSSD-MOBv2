import torch
import time
from lw_model import SSD_LW
from rgb_model import SSD_RGB
from utils import *
from torchvision import transforms
from mobilenet_ssd_priors import priors
from PIL import Image, ImageDraw, ImageFont
import cv2
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import time
from imutils.video import FPS

s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()
# Initialise cuda tensors here. E.g.:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kaist_path="/home/fereshteh/kaist"
sanitest_rgb = '/home/fereshteh/kaist/sanitest'
sanitest_lw = '/home/fereshteh/kaist/sanitest_lw'



priors_cxcy = priors
priors_cxcy = priors_cxcy.to(device)
suppress=['nonperson']

def rgb(image_rgb):
    checkpoint_rgb = torch.load('/home/fereshteh/codergb_new/fine_checkpoint_ssd300_7.pth.tar')
    model_rgb = SSD_RGB(num_classes=2, backbone_network="MobileNetV1")
    model_rgb = model_rgb.to(device)
    model_rgb.load_state_dict(checkpoint_rgb['model'])
    model_rgb.eval()
    # image_rgb_path = os.path.join(kaist_path, 'sanitized', 'I' + '{0:05}'.format(id) + '.png')
    # image_rgb = Image.open(image_rgb_path, mode='r')
    image_rgb = image_rgb.convert('RGB')
    resize = transforms.Resize((300, 300))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    image = normalize(to_tensor(resize(image_rgb)))
    image_rgb = image.to(device)
    predicted_locs_rgb, predicted_scores_rgb = model_rgb(image_rgb.unsqueeze(0))
    return predicted_locs_rgb, predicted_scores_rgb

def lw(filename):
    checkpoint_lw =torch.load('/home/fereshteh/codelw_new/fine_checkpoint_ssd300_lw_3.pth.tar')
    model_lw = SSD_LW(num_classes=2, backbone_network="MobileNetV1")
    model_lw = model_lw.to(device)
    model_lw.load_state_dict(checkpoint_lw['model'])
    model_lw.eval()
    # image_lw_path = os.path.join(kaist_path, 'sanitest_lw', 'I' + '{0:05}'.format(id) + '.png')
    image_lw_path = os.path.join(sanitest_lw,filename)
    image_lw = cv2.imread(image_lw_path, 1)
    lab = cv2.cvtColor(image_lw, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    original_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    color_coverted = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = Image.fromarray(color_coverted)
    image_lw = original_image.convert('RGB')
    resize = transforms.Resize((300, 300))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    image1 = normalize(to_tensor(resize(image_lw)))
    # Move to default device
    image_lw = image1.to(device)
    predicted_locs_lw, predicted_scores_lw = model_lw(image_lw.unsqueeze(0))
    return predicted_locs_lw, predicted_scores_lw
# # Wait for the above tensors to initialise.
# for i in range(0,5):
i=0
filename='set06_V001_I00439.png'
fps=FPS().start()
# image_rgb_path = os.path.join(kaist_path, 'sanitest', 'I' + '{0:05}'.format(id) + '.png')
image_rgb_path = os.path.join(sanitest_rgb, filename)
image_rgb = Image.open(image_rgb_path, mode='r')
start = time.time()
torch.cuda.synchronize()
with torch.cuda.stream(s1):
    start1 = time.time()
    print(start1-start)
    # i=i+10
    predicted_locs_rgb, predicted_scores_rgb = rgb(image_rgb)
    fps.update()


    start3 = time.time()
    print(start3-start1 )

with torch.cuda.stream(s2):
    start2 = time.time()
    print(start2 - start)
    # i=i+10
    predicted_locs_lw, predicted_scores_lw = lw(filename)
    fps.update()
    start4 = time.time()
    print(start4-start2)

torch.cuda.synchronize()
start5 = time.time()
print(start5-start)
# predicted_locs=predicted_locs_lw+predicted_locs_rgb
# predicted_scores=predicted_scores_lw+predicted_scores_rgb

# predicted_locs=torch.cat((predicted_locs_lw, predicted_locs_rgb), 0)
# predicted_locs=predicted_locs.permute(1, 0, 2).contiguous()
# predicted_locs=predicted_locs.to('cpu')
# predicted_scores=torch.cat((predicted_scores_lw, predicted_scores_rgb), 0)
# # predicted_scores=predicted_scores.permute(1, 0, 2).contiguous()
# # predicted_scores=predicted_scores.to('cpu')
# n_in=predicted_locs.size(1)
# n_priors=priors_cxcy.size(0)
# layer = torch.nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1, padding=0)
# layer1 = torch.nn.Conv1d(in_channels=4, out_channels=2, kernel_size=1, padding=0)
# # m=torch.nn.Conv1d(n_in, n_priors, 1, stride=1)
# predicted_locs1=layer(predicted_locs)
# # predicted_scores1=layer(predicted_scores)
#
# predicted_locs1=predicted_locs1.permute(1, 0, 2).contiguous().to(device)
# predicted_scores1=predicted_scores1.permute(1, 0, 2).contiguous().to(device)
# predicted_scores1=predicted_locs_lw+ predicted_locs_rgb
predicted_scores1=torch.add(predicted_scores_lw,predicted_scores_rgb)
# predicted_locs1=torch.add(predicted_locs_lw,predicted_locs_rgb)
det_boxes, det_labels, det_scores = detect_objects1(priors_cxcy, predicted_locs_lw, predicted_locs_rgb, predicted_scores_lw, predicted_scores_rgb,predicted_scores1, min_score=0.28,
                                                         max_overlap=0.5, top_k=200, n_classes=2)
# det_boxes_lw, det_labels_lw, det_scores_lw = detect_objects(priors_cxcy, predicted_locs_lw, predicted_scores_lw, min_score=0.28,
#                                                          max_overlap=0.5, top_k=200, n_classes=2)

# det_boxes =[det_boxes_lw[i]+det_boxes_rgb[i] for i in range(max[len(det_boxes_lw),len(det_boxes_rgb)]
# # det_labels= det_labels_rgb
# # det_scores=det_scores_rgb
# det_labels= det_labels_lw+ det_scores_rgb for i in range(len(det_boxes_lw))]
# det_scores=det_scores_lw + det_scores_rgb for i in range(len(det_boxes_lw))]
# det_boxes = det_boxes[0].to('cpu')
# print(det_scores)
det_boxes = det_boxes[0].to('cpu')
det_scores = det_scores[0].to('cpu').tolist()
# Transform to original image dimensions
original_dims = torch.FloatTensor(
    [image_rgb.width, image_rgb.height, image_rgb.width, image_rgb.height]).unsqueeze(0)
det_boxes = det_boxes * original_dims

# Decode class integer labels
det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

# If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
if det_labels == ['nonperson']:
    # Just return original image
    image_rgb.show()

else:
# Annotate
    annotated_image = image_rgb
    draw = ImageDraw.Draw(annotated_image)

    font = ImageFont.load_default()

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue


        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness

        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        text_size1 = font.getsize(det_scores[i])
        text_location1 = [box_location[0] + 2., box_location[3]]
        # textbox_location1 = [box_location[0], box_location[3] + text_size1[1], box_location[0] + text_size1[0]+8.,
        #                     box_location[3]]
        # draw.rectangle(xy=textbox_location1, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location1, text="{:.2f}".format(det_scores[i]), fill='white',
                  font=font)
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                  font=font)
        print(det_scores)
    del draw

    annotated_image.show()
fps.stop()
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))