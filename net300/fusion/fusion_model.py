import multiprocessing as mp
from multiprocessing import Process, Queue, Value
from lw_model import SSD_LW
from rgb_model import SSD_RGB
from utils import *
from torchvision import transforms
from mobilenet_ssd_priors import priors
from PIL import Image, ImageDraw, ImageFont
import cv2
# import Queue
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kaist_path="/home/fereshteh/kaist"
def setup():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    if torch.cuda.is_available():
        backend = 'nccl'

    # initialize the process group
    dist.init_process_group( backend='nccl', rank=1, world_size=2)
#
def cleanup():
    dist.destroy_process_group()

checkpoint_rgb =torch.load('/home/fereshteh/codergb_new/fine_checkpoint_ssd300_7.pth.tar')
checkpoint_lw =torch.load('/home/fereshteh/codelw_new/fine_checkpoint_ssd300_lw_3.pth.tar')
rank = 0
setup()
model_rgb = SSD_RGB(num_classes=2, backbone_network="MobileNetV1")
model_rgb = model_rgb.to(rank)
ddp_model_rgb = DDP(model_rgb)
ddp_model_rgb.load_state_dict(checkpoint_rgb['model'])
# model_rgb.load_state_dict(checkpoint_rgb['model'])
# ddp_model_rgb.load_state_dict(checkpoint_rgb['model'])
# model_rgb = model_rgb.to(device)
# ddp_model_rgb.eval()
# model_rgb.eval()

model_lw = SSD_LW(num_classes=2, backbone_network="MobileNetV1")
model_lw = model_lw.to(rank)

ddp_model_lw = DDP(model_lw)
ddp_model_lw.load_state_dict(checkpoint_lw['model'])
# ddp_model_lw.load_state_dict(checkpoint_lw['model'])
# model_lw = model_lw.to(device)
# model_lw.load_state_dict(checkpoint_lw['model'])
# ddp_model_lw.eval()
# model_lw.eval()
priors_cxcy = priors
priors_cxcy = priors_cxcy.to(device)
suppress=['nonperson']
# id=199
# image_rgb_path = os.path.join(kaist_path, 'sanitized', 'I' + '{0:05}'.format(id) + '.png')
# image_lw_path = os.path.join(kaist_path, 'sanitized_lw', 'I' + '{0:05}'.format(id) + '.png')
# image_rgb = Image.open(image_rgb_path, mode='r')
# image_lw = cv2.imread(image_lw_path, 1)

def rgb(image_rgb, predicted_locs_rgb,predicted_scores_rgb):
    # image_rgb.get()
    # image_rgb=list(image_rgb.Queue)
    # im_rgb = format(image_rgb, '05d')
    # im_rgb = image_rgb.format("05d")
    image_rgb_path = os.path.join(kaist_path, 'sanitized', 'I' +image_rgb.get()+ '.png')
    # # image_lw_path = os.path.join(kaist_path, 'sanitized_lw', 'I' + '{0:05}'.format(id) + '.png')
    image_rgb = Image.open(image_rgb_path, mode='r')
    # image_lw = cv2.imread(image_lw_path, 1)
    # image_rgb.get()
    image_rgb = image_rgb.convert('RGB')
    resize = transforms.Resize((300, 300))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    image = normalize(to_tensor(resize(image_rgb)))
    # image = (to_tensor(resize(original_image)))

    # Move to default device
    image_rgb = image.to(device)

    predicted_locs_r, predicted_scores_r = ddp_model_rgb(image_rgb.unsqueeze(0))
    # predicted_locs_r, predicted_scores_r = model_rgb(image_rgb.unsqueeze(0))
    predicted_locs_rgb.put(predicted_locs_r)
    predicted_scores_rgb.put(predicted_scores_r)
    cleanup()



def lw(image_lw, predicted_locs_lw,predicted_scores_lw):
    # image_lw.get()
    # im_lw = '05d'.format(image_lw)
    # image_lw=list(image_lw.Queue)
    image_lw_path = os.path.join(kaist_path, 'sanitized_lw', 'I' + image_lw.get()+ '.png')
    # # image_rgb = Image.open(image_rgb_path, mode='r')
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

    predicted_locs_l, predicted_scores_l = ddp_model_lw(image_lw.unsqueeze(0))
    # predicted_locs_l, predicted_scores_l = model_lw(image_lw.unsqueeze(0))
    predicted_locs_lw.put(predicted_locs_l)
    predicted_scores_lw.put(predicted_scores_l)
    cleanup()
id=199



if __name__ == '__main__':
    id = 199
    mp.set_start_method('spawn')
    predicted_locs_rgb = Queue()
    predicted_scores_rgb = Queue()
    predicted_locs_lw = Queue()
    predicted_scores_lw = Queue()
    image_rgb = Queue()
    image_lw = Queue()

    event = mp.Event()
    process_rgb = Process(target=rgb, args=(image_rgb,predicted_locs_rgb,predicted_scores_rgb))

    process_lw = Process(target=lw, args=(image_lw, predicted_locs_lw, predicted_scores_lw))
    process_rgb.start()
    process_lw.start()
    #
    image_rgb.put('{0:05}'.format(id))
    image_lw.put('{0:05}'.format(id))
    print(predicted_locs_rgb.get())
    print(predicted_scores_rgb.get())
    print(predicted_locs_lw.get())
    print(predicted_scores_lw.get())




    process_rgb.join()
    process_lw.join()


    # predicted_locs = sum(predicted_locs)
    # predicted_scores = sum(predicted_scores)
    # det_boxes, det_labels, det_scores = detect_objects(priors_cxcy, predicted_locs, predicted_scores, min_score=0.28,
    #                                                          max_overlap=0.5, top_k=200, n_classes=2)
    # det_boxes = det_boxes[0].to('cpu')
    #
    # # Transform to original image dimensions
    # original_dims = torch.FloatTensor(
    #     [image_rgb.width, image_rgb.height, image_rgb.width, image_rgb.height]).unsqueeze(0)
    # det_boxes = det_boxes * original_dims
    #
    # # Decode class integer labels
    # det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]
    #
    # # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    # if det_labels == ['nonperson']:
    #     # Just return original image
    #     image_rgb.show()
    #
    # # Annotate
    # annotated_image = image_rgb
    # draw = ImageDraw.Draw(annotated_image)
    #
    # font = ImageFont.load_default()
    #
    # # Suppress specific classes, if needed
    # for i in range(det_boxes.size(0)):
    #     if suppress is not None:
    #         if det_labels[i] in suppress:
    #             continue
    #
    #
    #     box_location = det_boxes[i].tolist()
    #     draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
    #     draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
    #         det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
    #
    #     # Text
    #     text_size = font.getsize(det_labels[i].upper())
    #     text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
    #     textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
    #                         box_location[1]]
    #     draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
    #     draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
    #               font=font)
    #     # print(det_scores[i])
    # del draw
    #
    # annotated_image.show()