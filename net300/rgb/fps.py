from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
import torch
import argparse
from mobilenet_ssd_priors import priors
import torch.nn.functional as F
from utils import detect_objects
import numpy as np
from imutils.video import FPS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
priors_cxcy = priors
priors_cxcy = priors_cxcy.to(device)
n_classes = 3



fps=FPS().start()
def detect(model,  min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.
    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transforms
    resize = transforms.Resize((300, 300))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    
    # id=500
    path="/home/fereshteh/kaist"  
    tp=0
    gt=0
    dt=0 
    idd=0 
    for id in range(0,45139):
        idd=idd+1 
        print(idd)     
        # address = os.path.join(path, 'testanno', 'I'+  '{0:05}'.format(id) + '.xml')
        
        imagepath=os.path.join(path, 'test', 'I'+ '{0:05}'.format(id)+ '.png')
    # img_path = '/home/fereshteh/kaist/person/I00650.png'
        # tree = ET.parse(address)
        # root = tree.getroot()
        original_image = Image.open(imagepath, mode='r')
        original_image = original_image.convert('RGB')
        image = normalize(to_tensor(resize(original_image)))
    #image = (to_tensor(resize(original_image)))

    # Move to default device
        image = image.to(device)

    # Forward prop.
        predicted_locs, predicted_scores = model(image.unsqueeze(0))
        fps.update()
    # Detect objects in SSD output
        det_boxes, det_labels, det_scores = detect_objects(model, priors_cxcy, predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k, n_classes=n_classes)

    # Move detections to the CPU
    #     det_boxes = det_boxes[0].to('cpu')
    #
    # # Transform to original image dimensions
    #     original_dims = torch.FloatTensor(
    #         [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    #     det_boxes = det_boxes * original_dims
    #
    # # Decode class integer labels
    #     det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]
    #
    # # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    # #     if det_labels == ['background','nonperson']:
    # #     # Just return original image
    # #         return original_image
    #
    # # # Annotate
    # #     annotated_image = original_image
    # #     draw = ImageDraw.Draw(annotated_image)
    #
    # #     font = ImageFont.load_default()
    #
    # # Suppress specific classes, if needed
    #     for i in range(det_boxes.size(0)):
    #
    #         if suppress is not None:
    #             if det_labels[i] in suppress:
    #                 continue
    #             # else:
    #             #     dt=dt+1
    #
    #
    #     # Boxes
    #         # box_location = det_boxes[i].tolist()
    #         # xmind=box_location[0]
    #         # xmaxd=box_location[2]
    #         # ymind=box_location[1]
    #         # ymaxd=box_location[3]
    #     #     draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
    #     #     draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
    #     #                 det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
    #
    #
    #     # # Text
    #     #     text_size = font.getsize(det_labels[i].upper())
    #     #     text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
    #     #     textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
    #     #                          box_location[1]]
    #     #     draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
    #     #     draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
    #     #              font=font)
    #
    # #         for object in root.iter('object'):
    # #             if (i==0):
    # #                 gt=gt+1
    # #             difficult = int(object.find('difficult').text == '1')
    #
    # #             label = object.find('name').text.lower().strip()
    #
    #
    # #             bbox = object.find('bndbox')
    #
    # #             xmin = int(bbox.find('x').text)-1
    # #             ymin = int(bbox.find('y').text)-1
    # #             w=int(bbox.find('w').text) if int(bbox.find('w').text)>0 else 1
    # #             xmax = xmin+w
    # #             h=int(bbox.find('h').text) if int(bbox.find('h').text)>0 else 1
    # #             ymax = ymin+h
    # #             w_intsec = np.maximum (0, (np.minimum(xmaxd, xmax) - np.maximum(xmind, xmin)))
    # #             h_intsec = np.maximum (0, (np.minimum(ymaxd, ymax) - np.maximum(ymin, ymind)))
    # #             wd=xmaxd-xmind
    # #             hd=ymaxd-ymind
    # #             wt=xmax-xmin
    # #             ht=ymax-ymin
    # #             s_intsec = w_intsec * h_intsec
    # #             sd=wd*hd
    # #             st=wt*ht
    # #             iou=s_intsec/(st+sd-s_intsec)
    # #             # print(iou)
    # #             if iou>0.5:
    # #                 tp=tp+1
    #
    # # recall=0
    # # if(gt!=0):
    # #     recall=tp/gt
    # # precision=0
    # # if(dt!=0):
    # #     precision=tp/dt
    # # f1=0
    # # if(tp!=0):
    # #     f1=2*(precision*recall)/(precision+recall)
    # # print(recall)
    # # print(precision)
    # # print(f1)
    # # print(idd)
    # # print(gt)
    # # del draw
    #
    #
    # # return annotated_image

checkpoint = torch.load('BEST_checkpoint_ssd300.pth.tar')
def main(checkpoint):
	
    #img_path = args.img_path
    
    #img_path = '/home/fereshteh/kaist/set00_V000/images/set00/V000/visible/I00000.jpg'
    # original_image = Image.open(img_path, mode='r')
    # original_image = original_image.convert('RGB')
    # Load model checkpoint
    #checkpoint = args.checkpoint
    #checkpoint = torch.load('BEST_checkpoint_ssd300.pth.tar')
    #checkpoint = torch.load(checkpoint, map_location='cpu')
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['best_loss']
    print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, best_loss))
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    detect(model,  min_score=0.2, max_overlap=0.5, top_k=200, suppress=['background','nonperson'])
    
if __name__ == '__main__':
    main(checkpoint)
    fps.stop()
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
  #detect(model, original_image, min_score=0.2, max_overlap=0.5, top_k=200).show()
  #  parser.add_argument('img_path',help='Image path')
   # parser.add_argument('checkpoint',help='Path for pretrained model')
    #args = parser.parse_args()
    
    #main(args)
    	