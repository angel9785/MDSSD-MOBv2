import torch
def parse_rpnbf():

    im=1
    txt_boxes=list()
    txt_scores = list()
    txt_labels = list()
    import scipy.io
    mat = scipy.io.loadmat(
        '/home/fereshteh/MultispectralPedestrianDetection-master/CVPRW2017/CVPR2017_Koenig_FusionRPNBF_Detections_KAIST_Reasonable.mat')
    # print(mat["dt"][0][0][0])

    m=0
    for image in range (len(mat["dt"][0])):
        boxes = list()
        score = list()
        labels = list()
        # print(mat["dt"][0][image])
        for obj in range (len(mat["dt"][0][image])):
            # print(obj)

            for word in range (len(mat["dt"][0][image][obj])):

                # print(mat["dt"][0][image][obj])







                if (word == 0):
                    xmin = int(float(mat["dt"][0][image][obj][word]))/640
                    # print(x)
                if (word == 1):
                    ymin = int(float(mat["dt"][0][image][obj][word]))/512
                    # print(y)
                if (word == 2):
                    w =int(float(mat["dt"][0][image][obj][word])) if int(float(mat["dt"][0][image][obj][word])) > 0 else 1
                    xmax = (xmin + w/640)
                    # print(w)
                if (word == 3):
                    h = int(float(mat["dt"][0][image][obj][word])) if int(float(mat["dt"][0][image][obj][word])) > 0 else 1
                    ymax = (ymin + h/512)
                if (word == 4):
                    score1 = int(float(mat["dt"][0][image][obj][word]))



            boxes.append([xmin, ymin, xmax, ymax])
            score.append(score1)
            labels.append(1)
        txt_boxes.append(torch.FloatTensor(boxes))
        txt_scores.append(torch.LongTensor(score))
        txt_labels.append(torch.ByteTensor(labels))
    return  txt_boxes,txt_scores,txt_labels

# txt_objects=parse_cfr()
def parse_cfr():
    # filename="/home/fereshteh/code_513/fusion/CFR-master_2/CFR-master/CFR_Results/det-test-all.txt"
    # filename="/home/fereshteh/MultispectralPedestrianDetection-master/BMVC2020/BMVC2020_Wolpert_FusionCSPNet_Detections_KAIST_Reasonable.txt"
    filename="/home/fereshteh/detections_2/detections/IAF/det-test-all.txt"
    # filename="/home/fereshteh/Downloads/arcnn_dets_reasonable.txt"

    im=1
    txt_boxes=list()
    txt_scores = list()
    txt_labels = list()

    boxes = list()
    score = list()
    labels = list()
    m=0
    with open(filename) as f:
        boxes = list()
        score = list()
        labels = list()
        for line in f.readlines():
            i=0
            line = line.replace(',', '\t')
            for word in line.split():
                if (i == 0):
                    im_txt = int(float(word))
                if im_txt!=im:
                    txt_boxes.append(torch.FloatTensor(boxes))
                    txt_scores.append(torch.LongTensor(score))
                    txt_labels.append(torch.ByteTensor(labels))
                    m=m+1
                    print(m)
                    boxes = list()
                    score = list()
                    labels = list()
                    im = im_txt






                if (i == 1):
                    xmin = int(float(word))/640
                    # print(x)
                if (i == 2):
                    ymin = int(float(word))/512
                    # print(y)
                if (i == 3):
                    w =int(float(word)) if int(float(word)) > 0 else 1
                    xmax = (xmin + w/640)
                    # print(w)
                if (i == 4):
                    h = int(float(word)) if int(float(word)) > 0 else 1
                    ymax = (ymin + h/512)
                if (i == 5):
                    score1 = int(float(word)*100)
                i = i + 1


            boxes.append([xmin, ymin, xmax, ymax])
            score.append(score1)
            labels.append(1)
        txt_boxes.append(torch.FloatTensor(boxes))
        txt_scores.append(torch.LongTensor(score))
        txt_labels.append(torch.ByteTensor(labels))
    return  txt_boxes,txt_scores,txt_labels

def parse_msds():
    # filename1="/home/fereshteh/msds_anno3"
    filename1 = "/home/fereshteh/faster"

    # filename="/home/fereshteh/MultispectralPedestrianDetection-master/BMVC2020/BMVC2020_Wolpert_FusionCSPNet_Detections_KAIST_Reasonable.txt"
    txt_boxes = list()
    txt_scores = list()
    txt_labels = list()
    for file in sorted(os.listdir(filename1)):
        im=1


        boxes = list()
        score = list()
        labels = list()
        m=0
        file=("/home/fereshteh/faster/"+str(file))
        with open(file) as f:
            boxes = list()
            score = list()
            labels = list()
            for line in f.readlines():
                i=0
                # line = line.replace(',', '\t')
                for word in line.split():
                    # if (i == 0):
                    #     im_txt = int(float(word))
                    # if im_txt!=im:
                    #     txt_boxes.append(torch.FloatTensor(boxes))
                    #     txt_scores.append(torch.LongTensor(score))
                    #     txt_labels.append(torch.ByteTensor(labels))
                    #     m=m+1
                    #     print(m)
                    #     boxes = list()
                    #     score = list()
                    #     labels = list()
                    #     im = im_txt






                    if (i == 1):
                        xmin = int(float(word))/640
                        # print(x)
                    if (i == 2):
                        ymin = int(float(word))/512
                        # print(y)
                    if (i == 3):
                        w =int(float(word)) if int(float(word)) > 0 else 1
                        xmax = (xmin + w/640)
                        # print(w)
                    if (i == 4):
                        h = int(float(word)) if int(float(word)) > 0 else 1
                        ymax = (ymin + h/512)
                    if (i == 5):
                        score1 = int(float(word)*100)
                        print(score1)
                    i = i + 1


                boxes.append([xmin, ymin, xmax, ymax])
                score.append(score1)
                labels.append(1)
            # if (boxes.__len__()==0):
            #     boxes.append([0, 0, 0, 0])
            #     score.append(0)
            #     labels.append(0)
        txt_boxes.append(torch.FloatTensor(boxes))
        txt_scores.append(torch.LongTensor(score))
        txt_labels.append(torch.ByteTensor(labels))
    return  txt_boxes,txt_scores,txt_labels

from utils import *
from datasets import KAISTdataset
from tqdm import tqdm
from pprint import PrettyPrinter
from mobilenet_ssd_priors import priors
from priors3 import priors

# from mobilev2ssd import SSD
# from second_score import SSD_FUSION
# from secondmodel2 import SSD_FUSION
# from deconv_fusion_v3 import SSD_FUSION
# from f513_v3 import SSD_FUSION
from fusion_513_v2 import SSD_FUSION

# from second_model_cocat import SSD_FUSION
# from first_model2_sum import SSD_FUSION
# from first_model2_concat import SSD_FUSION
# from second_model_concat2 import SSD_FUSION
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib.pyplot as plt
# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()
priors_cxcy = priors
priors_cxcy = priors_cxcy.to(device)
# Parameters
data_folder = "/home/fereshteh/code_fusion"
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 1
workers = 1
device = torch.device( "cpu")
print(device)
# checkpoint = './second_sum_fine_checkpoint_ssd300_fusion3.pth.tar'
# checkpoint = './deconv_fine_checkpoint_ssd300_fusion4.pth.tar'
# checkpoint = './NEW19_fine_checkpoint_ssd300_fusion_8.pth.tar'
# checkpoint ="first_sum_fine_checkpoint_ssd300_fusion_2.pth.tar"
# checkpoint = './score_second_fine_checkpoint_ssd300_fusion8.pth.tar'
# # checkpoint = 'NEW_fine_checkpoint_ssd300_fusion2.pth.tar'

# create_data_lists(kaist_path, output_folder=data_folder)

# Load test data
test_dataset = KAISTdataset(data_folder,
                                split='sanitest1',
                                keep_difficult=keep_difficult)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)
# train_dataset = KAISTdataset(data_folder,
#                                 split='allsanitrain1',
#                                 keep_difficult=keep_difficult)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
#                                               collate_fn=train_dataset.collate_fn, num_workers=workers, pin_memory=True)

# with open(os.path.join(data_folder, 'DAY' + '_images.json'), 'r') as j:
#     images = json.load(j)
# with open(os.path.join(data_folder, 'DAY' + '_objects.json'), 'r') as j:
#     objects = json.load(j)
# with open(os.path.join(data_folder, 'SANITEST' + '_images.json'), 'r') as j:
#     images1 = json.load(j)
# with open(os.path.join(data_folder, 'SANITEST' + '_objects.json'), 'r') as j:
#     objects1 = json.load(j)
# l=0
# l1=0
# for i in range(len(images)):
#     l = l + len((objects[i]['labels']))
# for i in range(len(images1)):
#     l1 = l1 + len((objects[i]['labels']))
# print(l)
# print(l1)
kaist_path=""
# line1=create_data_lists(kaist_path, output_folder=data_folder)
# print(line1)
def evaluate(test_loader):
    """
    Evaluate.
    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils2.pil
    bir=list()
    bil=list()
    with torch.no_grad():
        # Batches
        for i, (im,images_rgb,images_lw, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):

            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]


            true_boxes.extend(boxes)
            true_labels.extend(labels)

            true_difficulties.extend(difficulties)
            # bir.extend(ir)
            bir.extend(im)
            # bil.extend(il)
        # Calculate mAP
        h = 0
        # for i in range(boxes.__len__()):
        txt_boxes, txt_scores,txt_labels=parse_rpnbf()
        APs, mAP, precision,recall,f ,n_easy_class_objects,true_positives,false_positives,lamr,ap= calculate_mAP_kaist(txt_boxes, txt_labels, txt_scores, true_boxes, true_labels, true_difficulties)
        h=h=1
    # Print AP for each class
    # pp.pprint('AP% .3f' % APs)
    print('AP', APs)
    # pp.pprint("precision% .3f" % precision)
    print("precision", precision)
    # pp.pprint("recall% .3f" % recall)
    print("recall", recall)
    print("n_easy_class_objects", n_easy_class_objects)
    print("true_positives", true_positives)
    print("false_positives", false_positives)
    f

    print('\nMean Average Precision (mAP): %.3f' % mAP)

    print(lamr,"lamr")
    print(ap, "ap")
if __name__ == '__main__':
    evaluate(test_loader)
    # txt_boxes, txt_scores, txt_labels = parse_rpnbf()

# txt_boxes, txt_scores,txt_labels=parse_rpnbf()

