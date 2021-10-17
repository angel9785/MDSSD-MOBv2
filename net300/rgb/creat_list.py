data_folder = "/home/fereshteh/codergb_new"
kaist_path="/home/fereshteh/kaist/day"
check="/home/fereshteh/code/SSD_MobileNet-master/voc_checkpoint_ssd300.pth.tar"
import torch
c=torch.load(check)
print(c["loss"])



