# # import torch
# # import numpy as np
# # list1=[[[[4]],[[4]],[[2]]]]
# # list1=torch.tensor(list1)
# # print(list1.size())
# # h=[[[[3]],[[2]],[[9]]]]
# # list2=torch.tensor(h)
# # print(list2.size())
# #
# # list3=[]
# # # for i in range(1):
# # #     for j in range (512):
# # #         # for k in range (38):
# # #         for h in range(10):
# # #             for m in range(19):
# # #                 list3.append([m])
# # # for i in range(1):
# # #     for j in range (512):
# # #         # for k in range (38):
# # #         for h in range(171):
# # #             list3.append([i,h])
# # # y=torch.cat(list3, dim=0)
# # # list3=torch.tensor(list3)
# # # list3 = list3.view( 1,512,-1, 19)
# # # print(list3.size())
# # #
# # #     # list2=torch.ones(1, i,19,19)*2
# # from operator import add
# # y=list(map (add, list1, list2))
# # # # y=[t.size() for t in y]
# # y=torch.cat(y, dim=0)
# # # print(y.unsqueeze[0].size())
# # print(y.unsqueeze(0))
# # print(y.unsqueeze(0).size())
# # data_folder = "/home/fereshteh/codergb_new"
# # kaist_path="/home/fereshteh/kaist/day"
# check="./fine_checkpoint_ssd300_fusion2.pth.tar"
# import torch
# c=torch.load(check)
# print(c["epoch"])
#
#
#
# from second_model import SSD_FUSION
# import torch
# checkpoint_rgb = torch.load('/home/fereshteh/codergb_new/fine_checkpoint_ssd300_7.pth.tar')
# state_dict_RGB = checkpoint_rgb['model']
# checkpoint_lw = torch.load('/home/fereshteh/codelw_new/BEST_fine_checkpoint_ssd300_lw_3.pth.tar')
# state_dict_LW = checkpoint_lw['model']
# model = SSD_FUSION(num_classes=2, backbone_network="MobileNetV1")
# with torch.no_grad():
#     for param_name, param in model.LW_base_net.model.named_parameters():
#         param.copy_(state_dict_LW["LW_base_net."+param_name])
#
#     for param_name, param in model.RGB_base_net.model.named_parameters():
#         param.copy_(state_dict_RGB[ "RGB_base_net."+param_name])
#
#     for param_name, param in model.lw_aux_network.named_parameters():
#         param.copy_(state_dict_LW["LW_aux_network." + param_name])
#
#     for param_name, param in model.rgb_aux_network.named_parameters():
#         param.copy_(state_dict_RGB["RGB_aux_network." + param_name])
# with torch.no_grad():
#     for param_name, param in model.rgb_aux_network.named_parameters():
#         print(param)
#
import torch
a=[[[[3,5]],[[2,2]],[[9,3]]]]
b=[[[[4,2]],[[4,8]],[[2,7]]]]
a=torch.tensor(a)
b=torch.tensor(b)
c=a*b
#c=torch.add(a,b)
print(c)





