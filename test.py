import torchvision
import torch
# from  torchvision.models
print(torchvision.__version__)
print(torch.__version__)
import torch
print(torch.__version__)
print(torch.version.cuda)
# print(torch.cuda.is_available())  #输出为True，则安装无误
# import torch
# from torchvision.models import VisionTransformer, vit_b_16
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # num_classes = 2  # 分类的类别数（猫和狗）
# # # model = vit_b_16(num_classes=num_classes).to(device)
# model = vit_b_16().to(device)
# #
# print(model)
# torch.cuda.empty_cache()
