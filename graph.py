import netron
from torchvision.models import vit_b_16
import torch.nn as nn
import torch.onnx
import torch
from cnn import CNN

class VisionTransformer(nn.Module):
    def __init__(self, num_classes):
        super(VisionTransformer, self).__init__()
        self.vit = vit_b_16()
        self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)
    def forward(self, x):
        out = self.vit(x)
        return out

model = VisionTransformer(2)
model.load_state_dict(torch.load("model/vit.pth"))
model = CNN()
# model.load_state_dict(torch.load("model/vit.pth"))

# 保存模型为ONNX格式
dummy_input = torch.randn(32, 3, 224, 224)  # 创建一个虚拟输入
onnx_path = "model/cnn.pth"
torch.onnx.export(model, dummy_input, onnx_path)

# 启动Netron服务器并打开模型
netron.start(onnx_path)