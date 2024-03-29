import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.models.vision_transformer import VisionTransformer, vit_b_16
import torchvision.models
from torch.utils.tensorboard import SummaryWriter
import numpy as np
# 设置随机种子
torch.manual_seed(42)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理和增强
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# 加载训练集和测试集
trainset = torchvision.datasets.ImageFolder(root="data/train", transform=transform)
testset = torchvision.datasets.ImageFolder(root="data/val", transform=transform)

# 定义数据加载器
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, )
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True,)

# 定义Vision Transformer模型
class VisionTransformer(nn.Module):
    def __init__(self, num_classes):
        super(VisionTransformer, self).__init__()
        self.vit = vit_b_16(weights='DEFAULT')
        self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)

        # 冻结除最后一层外的所有参数
        for param in self.vit.parameters():
            param.requires_grad = False

        # 最后一层的参数需要进行优化
        for param in self.vit.heads.head.parameters():
            param.requires_grad = True

    def forward(self, x):
        out = self.vit(x)
        return out
# 初始化模型
# 设置模型参数
num_classes = 2  # 分类的类别数（猫和狗）
#
# # 初始化模型
model = VisionTransformer(num_classes).to(device)
#
#
# 定义损失函数和优化器
model.train()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 训练模型
num_epochs = 30
train_loss_history = []
train_acc_history = []
for epoch in range(num_epochs):
    writer = SummaryWriter()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(trainloader):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计准确率
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        running_loss += loss.item()

    epoch_loss = running_loss / len(trainloader)
    epoch_acc = correct / total

    train_loss_history.append(epoch_loss)
    train_acc_history.append(epoch_acc)
    writer.add_scalar('loss/train', epoch_loss, epoch)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

# 保存模型
torch.save(model.state_dict(), "model/vit.pth")

# 测试模型
model = VisionTransformer(2).to(device)
model.load_state_dict(torch.load("model/vit.pth"))
model.eval()
test_acc = 0
with torch.no_grad():
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        test_acc += (predicted == labels).sum().item()

test_acc /= len(testset)
print(f"Test Accuracy: {test_acc:.4f}")

# #逆向
def denormalize(image):
    # mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
    # (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array( [0.229, 0.224, 0.225])
    image = image.numpy().transpose(1, 2, 0)
    image = std * image + mean
    image = np.clip(image, 0, 1)
    return image

# 可视化测试集图片和预测标签
test_batch = next(iter(testloader))
images, labels = test_batch
images = images[:16]
labels = labels[:16]

model.eval()
with torch.no_grad():
    outputs = model(images.to(device))
    _, predicted = torch.max(outputs.data, 1)

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
fig.suptitle(f"Test Accuracy: {test_acc}", fontsize=12)
for i, ax in enumerate(axes.flatten()):
    image = denormalize(images[i])
    ax.imshow(image)
    ax.set_title(f"Predicted: {predicted[i].item()}, Actual: {labels[i].item()}")
    ax.axis('off')

plt.tight_layout()
plt.show()
