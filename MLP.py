import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置随机种子
torch.manual_seed(42)



# 定义超参数
batch_size = 32
learning_rate = 0.0001
num_epochs = 50

# 数据预处理与加载
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# transform_test = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

train_dataset = torchvision.datasets.ImageFolder("data/train", transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torchvision.datasets.ImageFolder("data/val", transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# MLP模型定义
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(224*224*3, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 64)
        self.fc5 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        # self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入图像
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        # x = self.sigmoid(x)
        x = self.softmax(x)
        return x



model = MLP().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
train_losses = []

for epoch in range(num_epochs):
    writer = SummaryWriter()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        # 将数据移动到GPU
        images = images.to(device)
        labels = labels.to(device)
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # 计算平均损失
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    writer.add_scalar('loss/train',avg_loss, epoch)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss}")

# 保存训练的模型
torch.save(model.state_dict(), 'model/mlp.pth')

# 测试模型
model.eval()
test_accuracy = 0.0
num_samples = 0
predictions = []
true_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_accuracy += (predicted == labels).sum().item()
        num_samples += labels.size(0)
        predictions.extend(predicted.tolist())
        true_labels.extend(labels.tolist())

test_accuracy = test_accuracy / num_samples
print(f"Test Accuracy: {test_accuracy}")

# 随机抽取测试集图片进行可视化
indices = np.random.choice(len(test_dataset), size=16, replace=False)
selected_images = [test_dataset[i][0] for i in indices]
selected_labels = [test_dataset[i][1] for i in indices]


# 逆转换函数，将归一化后的图像还原为原始图像
def denormalize(image):
    # mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array( [0.229, 0.224, 0.225])
    image = image.numpy().transpose(1, 2, 0)
    image = std * image + mean
    image = np.clip(image, 0, 1)
    return image


fig, axes = plt.subplots(4, 4, figsize=(10, 10))
fig.suptitle(f"Test Accuracy: {test_accuracy}", fontsize=12)

for i, ax in enumerate(axes.flatten()):
    image = denormalize(selected_images[i])
    true_label = selected_labels[i]
    predicted_label = predictions[indices[i]]

    ax.imshow(image)
    ax.axis('off')
    ax.set_title(f"True: {true_label}, Predicted: {predicted_label}")

plt.tight_layout()
plt.show()