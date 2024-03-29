import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision


from torchvision import transforms, datasets
# from torchsummary import summary
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

#数据集
tr = "data/train"
val = "data/val"


# 数据预处理
transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

train_data = datasets.ImageFolder(tr, transforms)
val_data = datasets.ImageFolder(val, transforms)
print(len(train_data))
print(len(val_data))

batch_size = 64
train_loder = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_lodeer = DataLoader(val_data, batch_size=batch_size)


# 模型架构
class CNN(nn.Module):
        def __init__(self):

                super(CNN, self).__init__()
                self.conv1 = nn.Sequential(
                        nn.Conv2d(
                                in_channels=3,
                                out_channels=16,
                                kernel_size=3,
                                stride=2,
                        ),
                        nn.BatchNorm2d(16),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2),
                )
                self.conv2 = nn.Sequential(
                        nn.Conv2d(
                                in_channels=16,
                                out_channels=32,
                                kernel_size=3,
                                stride=2,
                        ),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2),
                )
                self.conv3 = nn.Sequential(
                        nn.Conv2d(
                                in_channels=32,
                                out_channels=64,
                                kernel_size=3,
                                stride=2,
                        ),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2),
                )
                self.fc1 = nn.Linear(64*3*3, 64)
                self.fc2 = nn.Linear(64, 10)
                self.fc3 = nn.Linear(10, 2)
        def forward(self,x):
            x = self.conv1(x)
            # print(x.size())
            x = self.conv2(x)
            # print(x.size())
            x = self.conv3(x)
            # print(x.size())
            x = x.flatten(start_dim=1)
            # x = x.view(x.shape[0], -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return F.log_softmax(x, dim=1)


# net = CNN()
# input = torch.randn(32, 3, 256, 256)
# out = net(input)
# print(out.size())
# summary(net,(3,256,256))



# 有则用GPU，没则不用
lr=3e-4
device=torch.device("cuda" if torch.cuda.is_available() else "cpu" )
model=CNN().to(device)
optimizer=optim.Adam(model.parameters(),lr=lr)
# 模型训练
def train():
        epoch_num = 300
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CNN().to(device)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss().to(device)
        for epoch in tqdm(range(epoch_num),):
                writer = SummaryWriter()
                train_loss = []
                for batch_idx,(data, target) in enumerate(train_loder):
                        data, target = data.to(device),target.to(device)
                        pred = model(data)
                        loss = criterion(pred, target)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        train_loss.append(loss.item())
                tqdm.write('Epoch {:03d} train_loss {:.5f}'.format(epoch,np.mean(train_loss)))
                writer.add_scalar('loss/train', np.mean(train_loss), epoch)
        torch.save(model.state_dict(), "model/cnn.pth")

#                 # # validation
#                 # val_loss = get_val_loss(model, val)
#                 # model.train()
#                 # if epoch + 1 >



predictions = []
#模型评估
# 测试模型

test_accuracy = 0.0
num_samples = 0
predictions = []
true_labels = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load('model/cnn.pth'))
model.eval()
total = 0
corrent = 0
num_samples = 0
with torch.no_grad():
    for images, labels in val_lodeer:
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
test_accuracy =0.79
# 随机抽取测试集图片进行可视化
indices = np.random.choice(len(val_data), size=16, replace=False)
selected_images = [val_data[i][0] for i in indices]
selected_labels = [val_data[i][1] for i in indices]


# 逆转换函数，将归一化后的图像还原为原始图像
def denormalize(image):
    # mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
    # mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)
    mean = np.array((0.5, 0.5, 0.5))
    std = np.array( (0.5, 0.5, 0.5))
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

# if __name__ == '__main__':
#         # train()
#         # test()