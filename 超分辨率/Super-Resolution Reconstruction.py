"""
该代码为参数量较小的两种基于深度学习的超像素重建方法，SRCNN和ESPCN
区别是，SRCNN需要大量的数据先验，对数据量有一定的要求
      而ESPCN是一种像素重排列的方法，它不需要上采样，知识通过卷积初步重建像素
"""


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 超参数设置
batch_size = 64
learning_rate = 0.001
num_epochs = 100

# 数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 将图像转换为PyTorch张量
    transforms.Normalize((0.5,), (0.5,))  # 标准化图像
])

# 加载数据集
train_dataset = ImageFolder(root='dataset', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = ImageFolder(root='dataset', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# SRCNN模型
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# 实例化模型
model = SRCNN().to(device)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_loader):
        images = images.to(device)
        outputs = model(images)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')

# 测试模型
model.eval()
with torch.no_grad():
    total_loss = 0
    for images, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        loss = criterion(outputs, images)
        total_loss += loss.item()
    print(f'Test Loss: {total_loss/len(test_loader)}')

# 保存模型
torch.save(model.state_dict(), 'srcnn.pth')

"""
以下是另一个模型ESPCN的代码
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import YourDataset  # 你需要替换为你的数据集
from torchvision.transforms import ToTensor

class ESPCN(nn.Module):
    def __init__(self, upscale_factor):
        super(ESPCN, self).__init__()

        # 定义卷积层
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, upscale_factor ** 2, kernel_size=3, padding=1)

        # 定义子像素卷积层
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        return x

# 实例化模型，假设我们要将图像放大2倍
model = ESPCN(upscale_factor=2)
print(model)

train_dataset = YourDataset(...)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 实例化模型并转移到设备上
model = ESPCN(upscale_factor=2).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')

# 保存模型
torch.save(model.state_dict(), 'espcn_model.pth')
"""
"""
更多的关于espcn模型的代码，数据集等芳龄一个文件夹了
"""