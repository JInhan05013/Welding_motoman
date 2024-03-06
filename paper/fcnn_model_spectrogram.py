import torch, pickle
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import librosa
import matplotlib.pyplot as plt
from torch.utils.data.dataset import random_split

class SpectrogramDataset(Dataset):
    def __init__(self, layers_data):
        self.data = []
        for layer_name, layer_data in layers_data.items():
            for segment in layer_data['segments']:
                # 此处假设你已经有频谱图S_db作为输入
                self.data.append((segment['spectrogram'], np.mean(segment['height_segment'])))  # 假设每个段有一个标签

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        spectrogram, label = self.data[idx]
        return torch.tensor(spectrogram, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # 假设输入是单通道频谱图
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(64, 1)  # 假设是回归问题，调整输出大小为需要的维度

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.mean(x, dim=(2, 3))  # 全局平均池化
        x = self.fc(x)
        return x

# class FCNN(nn.Module):
#     def __init__(self):
#         super(FCNN, self).__init__()
#         # 增加卷积层
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         # 新增加的卷积层
#         self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

#         # 全连接层也可以根据需要添加更多
#         self.fc1 = nn.Linear(256, 128)  # 注意，这里的输入特征数要与conv5的输出通道数一致
#         self.fc2 = nn.Linear(128, 1)    # 最后一个全连接层的输出维度取决于问题的需求，这里假设为1

#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = torch.relu(self.conv2(x))
#         x = torch.relu(self.conv3(x))
#         x = torch.relu(self.conv4(x))
#         x = torch.relu(self.conv5(x))
#         # 假设使用全局平均池化，这里我们取每个特征图的平均值
#         x = torch.mean(x, dim=(2, 3))  
#         # 连接全连接层
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# 加载数据
# 假设 'layers_data' 是之前从pickle文件中加载的数据
with open('Test_data/ER4043_model_150ipm_2023_10_08_08_23_41_with_centroids.pkl', 'rb') as file:
    layers_data = pickle.load(file)
dataset = SpectrogramDataset(layers_data)
# 确定数据集的总大小
dataset_size = len(dataset)
# 设置训练集和验证集的比例为4:1
train_size = int(dataset_size * 0.8)  # 80%数据用于训练
val_size = dataset_size - train_size  # 剩余20%数据用于验证

# 随机分割数据集为训练集和验证集
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 为训练集和验证集创建DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# 初始化模型和优化器
model = FCNN()
optimizer = optim.Adam(model.parameters(), lr=0.000001)

# 假设 dataloader 和 val_dataloader 分别是训练和验证的 DataLoader
val_losses = []
num_epochs = 20  # 或根据需要调整
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    train_loss = 0
    for inputs, labels in train_dataloader:
        optimizer.zero_grad()  # 清除梯度
        inputs = inputs.unsqueeze(1)  # 增加一个通道维度，如果你的数据需要
        outputs = model(inputs)
        loss = nn.MSELoss()(outputs, labels)
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
        train_loss += loss.item()
    train_loss /= len(train_dataloader)
    print(f'Epoch {epoch+1}, Training Loss: {train_loss}')
    # 训练模型...
    model.eval()  # 设置模型为评估模式
    val_loss = 0
    with torch.no_grad():  # 在评估阶段不计算梯度
        for inputs, labels in val_dataloader:
            inputs = inputs.unsqueeze(1)  # 增加一个通道维度
            outputs = model(inputs)
            loss = nn.MSELoss()(outputs, labels)
            val_loss += loss.item()
    val_loss /= len(val_dataloader)
    print(f'Epoch {epoch+1}, Validation Loss: {val_loss}')
    val_losses.append(val_loss)

plt.figure(figsize=(10, 6))
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss Across Epochs')
plt.legend()
plt.show()

# 假设你有一个测试集 test_dataloader
y_true = []
y_pred = []
model.eval()
with torch.no_grad():
    for inputs, labels in val_dataloader:
        inputs = inputs.unsqueeze(1)  # 增加一个通道维度
        outputs = model(inputs)
        y_true.extend(labels.numpy())
        y_pred.extend(outputs.view(-1).numpy())

# 可视化实际值和预测值
plt.figure(figsize=(10, 6))
plt.plot(y_true, 'o-', color='black', label='Actual Value')
plt.plot(y_pred, 'o-', color='red', label='Predicted Value')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Comparison of Actual and Predicted Values')
plt.legend()
plt.show()