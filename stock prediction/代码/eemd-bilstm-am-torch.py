import pandas as pd
import numpy as np
from PyEMD import EEMD
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 读取文件
df = pd.read_csv('中国联通.csv')

# 假设您想要提取的列名为 'close_prices'
column_name = 'close'

# 获取某一列数据
column_data = df[column_name].values

# 创建 EEMD 实例
eemd = EEMD()

# 对提取的列数据进行 EEMD 分解
imfs = eemd.eemd(column_data)

# 创建一个空的列表用于保存分解后的各个结果
imfs_data = []

# 将分解结果转换为numpy数组并进行归一化
scaler = MinMaxScaler(feature_range=(0, 1))
for imf in imfs:
    imf_scaled = scaler.fit_transform(imf.reshape(-1, 1))
    imfs_data.append(imf_scaled.flatten())

# 将分解结果转换为适用于BiLSTM的输入数据格式
X = np.array(imfs_data).T

# 准备目标数据（即下一个时间步的原始信号值）
y = scaler.fit_transform(column_data.reshape(-1, 1))

# 划分训练集和测试集
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 将数据转换为PyTorch的Tensor类型
X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
y_train = torch.Tensor(y_train)
y_test = torch.Tensor(y_test)


# 创建自定义数据集类
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 创建数据加载器
train_dataset = MyDataset(X_train, y_train)
test_dataset = MyDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 创建BiLSTM模型
# 构建注意力机制的BiLSTM模型
class AttentionBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AttentionBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_size*2, 1)
        self.fc = nn.Linear(hidden_size*2, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        attention_weights = self.attention(out).squeeze(2)
        attention_weights = torch.softmax(attention_weights, dim=1)
        attention_out = torch.bmm(out.transpose(1, 2), attention_weights.unsqueeze(2)).squeeze(2)
        out = self.fc(attention_out)
        return out


# 设置模型参数
input_size = X_train.shape[1]
hidden_size = 64
output_size = 1

# 创建模型实例
model = AttentionBiLSTM(input_size, hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.unsqueeze(0))
        loss = criterion(outputs, targets.unsqueeze(0))
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    train_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}")

# 使用模型进行预测
model.eval()
y_train_pred = model(X_train.unsqueeze(0)).squeeze().detach().numpy()
y_test_pred = model(X_test.unsqueeze(0)).squeeze().detach().numpy()

# 反归一化预测结果
y_train_pred = scaler.inverse_transform(y_train_pred.reshape(-1, 1))
y_test_pred = scaler.inverse_transform(y_test_pred.reshape(-1, 1))
y_train_true = scaler.inverse_transform(y_train.numpy().reshape(-1, 1))
y_test_true = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))

#
# print(y_test_true)
# print(y_test_pred)

# 计算评价指标
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#
# mae = mean_absolute_error(y_test_true, y_test_pred)
# mape = np.mean(np.abs((y_test_true - y_test_pred) / y_test_true)) * 100
# rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
# r2 = r2_score(y_test_true, y_test_pred)
#
# print(mae)
# print(mape)
# print(rmse)
# print(r2)
#
# # 绘制预测价格与真实价格的对比图
# plt.figure(figsize=(10, 6))
# # plt.plot(y_train_true, label='Train True')
# # plt.plot(y_train_pred, label='Train Predicted')
# # plt.plot(len(y_train_true) + np.arange(len(y_test_true)), y_test_true, label='Test True',color='red')
# plt.plot( np.arange(len(y_test_true)), y_test_true, label='Test True',color='red')
# # plt.plot(len(y_train_true) + np.arange(len(y_test_true)), y_test_pred, label='Test Predicted',color='darkviolet')
# plt.plot( np.arange(len(y_test_true)), y_test_pred, label='Test Predicted',color='darkviolet')
# plt.legend()
# plt.xlabel('T i m e  / day')
# plt.ylabel('P r i c e  / yuan')
# plt.show()