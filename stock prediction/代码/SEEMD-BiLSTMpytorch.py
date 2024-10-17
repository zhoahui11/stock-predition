import pandas as pd
import numpy as np
from PyEMD import EEMD
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from matplotlib import rcParams

# config = {
#     "font.family": 'serif',
#     "font.size": 10.5,
#     "mathtext.fontset": 'stix',
#     "font.serif": ['SimSun'],
# }
# rcParams.update(config)

# 读取文件
df = pd.read_csv('平安银行.csv')

# 假设您想要提取的列名为 'close_prices'
column_name = 'close'

# 获取某一列数据
column_data = df[column_name].values

cs = CubicSpline(np.arange(len(column_data)), column_data)

# 插值后的数据
interpolated_data = cs(np.arange(len(column_data)))

# 初始化EEMD
eemd = EEMD()

# 进行EEMD分解
imfs = eemd.eemd(interpolated_data)

#
# eemd = EEMD()
#
# # 对提取的列数据进行 EEMD 分解
# imfs = eemd.eemd(column_data)

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
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
        # self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        output = self.fc(hidden)
        # self.dropout = nn.Dropout(p=0.2)
        return output


# 设置模型参数
input_size = X_train.shape[1]
hidden_size = 64
output_size = 1

# 创建模型实例
model = BiLSTM(input_size, hidden_size, output_size)

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


# 计算评价指标
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test_true, y_test_pred)
mape = np.mean(np.abs((y_test_true - y_test_pred) / y_test_true)) * 100
rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
r2 = r2_score(y_test_true, y_test_pred)

print(mae)
print(mape)
print(rmse)
print(r2)

# plt.rcParams['font.serif']=['Times New Roman']

#中文
plt.rcParams['font.sans-serif'] = ['SimHei']     # 显示中文
# 为了坐标轴负号正常显示。matplotlib默认不支持中文，设置中文字体后，负号会显示异常。需要手动将坐标轴负号设为False才能正常显示负号。
plt.rcParams['axes.unicode_minus'] = False

# 绘制预测价格与真实价格的对比图
# plt.rcParams['font.serif']=['Times New Roman']
# plt.rcParams['font.family']=['Times New Roman']
plt.figure(figsize=(10, 6))
# plt.plot(y_train_true, label='Train True')
# plt.plot(y_train_pred, label='Train Predicted')
# plt.plot(len(y_train_true) + np.arange(len(y_test_true)), y_test_true, label='Test True',color='red')
plt.plot(np.arange(len(y_test_true)), y_test_true, label='Test True',color='#DDD305')#平安银行 #2401F7 比亚迪#3E848C  中国联通#908D01  上证#EC1A23   综指#DDD305   沪深#508AB2
# plt.plot( np.arange(len(y_test_true)), y_test_true, label='测试集真实数据',color='#908D01',linestyle=':')#平安银行 #2401F7 比亚迪#3E848C  中国联通#908D01
# plt.plot(len(y_train_true) + np.arange(len(y_test_true)), y_test_pred, label='Test Predicted',color='darkviolet')
plt.plot(np.arange(len(y_test_true)), y_test_pred, label='Test Predicted',color='#03CDA7')# 平安银行 #F20505 比亚迪#F29F05 中国联通#1A83B2 上证#00A651 综指#03CDA7 沪深#B36A6F
# plt.plot( np.arange(len(y_test_true)), y_test_pred, label='测试集预测数据',color='#1A83B2',linewidth=1)#deepskyblue royalblue 平安银行 #F20505 比亚迪#F29F05  中国联通#1A83B2
plt.legend()
# plt.xlabel('时间 / 日')
# plt.ylabel('价格 / 人民币')
plt.xlabel('Time / Day')
plt.ylabel('Price / CNY')
#plt.savefig('./C花旗银行.jpg',dpi=1000)
plt.show()