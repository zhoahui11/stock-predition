# import torch
# import torch.nn as nn
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#
# # 加载数据集
# df = pd.read_csv('比亚迪.csv')
#
# # 对收盘价进行归一化处理
# scaler = MinMaxScaler()
# df['close'] = scaler.fit_transform(df[['close']])
#
# # 定义函数将时间序列数据转换为监督学习数据
# def create_dataset(data, look_back=1, look_forward=1):
#     X, Y = [], []
#     for i in range(len(data)-look_back-look_forward+1):
#         X.append(data[i:(i+look_back)])
#         Y.append(data[(i+look_back):(i+look_back+look_forward)])
#     return np.array(X), np.array(Y)
#
# # 将数据集按4:1的比例划分为训练集和测试集
# train_size = int(len(df) * 0.8)
# test_size = len(df) - train_size
# train_data, test_data = df.iloc[0:train_size], df.iloc[train_size:len(df)]
#
# # 将数据集转换为监督学习数据
# look_back = 3
# look_forward = 3
# X_train, Y_train = create_dataset(train_data['close'].values, look_back, look_forward)
# X_test, Y_test = create_dataset(test_data['close'].values, look_back, look_forward)
#
# # 将数据集转换为张量
# X_train = torch.from_numpy(X_train).float()
# Y_train = torch.from_numpy(Y_train).float()
# X_test = torch.from_numpy(X_test).float()
# Y_test = torch.from_numpy(Y_test).float()
#
# # 构建CNN-LSTM模型
# class CNN_LSTM(nn.Module):
#     def __init__(self, look_back, look_forward):
#         super(CNN_LSTM, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
#         self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
#         self.fc = nn.Linear(64, look_forward)
#
#     def forward(self, x):
#         x = x.unsqueeze(1)
#         x = self.conv1(x)
#         x = nn.functional.relu(x)
#         x = self.conv2(x)
#         x = nn.functional.relu(x)
#         x = x.permute(0, 2, 1)
#         x, _ = self.lstm(x)
#         x = self.fc(x[:, -1, :])
#         return x
#
# # 训练模型
# model = CNN_LSTM(look_back, look_forward)
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# num_epochs = 100
# batch_size = 64
#
# for epoch in range(num_epochs):
#     for i in range(0, len(X_train), batch_size):
#         optimizer.zero_grad()
#         batch_X = X_train[i:i+batch_size]
#         batch_Y = Y_train[i:i+batch_size]
#         outputs = model(batch_X)
#         loss = criterion(outputs, batch_Y)
#         loss.backward()
#         optimizer.step()
#     if (epoch+1) % 10 == 0:
#         print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
#
# # 测试模型
# model.eval()
# with torch.no_grad():
#     test_outputs = model(X_test)
#     test_loss = criterion(test_outputs, Y_test)
#     print('Test Loss: {:.4f}'.format(test_loss.item()))
#
# # 对结果进行反归一化处理
# test_outputs = test_outputs.detach().numpy()
# Y_test = Y_test.numpy()
# test_outputs = scaler.inverse_transform(test_outputs)
# Y_test = scaler.inverse_transform(Y_test)
#
# # 计算评价指标
# mae = mean_absolute_error(Y_test, test_outputs)
# mape = np.mean(np.abs((Y_test - test_outputs) / Y_test)) * 100
# rmse = np.sqrt(mean_squared_error(Y_test, test_outputs))
# r2 = r2_score(Y_test, test_outputs)
# print('MAE: {:.4f}'.format(mae))
# print('MAPE: {:.4f}'.format(mape))
# print('RMSE: {:.4f}'.format(rmse))
# print('R²: {:.4f}'.format(r2))
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time  # 导入time模块

# 加载数据集
start_time = time.time()  # 开始计时
df = pd.read_csv('平安银行.csv')
load_time = time.time() - start_time  # 计算加载数据的时间

# 对收盘价进行归一化处理
scaler = MinMaxScaler()
df['close'] = scaler.fit_transform(df[['close']])

# 定义函数将时间序列数据转换为监督学习数据
def create_dataset(data, look_back=1, look_forward=1):
    X, Y = [], []
    for i in range(len(data)-look_back-look_forward+1):
        X.append(data[i:(i+look_back)])
        Y.append(data[(i+look_back):(i+look_back+look_forward)])
    return np.array(X), np.array(Y)

# 将数据集按4:1的比例划分为训练集和测试集
train_size = int(len(df) * 0.8)
test_size = len(df) - train_size
train_data, test_data = df.iloc[0:train_size], df.iloc[train_size:len(df)]

# 将数据集转换为监督学习数据
look_back = 3
look_forward = 3
X_train, Y_train = create_dataset(train_data['close'].values, look_back, look_forward)
X_test, Y_test = create_dataset(test_data['close'].values, look_back, look_forward)

# 将数据集转换为张量
X_train = torch.from_numpy(X_train).float()
Y_train = torch.from_numpy(Y_train).float()
X_test = torch.from_numpy(X_test).float()
Y_test = torch.from_numpy(Y_test).float()

# 构建CNN-LSTM模型
class CNN_LSTM(nn.Module):
    def __init__(self, look_back, look_forward):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, look_forward)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

# 训练模型
model = CNN_LSTM(look_back, look_forward)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100
batch_size = 64

start_time = time.time()  # 开始计时训练
for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        optimizer.zero_grad()
        batch_X = X_train[i:i+batch_size]
        batch_Y = Y_train[i:i+batch_size]
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
training_time = time.time() - start_time  # 计算训练时间

# 测试模型
start_time = time.time()  # 开始计时测试
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, Y_test)
    print('Test Loss: {:.4f}'.format(test_loss.item()))
testing_time = time.time() - start_time  # 计算测试时间

# 对结果进行反归一化处理
test_outputs = test_outputs.detach().numpy()
Y_test = Y_test.numpy()
test_outputs = scaler.inverse_transform(test_outputs)
Y_test = scaler.inverse_transform(Y_test)

# 计算评价指标
mae = mean_absolute_error(Y_test, test_outputs)
mape = np.mean(np.abs((Y_test - test_outputs) / Y_test)) * 100
rmse = np.sqrt(mean_squared_error(Y_test, test_outputs))
r2 = r2_score(Y_test, test_outputs)
print('MAE: {:.4f}'.format(mae))
print('MAPE: {:.4f}'.format(mape))
print('RMSE: {:.4f}'.format(rmse))
print('R²: {:.4f}'.format(r2))

# 输出运行时间和训练时间
print(f"Load Time: {load_time:.4f} seconds")
print(f"Training Time: {training_time:.4f} seconds")
print(f"Testing Time: {testing_time:.4f} seconds")