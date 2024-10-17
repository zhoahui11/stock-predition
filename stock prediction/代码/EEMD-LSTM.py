# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from PyEMD import EEMD
# import torch
# import torch.nn as nn
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#
# # 读取CSV文件
# data = pd.read_csv('平安银行.csv')
#
# # 提取需要分解的列数据
# signal = data['close'].values
#
# # 初始化EEMD算法
# eemd = EEMD()
# eIMFs = eemd.eemd(signal)
# nIMFs = eIMFs.shape[0]
#
# # 准备数据集
# window_size = 10
# X, y = [], []
#
# for i in range(nIMFs):
#     for j in range(len(eIMFs[i])-window_size):
#         X.append(eIMFs[i][j:j+window_size])
#         y.append(eIMFs[i][j+window_size])
#
# X = np.array(X)
# y = np.array(y)
#
# # 归一化数据
# scaler = MinMaxScaler(feature_range=(0, 1))
# X = scaler.fit_transform(X)
# y = scaler.fit_transform(y.reshape(-1, 1))
#
# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 转换为PyTorch张量
# X_train = torch.from_numpy(X_train).float()
# X_test = torch.from_numpy(X_test).float()
# y_train = torch.from_numpy(y_train).float()
# y_test = torch.from_numpy(y_test).float()
#
# # 定义LSTM模型
# class LSTM(nn.Module):
#     def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
#         super().__init__()
#         self.hidden_layer_size = hidden_layer_size
#         self.lstm = nn.LSTM(input_size, hidden_layer_size)
#         self.linear = nn.Linear(hidden_layer_size, output_size)
#         self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
#                             torch.zeros(1,1,self.hidden_layer_size))
#
#     def forward(self, input_seq):
#         lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
#         predictions = self.linear(lstm_out.view(len(input_seq), -1))
#         return predictions[-1]
#
# model = LSTM()
# loss_function = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
# # 训练模型
# epochs = 150
# for i in range(epochs):
#     for seq, labels in zip(X_train, y_train):
#         optimizer.zero_grad()
#         model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
#                         torch.zeros(1, 1, model.hidden_layer_size))
#
#         y_pred = model(seq)
#
#         single_loss = loss_function(y_pred, labels)
#         single_loss.backward()
#         optimizer.step()
#
# # 预测并评估模型
# model.eval()
# test_predictions = []
# for seq in X_test:
#     with torch.no_grad():
#         model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
#                         torch.zeros(1, 1, model.hidden_layer_size))
#         test_predictions.append(model(seq).item())
#
# # 反归一化
# y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
# test_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))
#
# # 计算评价指标
# mae = mean_absolute_error(y_test, test_predictions)
# rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
# mape = np.mean(np.abs((y_test - test_predictions) / y_test)) * 100
# r2 = r2_score(y_test, test_predictions)
#
# print("MAE:", mae)
# print("RMSE:", rmse)
# print("MAPE:", mape)
# print("R²:", r2)
# 导入所需的库
import numpy as np
import pandas as pd
from PyEMD import EEMD
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 读取CSV文件
file_path = '平安银行.csv'
data = pd.read_csv(file_path)

# 提取需要分解的列数据
column_to_decompose = 'close'
signal = data[column_to_decompose].values

# 定义EEMD分解函数
def eemd_decomposition(signal):
    eemd = EEMD()
    IMF = eemd.eemd(signal)
    return IMF

# 对列数据进行EEMD分解
IMF = eemd_decomposition(signal)

# 数据预处理
def create_dataset(signal, look_back=1):
    dataX, dataY = [], []
    for i in range(len(signal)-look_back-1):
        dataX.append(signal[i:(i+look_back), 0])
        dataY.append(signal[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

scaler = MinMaxScaler(feature_range=(0, 1))
IMF_scaled = scaler.fit_transform(IMF.reshape(-1, 1))

look_back = 1
X, y = create_dataset(IMF_scaled, look_back)
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# 初始化模型和损失函数
model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 100
for i in range(epochs):
    for seq, labels in zip(X_train, y_train):
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

# 模型预测
test_losses = []
with torch.no_grad():
    for seq, labels in zip(X_test, y_test):
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        y_test_pred = model(seq)
        test_loss = loss_function(y_test_pred, labels)
        test_losses.append(test_loss.item())

# 结果评估
y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))
y_test_pred_unscaled = scaler.inverse_transform(np.array([x.item() for x in y_test_pred]).reshape(-1, 1))

mae = mean_absolute_error(y_test_unscaled, y_test_pred_unscaled)
rmse = mean_squared_error(y_test_unscaled, y_test_pred_unscaled, squared=False)
mape = np.mean(np.abs((y_test_unscaled - y_test_pred_unscaled) / y_test_unscaled)) * 100
r2 = r2_score(y_test_unscaled, y_test_pred_unscaled)

print("MAE: ", mae)
print("RMSE: ", rmse)
print("MAPE: ", mape)
print("R²: ", r2)
