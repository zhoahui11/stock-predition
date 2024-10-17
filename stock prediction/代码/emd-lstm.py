# # # import pandas as pd
# # # import numpy as np
# # # from PyEMD import EMD
# # # import tensorflow as tf
# # # from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# # # from sklearn.preprocessing import MinMaxScaler
# # # from tensorflow.keras.models import Sequential
# # # from tensorflow.keras.layers import Dense, LSTM
# # #
# # # # 读取CSV文件
# # # df = pd.read_csv('平安银行.csv')
# # #
# # # # 执行EMD分解
# # # def emd_decomposition(data):
# # #     emd = EMD()
# # #     IMF = emd(data)
# # #     return IMF
# # #
# # # decomposed_column = emd_decomposition(df['close'].values)
# # #
# # # # 数据预处理
# # # def prepare_data_for_lstm(data, time_steps):
# # #     X, y = [], []
# # #     for i in range(len(data) - time_steps):
# # #         X.append(data[i:(i + time_steps)])
# # #         y.append(data[i + time_steps])
# # #     return np.array(X), np.array(y)
# # #
# # # time_steps = 10
# # # X, y = prepare_data_for_lstm(decomposed_column[0], time_steps)
# # #
# # # # 归一化数据
# # # scaler = MinMaxScaler(feature_range=(0, 1))
# # # X = scaler.fit_transform(X)
# # # y = scaler.fit_transform(y.reshape(-1, 1))
# # #
# # # X = X.reshape((X.shape[0], X.shape[1], 1))
# # #
# # # # 构建并训练LSTM模型
# # # model = Sequential([
# # #     LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
# # #     LSTM(50, return_sequences=False),
# # #     Dense(1)
# # # ])
# # #
# # # model.compile(optimizer='adam', loss='mean_squared_error')
# # # model.fit(X, y, epochs=100, batch_size=32, verbose=0)
# # #
# # # # 进行预测
# # # predicted_values = model.predict(X)
# # # predicted_values = scaler.inverse_transform(predicted_values)
# # #
# # # # 计算评价指标
# # # original_values = decomposed_column[0][time_steps:]
# # # mae = mean_absolute_error(original_values, predicted_values)
# # # rmse = np.sqrt(mean_squared_error(original_values, predicted_values))
# # # mape = np.mean(np.abs((original_values - predicted_values) / original_values)) * 100
# # # r2 = r2_score(original_values, predicted_values)
# # #
# # # print(f"MAE: {mae}, RMSE: {rmse}, MAPE: {mape}, R²: {r2}")
# #
# # import pandas as pd
# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# # from PyEMD import EMD
# # from sklearn.preprocessing import MinMaxScaler
# #
# # # 读取CSV文件
# # df = pd.read_csv('平安银行.csv')
# #
# # # 执行EMD分解
# # def emd_decomposition(data):
# #     emd = EMD()
# #     IMF = emd(data)
# #     return IMF
# #
# # decomposed_column = emd_decomposition(df['close'].values)
# #
# # # 数据预处理
# # def prepare_data_for_lstm(data, time_steps):
# #     X, y = [], []
# #     for i in range(len(data) - time_steps):
# #         X.append(data[i:(i + time_steps)])
# #         y.append(data[i + time_steps])
# #     return np.array(X), np.array(y)
# #
# # time_steps = 5
# # X, y = prepare_data_for_lstm(decomposed_column[0], time_steps)
# #
# # scaler = MinMaxScaler(feature_range=(0, 1))
# # X = scaler.fit_transform(X)
# # y = scaler.fit_transform(y.reshape(-1, 1))
# #
# # X = torch.from_numpy(X).float().unsqueeze(2)
# # y = torch.from_numpy(y).float()
# #
# # # 定义LSTM模型
# # class LSTMModel(nn.Module):
# #     def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
# #         super().__init__()
# #         self.hidden_layer_size = hidden_layer_size
# #         self.lstm = nn.LSTM(input_size, hidden_layer_size)
# #         self.linear = nn.Linear(hidden_layer_size, output_size)
# #
# #     def forward(self, input_seq):
# #         lstm_out, _ = self.lstm(input_seq.view(len(input_seq), 1, -1))
# #         predictions = self.linear(lstm_out.view(len(input_seq), -1))
# #         return predictions[-1]
# #
# # model = LSTMModel()
# #
# # loss_function = nn.MSELoss()
# # optimizer = optim.Adam(model.parameters(), lr=0.001)
# #
# # epochs = 100
# # for i in range(epochs):
# #     for seq, labels in zip(X, y):
# #         optimizer.zero_grad()
# #         y_pred = model(seq)
# #         single_loss = loss_function(y_pred, labels)
# #         single_loss.backward()
# #         optimizer.step()
# #
# # # 进行预测
# # with torch.no_grad():
# #     predicted_values = []
# #     for seq in X:
# #         predicted_values.append(model(seq).item())
# #
# # predicted_values = scaler.inverse_transform(np.array(predicted_values).reshape(-1, 1))
# #
# # # 计算评价指标
# # original_values = scaler.inverse_transform(y.numpy())
# # mae = mean_absolute_error(original_values, predicted_values)
# # rmse = np.sqrt(mean_squared_error(original_values, predicted_values))
# # mape = np.mean(np.abs((original_values - predicted_values) / original_values)) * 100
# # r2 = r2_score(original_values, predicted_values)
# #
# # print(f"MAE: {mae}, RMSE: {rmse}, MAPE: {mape}, R²: {r2}")
# import pandas as pd
# import numpy as np
# from PyEMD import EMD
# from keras.models import Sequential
# from keras.layers import LSTM, Dense
# import time
# # 读取CSV文件
# df = pd.read_csv('平安银行.csv')
#
# # 选择需要分解的列
# data = df['close'].values
#
# # EMD分解函数
# def emd_decomposition(data):
#     imfs = []
#     while True:
#         # 计算当前信号的IMF
#         emd = EMD()
#         imf = emd(data)
#         imfs.append(imf)
#         # 检查是否达到终止条件
#         if len(imf) == 1:
#             break
#         # 更新信号，准备下一轮分解
#         data = data - imf[-1]
#     return imfs
#
# # 划分训练集和测试集
#
# train_size = int(len(data) * 0.8)
# train_data = data[:train_size]
# test_data = data[train_size:]
#
# # 使用EMD分解训练集
# start_time = time.time()
# imfs = emd_decomposition(train_data)
# train_time = time.time() - start_time
# # 构建LSTM模型
# model = Sequential()
# model.add(LSTM(50, input_shape=(None, 1)))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
#
# # 准备训练数据
# X_train = np.array([imf[:-1] for imf in imfs]).reshape(len(imfs), -1, 1)
# y_train = np.array([imf[1:] for imf in imfs]).reshape(len(imfs), -1, 1)
#
# # 训练LSTM模型
# start_time = time.time()
# model.fit(X_train, y_train, epochs=10, batch_size=32)
# training_time = time.time() - start_time
# # 使用EMD分解测试集
# start_time = time.time()
# test_imfs = emd_decomposition(test_data)
# test_time = time.time() - start_time
# # 准备测试数据
# X_test = np.array([imf[:-1] for imf in test_imfs]).reshape(len(test_imfs), -1, 1)
# y_test = np.array([imf[1:] for imf in test_imfs]).reshape(len(test_imfs), -1, 1)
#
# # 预测结果
# start_time = time.time()  # 开始计时
# y_pred = model.predict(X_test)
# prediction_time = time.time() - start_time  # 计算预测时间
# # 计算评价指标
# def mae(y_true, y_pred):
#     return np.mean(np.abs(y_true - y_pred))
#
# def mape(y_true, y_pred):
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#
# def rmse(y_true, y_pred):
#     return np.sqrt(np.mean(np.square(y_true - y_pred)))
#
# def r_squared(y_true, y_pred):
#     y_mean = np.mean(y_true)
#     ss_total = np.sum(np.square(y_true - y_mean))
#     ss_residual = np.sum(np.square(y_true - y_pred))
#     return 1 - (ss_residual / ss_total)
#
# mae_score = mae(y_test, y_pred)
# mape_score = mape(y_test, y_pred)
# rmse_score = rmse(y_test, y_pred)
# r_squared_score = r_squared(y_test, y_pred)
#
# print("MAE:", mae_score)
# print("MAPE:", mape_score)
# print("RMSE:", rmse_score)
# print("R²:", r_squared_score)
#
# print(f"EMD Decomposition Train Time: {train_time:.4f} seconds")
# print(f"Training Time: {training_time:.4f} seconds")
# print(f"EMD Decomposition Test Time: {test_time:.4f} seconds")
# print(f"Prediction Time: {prediction_time:.4f} seconds")


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PyEMD import EMD
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
import time

# 自定义数据集
class StockDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length, :]
        y = self.data[idx + self.seq_length, 0]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        out = self.fc(h_lstm[:, -1, :])  # 只取最后一个时间步的输出
        return out

# 读取CSV数据
data = pd.read_csv('平安银行.csv')  # 替换为你的CSV文件路径
prices = data['close'].values  # 替换为你需要的列名

# EMD分解
emd = EMD()
IMFs = emd.emd(prices)

# 超参数
seq_length = 10
input_size = 1
hidden_size = 50
output_size = 1
batch_size = 16
num_epochs = 50
learning_rate = 0.001

# 数据集划分
train_size = int(len(IMFs) * 0.8)
train_IMFs = IMFs[:train_size]
test_IMFs = IMFs[train_size:]

# 创建训练和测试数据集
train_dataset = StockDataset(train_IMFs, seq_length)
test_dataset = StockDataset(test_IMFs, seq_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型、损失函数和优化器
model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
start_train_time = time.time()
model.train()
for epoch in range(num_epochs):
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(x_batch.unsqueeze(-1))  # 需要添加一个维度
        loss = criterion(y_pred.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
print(f'Training time: {time.time() - start_train_time:.4f} seconds')

# 测试模型
model.eval()
all_preds = []
all_targets = []
start_test_time = time.time()
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        y_pred = model(x_batch.unsqueeze(-1))
        all_preds.append(y_pred.squeeze().numpy())
        all_targets.append(y_batch.numpy())
all_preds = np.concatenate(all_preds)
all_targets = np.concatenate(all_targets)
print(f'Testing time: {time.time() - start_test_time:.4f} seconds')

# 计算评价指标
mae = mean_absolute_error(all_targets, all_preds)
mape = mean_absolute_percentage_error(all_targets, all_preds)
rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
r2 = r2_score(all_targets, all_preds)

print(f'MAE: {mae:.4f}')
print(f'MAPE: {mape:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'R²: {r2:.4f}')