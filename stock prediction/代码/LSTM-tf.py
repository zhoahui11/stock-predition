# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
#
# # 读取数据集
# data = pd.read_csv('平安银行.csv')
#
# # 提取股价列并进行归一化
# scaler = MinMaxScaler()
# scaled_data = scaler.fit_transform(data['close'].values.reshape(-1, 1))
#
# # 划分训练集和测试集
# train_size = int(len(scaled_data) * 0.8)
# train_data = scaled_data[:train_size]
# test_data = scaled_data[train_size:]
#
# # 准备训练数据
# def create_dataset(dataset, lookback):
#     X, Y = [], []
#     for i in range(len(dataset) - lookback):
#         X.append(dataset[i : i + lookback, 0])
#         Y.append(dataset[i + lookback, 0])
#     return np.array(X), np.array(Y)
#
# lookback = 20
# X_train, Y_train = create_dataset(train_data, lookback)
# X_test, Y_test = create_dataset(test_data, lookback)
#
# # 构建和训练LSTM模型
# model = Sequential()
# model.add(LSTM(50, input_shape=(lookback, 1)))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(X_train, Y_train, epochs=100, batch_size=32, verbose=2)
#
# # 进行预测
# train_predictions = model.predict(X_train)
# test_predictions = model.predict(X_test)
#
# # 反归一化预测结果
# train_predictions = scaler.inverse_transform(train_predictions)
# Y_train = scaler.inverse_transform([Y_train])
# test_predictions = scaler.inverse_transform(test_predictions)
# Y_test = scaler.inverse_transform([Y_test])
#
# # 计算评价指标
# mae_train = mean_absolute_error(Y_train[0], train_predictions[:, 0])
# mae_test = mean_absolute_error(Y_test[0], test_predictions[:, 0])
# mape_train = mean_absolute_percentage_error(Y_train[0], train_predictions[:, 0])
# mape_test = mean_absolute_percentage_error(Y_test[0], test_predictions[:, 0])
# rmse_train = np.sqrt(mean_squared_error(Y_train[0], train_predictions[:, 0]))
# rmse_test = np.sqrt(mean_squared_error(Y_test[0], test_predictions[:, 0]))
# r2_train = r2_score(Y_train[0], train_predictions[:, 0])
# r2_test = r2_score(Y_test[0], test_predictions[:, 0])
#
# # 输出评价指标
# print("训练集指标：")
# print("MAE:", mae_train)
# print("MAPE:", mape_train)
# print("RMSE:", rmse_train)
# print("R²:", r2_train)
# print("测试集指标：")
# print("MAE:", mae_test)
# print("MAPE:", mape_test)
# print("RMSE:", rmse_test)
# print("R²:", r2_test)

# 绘制预测结果
# import matplotlib.pyplot as plt
#
# train_dates = data['date'].values[lookback:train_size]
# test_dates = data['date'].values[train_size + lookback:]
# plt.figure(figsize=(12, 8))
# plt.plot(train_dates, Y_train[0], label='实际训练集股价')
# plt.plot(train_dates, train_predictions[:, 0], label='训练集预测股价')
# plt.plot(test_dates, Y_test[0], label='实际测试集股价')
# plt.plot(test_dates, test_predictions[:, 0], label='测试集预测股价')
# plt.xlabel('日期')
# plt.ylabel('股价')
# plt.title('股价预测结果')
# plt.legend()
# plt.show()

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import time  # 导入time模块

# 读取数据集
data = pd.read_csv('平安银行.csv')

# 提取股价列并进行归一化
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data['close'].values.reshape(-1, 1))

# 划分训练集和测试集
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# 准备训练数据
def create_dataset(dataset, lookback):
    X, Y = [], []
    for i in range(len(dataset) - lookback):
        X.append(dataset[i : i + lookback, 0])
        Y.append(dataset[i + lookback, 0])
    return np.array(X), np.array(Y)

lookback = 20
X_train, Y_train = create_dataset(train_data, lookback)
X_test, Y_test = create_dataset(test_data, lookback)

# 构建和训练LSTM模型
model = Sequential()
model.add(LSTM(50, input_shape=(lookback, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# 记录训练开始时间
train_start_time = time.time()
model.fit(X_train, Y_train, epochs=100, batch_size=32, verbose=2)
# 记录训练结束时间并计算训练时间
train_end_time = time.time()
train_time = train_end_time - train_start_time

# 进行预测
# 记录测试开始时间
test_start_time = time.time()
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)
# 记录测试结束时间并计算测试时间
test_end_time = time.time()
test_time = test_end_time - test_start_time

# 反归一化预测结果
train_predictions = scaler.inverse_transform(train_predictions)
Y_train = scaler.inverse_transform([Y_train])
test_predictions = scaler.inverse_transform(test_predictions)
Y_test = scaler.inverse_transform([Y_test])

# 计算评价指标
mae_train = mean_absolute_error(Y_train[0], train_predictions[:, 0])
mae_test = mean_absolute_error(Y_test[0], test_predictions[:, 0])
mape_train = mean_absolute_percentage_error(Y_train[0], train_predictions[:, 0])
mape_test = mean_absolute_percentage_error(Y_test[0], test_predictions[:, 0])
rmse_train = np.sqrt(mean_squared_error(Y_train[0], train_predictions[:, 0]))
rmse_test = np.sqrt(mean_squared_error(Y_test[0], test_predictions[:, 0]))
r2_train = r2_score(Y_train[0], train_predictions[:, 0])
r2_test = r2_score(Y_test[0], test_predictions[:, 0])

# 输出评价指标
print("训练集指标：")
print("MAE:", mae_train)
print("MAPE:", mape_train)
print("RMSE:", rmse_train)
print("R²:", r2_train)
print("测试集指标：")
print("MAE:", mae_test)
print("MAPE:", mape_test)
print("RMSE:", rmse_test)
print("R²:", r2_test)

# 输出训练时间和测试时间
print(f'Training Time: {train_time:.2f} seconds')
print(f'Testing Time: {test_time:.2f} seconds')