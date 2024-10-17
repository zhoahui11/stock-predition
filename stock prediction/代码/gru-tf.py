# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import GRU, Dense
#
# # import tensorflow as tf
# # import os
# # import numpy as np
# # import random
# #
# # SEED = 1234
# # def set_seeds(seed=SEED):
# #     os.environ['PYTHONHASHSEED'] = str(seed)
# #     random.seed(seed)
# #     tf.random.set_seed(seed)
# #     np.random.seed(seed)
# #
# #
# # def set_global_determinism(seed=SEED):
# #     set_seeds(seed=seed)
# #
# #     os.environ['TF_DETERMINISTIC_OPS'] = '1'
# #     os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
# #
# #     tf.config.threading.set_inter_op_parallelism_threads(1)
# #     tf.config.threading.set_intra_op_parallelism_threads(1)
# #
# #
# # # Call the above function with seed value
# # set_global_determinism(seed=SEED)
#
#
#
# # 读取数据
# data = pd.read_csv('平安银行.csv')
#
# # 选择单一输入变量（特征）
# feature_name = 'close'
# feature = data[feature_name].values
#
# # 数据归一化
# scaler = MinMaxScaler()
# feature_scaled = scaler.fit_transform(feature.reshape(-1, 1))
#
# # 划分训练集和测试集
# train_ratio = 0.8
# train_size = int(len(feature_scaled) * train_ratio)
# train_data = feature_scaled[:train_size]
# test_data = feature_scaled[train_size:]
#
# # 构建训练数据集
# def create_dataset(data, sequence_length):
#     X, y = [], []
#     for i in range(len(data) - sequence_length):
#         X.append(data[i:i+sequence_length])
#         y.append(data[i+sequence_length])
#     return np.array(X), np.array(y)
#
# sequence_length = 5  # 序列长度，即输入历史数据的个数
# (X_train, y_train) = create_dataset(train_data, sequence_length)
# (X_test, y_test) = create_dataset(test_data, sequence_length)
#
# # 构建GRU模型
# model = Sequential()
# model.add(GRU(units=50, input_shape=(X_train.shape[1], 1)))
# model.add(Dense(units=1))
# model.compile(optimizer='adam', loss='mean_squared_error')
#
# # 模型训练
# model.fit(X_train, y_train, epochs=100, batch_size=32)
#
# # 模型预测
# train_pred = model.predict(X_train)
# test_pred = model.predict(X_test)
#
# # 反归一化
# train_pred_inverse = scaler.inverse_transform(train_pred)
# y_train_inverse = scaler.inverse_transform(y_train)
# test_pred_inverse = scaler.inverse_transform(test_pred)
# y_test_inverse = scaler.inverse_transform(y_test)
#
# # 计算评价指标
# mae_train = mean_absolute_error(y_train_inverse, train_pred_inverse)
# mape_train = mean_absolute_percentage_error(y_train_inverse, train_pred_inverse)
# rmse_train = np.sqrt(mean_squared_error(y_train_inverse, train_pred_inverse))
# r_squared_train = r2_score(y_train_inverse, train_pred_inverse)
#
# mae_test = mean_absolute_error(y_test_inverse, test_pred_inverse)
# mape_test = mean_absolute_percentage_error(y_test_inverse, test_pred_inverse)
# rmse_test = np.sqrt(mean_squared_error(y_test_inverse, test_pred_inverse))
# r_squared_test = r2_score(y_test_inverse, test_pred_inverse)
#
# print("训练集评价指标：")
# print("MAE:", mae_train)
# print("MAPE:", mape_train)
# print("RMSE:", rmse_train)
# print("R²:", r_squared_train)
#
# print("测试集评价指标：")
# print("MAE:", mae_test)
# print("MAPE:", mape_test)
# print("RMSE:", rmse_test)
# print("R²:", r_squared_test)

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import time  # 导入time模块

# 开始计时
start_time = time.time()

# 读取数据
data = pd.read_csv('平安银行.csv')

# 选择单一输入变量（特征）
feature_name = 'close'
feature = data[feature_name].values

# 数据归一化
scaler = MinMaxScaler()
feature_scaled = scaler.fit_transform(feature.reshape(-1, 1))

# 划分训练集和测试集
train_ratio = 0.8
train_size = int(len(feature_scaled) * train_ratio)
train_data = feature_scaled[:train_size]
test_data = feature_scaled[train_size:]

# 构建训练数据集
def create_dataset(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

sequence_length = 5  # 序列长度，即输入历史数据的个数
(X_train, y_train) = create_dataset(train_data, sequence_length)
(X_test, y_test) = create_dataset(test_data, sequence_length)

# 构建GRU模型
model = Sequential()
model.add(GRU(units=50, input_shape=(X_train.shape[1], 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 模型训练
train_start_time = time.time()  # 开始计时训练
model.fit(X_train, y_train, epochs=100, batch_size=32)
train_time = time.time() - train_start_time  # 计算训练时间

# 模型预测
predict_start_time = time.time()  # 开始计时预测
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
predict_time = time.time() - predict_start_time  # 计算预测时间

# 反归一化
train_pred_inverse = scaler.inverse_transform(train_pred)
y_train_inverse = scaler.inverse_transform(y_train)
test_pred_inverse = scaler.inverse_transform(test_pred)
y_test_inverse = scaler.inverse_transform(y_test)

# 计算评价指标
mae_train = mean_absolute_error(y_train_inverse, train_pred_inverse)
mape_train = mean_absolute_percentage_error(y_train_inverse, train_pred_inverse)
rmse_train = np.sqrt(mean_squared_error(y_train_inverse, train_pred_inverse))
r_squared_train = r2_score(y_train_inverse, train_pred_inverse)

mae_test = mean_absolute_error(y_test_inverse, test_pred_inverse)
mape_test = mean_absolute_percentage_error(y_test_inverse, test_pred_inverse)
rmse_test = np.sqrt(mean_squared_error(y_test_inverse, test_pred_inverse))
r_squared_test = r2_score(y_test_inverse, test_pred_inverse)

print("训练集评价指标：")
print("MAE:", mae_train)
print("MAPE:", mape_train)
print("RMSE:", rmse_train)
print("R²:", r_squared_train)

print("测试集评价指标：")
print("MAE:", mae_test)
print("MAPE:", mape_test)
print("RMSE:", rmse_test)
print("R²:", r_squared_test)

# 结束计时并计算总运行时间
end_time = time.time()
total_time = end_time - start_time
print(f'Total Time: {total_time:.2f} seconds')
print(f'Training Time: {train_time:.2f} seconds')
print(f'Prediction Time: {predict_time:.2f} seconds')