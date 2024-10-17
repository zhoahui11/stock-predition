# import pandas as pd
# import numpy as np
# from statsmodels.tsa.arima.model import ARIMA
# from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
#
# # 读取数据
# df = pd.read_csv('平安银行.csv')
# close_prices = df['close'].values
#
# # 划分数据集
# train_size = int(len(close_prices) * 0.8)
# train, test = close_prices[0:train_size], close_prices[train_size:]
#
# # 建立ARIMA模型（参数p, d, q需要根据实际情况调整）
# model = ARIMA(train, order=(5, 1, 0))
# model_fit = model.fit()
#
# # 进行预测
# predictions = model_fit.forecast(len(test))
#
# # 计算评价指标
# mae = mean_absolute_error(test, predictions)
# mape = mean_absolute_percentage_error(test, predictions)
# rmse = np.sqrt(mean_squared_error(test, predictions))
# r2 = r2_score(test, predictions)
#
# print(f'MAE: {mae}')
# print(f'MAPE: {mape}')
# print(f'RMSE: {rmse}')
# print(f'R²: {r2}')

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
import time  # 导入time模块

# 读取数据
start_time = time.time()  # 开始计时
df = pd.read_csv('平安银行.csv')
load_time = time.time() - start_time  # 计算加载数据的时间

close_prices = df['close'].values

# 划分数据集
train_size = int(len(close_prices) * 0.8)
train, test = close_prices[0:train_size], close_prices[train_size:]

# 建立ARIMA模型（参数p, d, q需要根据实际情况调整）
start_time = time.time()  # 开始计时
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()
arima_time = time.time() - start_time  # 计算ARIMA模型建立和拟合的时间

# 进行预测
start_time = time.time()  # 开始计时
predictions = model_fit.forecast(len(test))
prediction_time = time.time() - start_time  # 计算预测时间

# 计算评价指标
mae = mean_absolute_error(test, predictions)
mape = mean_absolute_percentage_error(test, predictions)
rmse = np.sqrt(mean_squared_error(test, predictions))
r2 = r2_score(test, predictions)

print(f'MAE: {mae}')
print(f'MAPE: {mape}')
print(f'RMSE: {rmse}')
print(f'R²: {r2}')

# 输出运行时间
print(f"Load Time: {load_time:.4f} seconds")
print(f"ARIMA Model Time: {arima_time:.4f} seconds")
print(f"Prediction Time: {prediction_time:.4f} seconds")