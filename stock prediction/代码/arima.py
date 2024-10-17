import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
import time
start_time = time.time()
df = pd.read_csv('平安银行.csv')
load_time = time.time() - start_time
close_prices = df['close'].values
train_size = int(len(close_prices) * 0.8)
train, test = close_prices[0:train_size], close_prices[train_size:]
start_time = time.time()
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()
arima_time = time.time() - start_time
start_time = time.time()
predictions = model_fit.forecast(len(test))
prediction_time = time.time() - start_time
mae = mean_absolute_error(test, predictions)
mape = mean_absolute_percentage_error(test, predictions)
rmse = np.sqrt(mean_squared_error(test, predictions))
r2 = r2_score(test, predictions)
print(f'MAE: {mae}')
print(f'MAPE: {mape}')
print(f'RMSE: {rmse}')
print(f'R²: {r2}')
print(f"Load Time: {load_time:.4f} seconds")
print(f"ARIMA Model Time: {arima_time:.4f} seconds")
print(f"Prediction Time: {prediction_time:.4f} seconds")