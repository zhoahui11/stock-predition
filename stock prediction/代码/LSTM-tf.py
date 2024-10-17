import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import time
data = pd.read_csv('平安银行.csv')
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data['close'].values.reshape(-1, 1))
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]
def create_dataset(dataset, lookback):
    X, Y = [], []
    for i in range(len(dataset) - lookback):
        X.append(dataset[i : i + lookback, 0])
        Y.append(dataset[i + lookback, 0])
    return np.array(X), np.array(Y)
lookback =
X_train, Y_train = create_dataset(train_data, lookback)
X_test, Y_test = create_dataset(test_data, lookback)

model = Sequential()
model.add(LSTM(50, input_shape=(lookback, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

train_start_time = time.time()
model.fit(X_train, Y_train, epochs=100, batch_size=32, verbose=2)
train_end_time = time.time()
train_time = train_end_time - train_start_time

test_start_time = time.time()
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)
test_end_time = time.time()
test_time = test_end_time - test_start_time

train_predictions = scaler.inverse_transform(train_predictions)
Y_train = scaler.inverse_transform([Y_train])
test_predictions = scaler.inverse_transform(test_predictions)
Y_test = scaler.inverse_transform([Y_test])

mae_train = mean_absolute_error(Y_train[0], train_predictions[:, 0])
mae_test = mean_absolute_error(Y_test[0], test_predictions[:, 0])
mape_train = mean_absolute_percentage_error(Y_train[0], train_predictions[:, 0])
mape_test = mean_absolute_percentage_error(Y_test[0], test_predictions[:, 0])
rmse_train = np.sqrt(mean_squared_error(Y_train[0], train_predictions[:, 0]))
rmse_test = np.sqrt(mean_squared_error(Y_test[0], test_predictions[:, 0]))
r2_train = r2_score(Y_train[0], train_predictions[:, 0])
r2_test = r2_score(Y_test[0], test_predictions[:, 0])

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

print(f'Training Time: {train_time:.2f} seconds')
print(f'Testing Time: {test_time:.2f} seconds')