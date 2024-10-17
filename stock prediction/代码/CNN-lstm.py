import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Flatten
import matplotlib.pyplot as plt
data = pd.read_csv('比亚迪.csv', header=0)
prices = data['close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)
train_size = int(len(prices_scaled) * 0.8)
train_data, test_data = prices_scaled[:train_size], prices_scaled[train_size:]
def create_dataset(dataset, look_back=1):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)

look_back =
X_train, y_train = create_dataset(train_data, look_back)
X_test, y_test = create_dataset(test_data, look_back)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(look_back, 1)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(LSTM(50, return_sequences=True))
model.add(Flatten())
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=2)
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)
train_predictions = scaler.inverse_transform(train_predictions)
y_train = scaler.inverse_transform([y_train])
test_predictions = scaler.inverse_transform(test_predictions)
y_test = scaler.inverse_transform([y_test])

mae_train = mean_absolute_error(y_train[0], train_predictions[:, 0])
mape_train = mean_absolute_percentage_error(y_train[0], train_predictions[:, 0])
rmse_train = np.sqrt(mean_squared_error(y_train[0], train_predictions[:, 0]))
r2_train = r2_score(y_train[0], train_predictions[:, 0])

mae_test = mean_absolute_error(y_test[0], test_predictions[:, 0])
mape_test = mean_absolute_percentage_error(y_test[0], test_predictions[:, 0])
rmse_test = np.sqrt(mean_squared_error(y_test[0], test_predictions[:, 0]))
r2_test = r2_score(y_test[0], test_predictions[:, 0])

print("训练集：")
print("MAE:", mae_train)
print("MAPE:", mape_train)
print("RMSE:", rmse_train)
print("R2:", r2_train)
print("测试集：")
print("MAE:", mae_test)
print("MAPE:", mape_test)
print("RMSE:", rmse_test)
print("R2:", r2_test)

plt.plot(y_test[0], label='真实股价')
plt.plot(test_predictions[:, 0], label='预测股价')
plt.legend()
plt.show()