import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

# 读取股价数据
df = pd.read_csv("AAPL苹果.csv")

# 提取要预测的目标列（股价）
data = df["close"].values.reshape(-1, 1)

# 归一化数据
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# 划分训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

def create_lstm_model():
    model = Sequential()
    model.add(LSTM(64, input_shape=(1, 1)))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model

model = create_lstm_model()

X_train = train_data[:-1]
y_train = train_data[1:]

X_train = X_train.reshape(X_train.shape[0], 1, 1)
y_train = y_train.reshape(y_train.shape[0], 1)

model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)

# 使用训练好的模型进行预测
X_test = test_data[:-1]
y_test = test_data[1:]

X_test = X_test.reshape(X_test.shape[0], 1, 1)
y_test = y_test.reshape(y_test.shape[0], 1)

predictions = model.predict(X_test)

# 反归一化预测结果
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test)

from sklearn.metrics import r2_score
print('R²：',r2_score(y_test,predictions))

# 计算评价指标
mape = mean_absolute_percentage_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("MAPE:", mape)
print("MAE:", mae)
print("RMSE:", rmse)