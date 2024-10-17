import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

# 读取数据文件
df = pd.read_csv('平安银行.csv')

# 获取收盘价数据
closing_prices = df['close'].values

# 定义函数将数据划分为输入和输出
def split_data(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)-n_steps):
        X.append(sequence[i:i+n_steps])
        y.append(sequence[i+n_steps])
    return np.array(X), np.array(y)

# 划分训练集和测试集
X, y = split_data(closing_prices, 7)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 创建模型
model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True),
                                  input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.Attention(),
    tf.keras.layers.Dense(1)
])

# 编译模型
model.compile(loss='mean_squared_error', optimizer='adam')

# 训练模型
model.fit(X_train, y_train, epochs=50, batch_size=32)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算评价指标
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# 输出评价指标
print('MAE:', mae)
print('MAPE:', mape)
print('RMSE:', rmse)
print('R²:', r2)