import numpy as np
import pandas as pd
from PyEMD import EEMD
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

# 读取文件
data = pd.read_csv('平安银行.csv')

# 提取某一列数据
column_data = data['close'].values.reshape(-1, 1)

# EEMD 分解
eemd = EEMD()
IMFs = eemd.eemd(column_data)

# 构建输入输出数据
X = np.concatenate(IMFs, axis=1)
y = column_data

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建 BiLSTM 模型
model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True), input_shape=(X.shape[1], 1)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(1)
])

# 编译模型
model.compile(loss='mse', optimizer='adam')

# 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# 预测结果
predictions = model.predict(X_test)

# 绘制预测价格与真实价格的对比图
plt.plot(predictions, label='Predictions')
plt.plot(y_test, label='True values')
plt.legend()
plt.show()