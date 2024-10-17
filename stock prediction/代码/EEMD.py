import pandas as pd
import numpy as np
from PyEMD import EEMD
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('比亚迪22.csv')

# 选择要进行EEMD分解的列
data = df['close'].values

# 创建EEMD对象
eemd = EEMD()

# 执行EEMD分解
eIMFs = eemd.eemd(data)
nIMFs = eIMFs.shape[0]

# 绘制原始数据及分解后的IMF结果
plt.figure(figsize=(12, 9))
# plt.subplot(nIMFs+1, 1, 1)
# plt.plot(data, 'r')
# plt.title('Original Data')

for i in range(nIMFs):
    plt.subplot(nIMFs+1, 1, i+2)
    plt.plot(eIMFs[i], 'g')
    plt.title('IMF %i' % (i+1))

plt.tight_layout()
plt.show()