import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EEMD
data = pd.read_csv('比亚迪22.csv')
column_data = data['close'].values  # 替换'column_name'为需要读取的列名

def cubic_spline_interpolation(x, y):
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(x, y)
    return cs(x)

def improved_eemd_decomposition(data):
    eemd = EEMD()
    IMF = eemd.eemd(data)
    residue = data - np.sum(IMF, axis=0)
    return IMF, residue

x = np.arange(len(column_data))
interpolated_data = cubic_spline_interpolation(x, column_data)

IMF, residue = improved_eemd_decomposition(interpolated_data)
plt.figure(figsize=(12, 8))
for i in range(IMF.shape[0]):
    plt.subplot(IMF.shape[0]+1, 1, i+2)
    plt.plot(x, IMF[i], 'g')
    plt.title('IMF %d' % (i+1))
    #plt.title('内涵模态分量%d' % (i + 1),fontsize=8)
plt.savefig('./比亚迪分解2.jpg',dpi=1000)

plt.tight_layout()
plt.show()
