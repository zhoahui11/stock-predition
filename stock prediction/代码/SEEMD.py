# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from PyEMD import EEMD
# from scipy.interpolate import CubicSpline
#
# # 读取CSV文件
# data = pd.read_csv('平安银行.csv')
#
# # 获取某一列数据
# column_data = data['close'].values
#
# # 三样条插值法
# interpolator = CubicSpline(np.arange(len(column_data)), column_data)
#
# # 插值后的数据
# interpolated_data = interpolator(np.linspace(0, len(column_data)-1, len(column_data)*10))
#
# # 改进的EEMD分解
# eemd = EEMD()
# eemd_data = eemd.eemd(interpolated_data)
# imfs, res = eemd.get_imfs_and_residue()
#
# # 绘制分解结果
# plt.figure(figsize=(10, 8))
# for i, imf in enumerate(imfs):
#     plt.subplot(len(imfs)+1, 1, i+1)
#     plt.plot(imf, label='IMF {}'.format(i+1))
#     plt.legend()
# plt.subplot(len(imfs)+1, 1, len(imfs)+1)
# plt.plot(res, label='Residue')
# plt.legend()
# plt.tight_layout()
# plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EEMD

# 读取csv文件的某一列数据
data = pd.read_csv('比亚迪22.csv')
column_data = data['close'].values  # 替换'column_name'为需要读取的列名

# 三次样条插值
def cubic_spline_interpolation(x, y):
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(x, y)
    return cs(x)

# 改进的EEMD分解
def improved_eemd_decomposition(data):
    eemd = EEMD()
    IMF = eemd.eemd(data)
    residue = data - np.sum(IMF, axis=0)
    return IMF, residue

# 对数据进行三次样条插值
x = np.arange(len(column_data))
interpolated_data = cubic_spline_interpolation(x, column_data)

# 改进的EEMD分解
IMF, residue = improved_eemd_decomposition(interpolated_data)

# 绘制分解结果
# plt.rcParams['font.sans-serif'] = ['SimHei']     # 显示中文
# 为了坐标轴负号正常显示。matplotlib默认不支持中文，设置中文字体后，负号会显示异常。需要手动将坐标轴负号设为False才能正常显示负号。
# plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(12, 8))
# plt.subplot(IMF.shape[0]+1, 1, 1)
# plt.plot(x, interpolated_data, 'b')
#plt.title('Original Data')
# plt.title('子序列S1')
for i in range(IMF.shape[0]):
    plt.subplot(IMF.shape[0]+1, 1, i+2)
    plt.plot(x, IMF[i], 'g')
    plt.title('IMF %d' % (i+1))
    #plt.title('内涵模态分量%d' % (i + 1),fontsize=8)
plt.savefig('./比亚迪分解2.jpg',dpi=1000)

plt.tight_layout()
plt.show()
