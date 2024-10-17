import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 读取CSV文件
data = pd.read_csv('平安银行.csv')

# 提取需要预测的列数据
target_column = 'close'  # 替换为你想要预测的列名
target = data[target_column].values.reshape(-1, 1)

# 数据归一化
scaler = MinMaxScaler()
target = scaler.fit_transform(target)

# 划分训练集和测试集
train_ratio = 0.8
train_size = int(len(target) * train_ratio)
train_data = target[:train_size]
test_data = target[train_size:]

# 转换为PyTorch Tensor
train_data = torch.from_numpy(train_data).float()
test_data = torch.from_numpy(test_data).float()

# 创建BiLSTM模型
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, output_size)

    def forward(self, input):
        output, _ = self.lstm(input)
        output = self.fc(output[:, -1, :])
        return output

input_size = 1  # 输入特征数量，此处为单变量预测
hidden_size = 64  # LSTM隐藏层大小
output_size = 1  # 输出特征数量，此处为单变量预测

model = BiLSTM(input_size, hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
batch_size = 32

for epoch in range(num_epochs):
    for i in range(0, train_data.size(0)-batch_size, batch_size):
        input_seq = train_data[i:i+batch_size].view(batch_size, -1, input_size)
        target_seq = train_data[i+1:i+batch_size+1].view(batch_size, -1, input_size)

        output_seq = model(input_seq)
        loss = criterion(output_seq, target_seq)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 测试模型
test_input_seq = test_data[:-1].view(-1, 1, input_size)
test_target_seq = test_data[1:].view(-1, 1, input_size)

with torch.no_grad():
    test_output_seq = model(test_input_seq)

# 反归一化
test_output_seq = scaler.inverse_transform(test_output_seq.view(-1, 1))
test_target_seq = scaler.inverse_transform(test_target_seq.view(-1, 1))

# 计算评价指标
mae = mean_absolute_error(test_target_seq, test_output_seq)
mape = np.mean(np.abs((test_target_seq - test_output_seq) / test_target_seq)) * 100
rmse = np.sqrt(mean_squared_error(test_target_seq, test_output_seq))
r2 = r2_score(test_target_seq, test_output_seq)

# 输出评价指标结果
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.4f}%")
print(f"RMSE: {rmse:.4f}")
print(f"R^2: {r2:.4f}")
