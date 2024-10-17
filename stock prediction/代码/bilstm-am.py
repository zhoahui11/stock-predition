import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# 构建注意力机制的BiLSTM模型
class AttentionBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AttentionBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_size*2, 1)
        self.fc = nn.Linear(hidden_size*2, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        attention_weights = self.attention(out).squeeze(2)
        attention_weights = torch.softmax(attention_weights, dim=1)
        attention_out = torch.bmm(out.transpose(1, 2), attention_weights.unsqueeze(2)).squeeze(2)
        out = self.fc(attention_out)
        return out

# 定义评价指标函数
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, mape, rmse, r2

# 读取数据
data = pd.read_csv('平安银行.csv')  # 假设数据文件名为stock_prices.csv，包含收盘价列

closing_prices = data['close'].values  # 获取收盘价数据

# 数据准备
input_size = 1  # 输入特征维度（收盘价）
output_size = 1  # 输出步数
test_ratio = 0.2  # 测试集比例

# 创建训练集和测试集
train_data, test_data = train_test_split(closing_prices, test_size=test_ratio, shuffle=False)

# 数据转换为序列格式
def sliding_windows(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length - output_size + 1):
        _x = np.expand_dims(data[i:i+seq_length], axis=1)
        _y = data[i+seq_length:i+seq_length+output_size]
        x.append(_x)
        y.append(_y)
    return np.array(x), np.array(y)

seq_length = 5  # 单输入收盘价，所以序列长度为1
x_train, y_train = sliding_windows(train_data, seq_length)
x_test, y_test = sliding_windows(test_data, seq_length)

# 转换成Tensor类型并加载到GPU（如果有）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_train = torch.from_numpy(x_train).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
x_test = torch.from_numpy(x_test).float().to(device)
y_test = torch.from_numpy(y_test).float().to(device)

# 定义模型和优化器
hidden_size = 64
model = AttentionBiLSTM(input_size, hidden_size, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
batch_size = 32
model.train()
for epoch in range(num_epochs):
    for i in range(0, len(x_train), batch_size):
        inputs = x_train[i:i+batch_size]
        targets = y_train[i:i+batch_size]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# 测试模型
model.eval()
with torch.no_grad():
    y_pred = model(x_test)
    y_pred = y_pred.cpu().numpy().flatten()

# 计算评价指标
mae, mape, rmse, r2 = evaluate(y_test.cpu().numpy().flatten(), y_pred)
print('MAE:', mae)
print('MAPE:', mape)
print('RMSE:', rmse)
print('R²:', r2)
