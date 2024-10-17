import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score
# 定义GRU模型
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# 准备数据
df = pd.read_csv("上证指数.csv")

# 提取要预测的目标列（股价）
data = df["close"].values.reshape(-1, 1)
# 准备数据
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
train_input = torch.tensor(train_data[:-1]).unsqueeze(1).float()
train_target = torch.tensor(train_data[1:]).float()
test_input = torch.tensor(test_data[:-1]).unsqueeze(1).float()
test_target = torch.tensor(test_data[1:]).float()

# 初始化模型
input_size = 1
hidden_size = 128
num_layers = 2
model = GRUModel(input_size, hidden_size, num_layers)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
batch_size = 32
total_steps = len(train_input) // batch_size

for epoch in range(num_epochs):
    for step in range(total_steps):
        start = step * batch_size
        end = start + batch_size
        inputs = train_input[start:end]
        targets = train_target[start:end]

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch[{epoch+1}/{num_epochs}], Loss:{loss.item()}")

# 在测试集上进行预测
model.eval()
with torch.no_grad():
    test_output = model(test_input)

# 转换为numpy数组
test_output = test_output.numpy()
test_target = test_target.numpy()

# 计算评价指标
mae = mean_absolute_error(test_target, test_output)
mape = np.mean(np.abs((test_target - test_output) / test_target)) * 100
rmse = np.sqrt(mean_squared_error(test_target, test_output))
r2 = r2_score(test_target, test_output)

print(f"MAE: {mae}")
print(f"MAPE: {mape}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")