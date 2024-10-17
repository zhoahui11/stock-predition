# import pandas as pd
# import numpy as np
# import torch
# from torch import nn
# from torch.utils.data import DataLoader, Dataset
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# # from pyemd import emd
# from PyEMD import EMD
# # 加载CSV文件
# data = pd.read_csv('平安银行.csv')
#
# # 选择需要进行分解的列
# target_column = 'close'
# target_data = data[target_column].values.reshape(-1, 1)
#
# # 使用EMD对列数据进行分解
# decomposed_data = []
# for i in range(len(target_data)):
#     decomp = EMD(target_data[i])
#     decomposed_data.append(decomp)
#
# # 转换为numpy数组
# decomposed_data = np.array(decomposed_data)
#
# # 数据归一化
# scaler = MinMaxScaler()
# normalized_data = scaler.fit_transform(decomposed_data)
#
# # 划分训练集和测试集
# train_data, test_data = train_test_split(normalized_data, test_size=0.2, random_state=123)
#
# # 定义数据集类
# class MyDataset(Dataset):
#     def __init__(self, data):
#         self.data = torch.FloatTensor(data)
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, index):
#         return self.data[index]
#
# # 定义BiLSTM模型
# class BiLSTM(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(BiLSTM, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
#         self.fc = nn.Linear(hidden_dim * 2, output_dim)
#
#     def forward(self, x):
#         lstm_out, _ = self.lstm(x)
#         lstm_out = lstm_out[:, -1, :]
#         out = self.fc(lstm_out)
#         return out
#
# # 设置超参数
# input_dim = normalized_data.shape[1]
# hidden_dim = 32
# output_dim = 1
# num_epochs = 100
# batch_size = 32
# learning_rate = 0.001
#
# # 创建数据加载器
# train_dataset = MyDataset(train_data)
# test_dataset = MyDataset(test_data)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
# # 初始化模型和优化器
# model = BiLSTM(input_dim, hidden_dim, output_dim)
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# # 训练模型
# for epoch in range(num_epochs):
#     model.train()
#     train_losses = []
#     for data in train_loader:
#         optimizer.zero_grad()
#         outputs = model(data)
#         loss = criterion(outputs, data)
#         loss.backward()
#         optimizer.step()
#         train_losses.append(loss.item())
#
#     mean_train_loss = np.mean(train_losses)
#
#     # 在测试集上进行预测
#     model.eval()
#     with torch.no_grad():
#         test_losses = []
#         predictions = []
#         for data in test_loader:
#             outputs = model(data)
#             loss = criterion(outputs, data)
#             test_losses.append(loss.item())
#             predictions.extend(outputs.detach().numpy())
#
#     mean_test_loss = np.mean(test_losses)
#     predictions = scaler.inverse_transform(predictions)  # 反归一化
#
#     # 计算评价指标
#     true_values = target_data[test_data.shape[0]:].flatten()
#     mae = mean_absolute_error(true_values, predictions)
#     mape = np.mean(np.abs((true_values - predictions) / true_values)) * 100
#     rmse = np.sqrt(mean_squared_error(true_values, predictions))
#     r2 = r2_score(true_values, predictions)
#
#     print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {mean_train_loss:.4f}, '
#           f'Test Loss: {mean_test_loss:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}%, '
#           f'RMSE: {rmse:.4f}, R^2: {r2:.4f}')
import numpy as np
import pandas as pd
from pyhht.analysis import EMD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# 读取数据
data = pd.read_csv('平安银行.csv')
x = data.iloc[:, 2].values.reshape(-1, 1)

# EMD分解
decomposer = EMD(x)
imfs = decomposer.decompose()

# 归一化
scaler = StandardScaler()
imfs = scaler.fit_transform(imfs.T).T
x = scaler.fit_transform(x)

# 构造数据集和数据加载器
X_train, X_test, y_train, y_test = train_test_split(imfs, x, test_size=0.2, random_state=0)
train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256)

# BiLSTM模型定义
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 模型训练
input_size = imfs.shape[1]
hidden_size = 16
num_layers = 2
output_size = 1
lr = 1e-3
num_epochs = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BiLSTM(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 训练集上的表现
    train_preds = []
    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            preds = outputs.detach().cpu().numpy()
            train_preds.extend(preds)
    train_mae = mean_absolute_error(y_train, train_preds)
    train_mape = np.mean(np.abs((y_train - train_preds) / y_train))
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    train_r2 = r2_score(y_train, train_preds)

    # 测试集上的表现
    test_preds = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            preds = outputs.detach().cpu().numpy()
            test_preds.extend(preds)
    test_mae = mean_absolute_error(y_test, test_preds)
    test_mape = np.mean(np.abs((y_test - test_preds) / y_test))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_r2 = r2_score(y_test, test_preds)

    # 训练结果输出
    print(f'Epoch {epoch + 1} | Train MAE: {train_mae:.4f} | Test MAE: {test_mae:.4f}')
    print(f'Epoch {epoch + 1} | Train MAPE: {train_mape:.4f} | Test MAPE: {test_mape:.4f}')
    print(f'Epoch {epoch + 1} | Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}')
    print(f'Epoch {epoch + 1} | Train R2: {train_r2:.4f} | Test R2: {test_r2:.4f}')
