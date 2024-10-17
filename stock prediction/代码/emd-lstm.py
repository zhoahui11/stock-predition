import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PyEMD import EMD
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
import time

class StockDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length, :]
        y = self.data[idx + self.seq_length, 0]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        out = self.fc(h_lstm[:, -1, :])
        return out
data = pd.read_csv('平安银行.csv')
prices = data['close'].values
emd = EMD()
IMFs = emd.emd(prices)
seq_length =
input_size =
hidden_size =
output_size =
batch_size =
num_epochs =
learning_rate =
train_size = int(len(IMFs) * 0.8)
train_IMFs = IMFs[:train_size]
test_IMFs = IMFs[train_size:]


train_dataset = StockDataset(train_IMFs, seq_length)
test_dataset = StockDataset(test_IMFs, seq_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

start_train_time = time.time()
model.train()
for epoch in range(num_epochs):
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(x_batch.unsqueeze(-1))  # 需要添加一个维度
        loss = criterion(y_pred.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
print(f'Training time: {time.time() - start_train_time:.4f} seconds')

model.eval()
all_preds = []
all_targets = []
start_test_time = time.time()
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        y_pred = model(x_batch.unsqueeze(-1))
        all_preds.append(y_pred.squeeze().numpy())
        all_targets.append(y_batch.numpy())
all_preds = np.concatenate(all_preds)
all_targets = np.concatenate(all_targets)
print(f'Testing time: {time.time() - start_test_time:.4f} seconds')

mae = mean_absolute_error(all_targets, all_preds)
mape = mean_absolute_percentage_error(all_targets, all_preds)
rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
r2 = r2_score(all_targets, all_preds)

print(f'MAE: {mae:.4f}')
print(f'MAPE: {mape:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'R²: {r2:.4f}')