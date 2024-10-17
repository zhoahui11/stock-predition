import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class StockDataset(Dataset):
    def __init__(self, data, seq_len):
        self.seq_len = seq_len
        self.data = torch.FloatTensor(data)

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len]
        return x, y


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, h = self.gru(x)
        out = self.fc(h[-1])
        return out


data = pd.read_csv('上证指数.csv')
prices = data['close'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(-1, 1))
prices_normalized = scaler.fit_transform(prices)

train_size = int(len(prices_normalized) * 0.8)
train_data = prices_normalized[:train_size]
test_data = prices_normalized[train_size:]

seq_len =
input_dim =
hidden_dim =
output_dim =
batch_size =
num_epochs =


train_dataset = StockDataset(train_data, seq_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = GRUModel(input_dim, hidden_dim, output_dim)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss}')

model.eval()
test_dataset = StockDataset(test_data, seq_len)
test_loader = DataLoader(test_dataset, batch_size=1)
predictions = []
targets = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predictions.append(outputs.item())
        targets.append(labels.item())

predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
targets = scaler.inverse_transform(np.array(targets).reshape(-1, 1))

mae = mean_absolute_error(targets, predictions)
mape = np.mean(np.abs((targets - predictions) / targets)) * 100
rmse = np.sqrt(mean_squared_error(targets, predictions))
r2 = r2_score(targets, predictions)

print(f'MAE: {mae}')
print(f'MAPE: {mape}')
print(f'RMSE: {rmse}')
print(f'R^2: {r2}')