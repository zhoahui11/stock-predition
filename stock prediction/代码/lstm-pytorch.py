import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyeemd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

data = pd.read_csv('data.csv')
column_data = data['column_name'].values.reshape(-1, 1)  # 调整数据形状为二维数组

# EEMD分解
imfs = pyeemd.eemd(column_data.squeeze(), NE=10, num_siftings=50)


class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


train_size = int(len(imfs) * 0.8)
test_size = len(imfs) - train_size
train_data = imfs[:train_size]
test_data = imfs[train_size:]
train_dataset = MyDataset(train_data)
test_dataset = MyDataset(test_data)
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_dim = imfs.shape[1]
hidden_dim =
output_dim =

model = BiLSTM(input_dim, hidden_dim, output_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    train_loss =
    for batch_data in train_loader:
        optimizer.zero_grad()
        batch_data = batch_data.float()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_data[:, -1, :])
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_data in test_loader:
            batch_data = batch_data.float()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_data[:, -1, :])
            test_loss += loss.item()
        test_loss /= len(test_loader)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

model.eval()
predictions = []
with torch.no_grad():
    for batch_data in test_loader:
        batch_data = batch_data.float()
        outputs = model(batch_data)
        predictions.extend(outputs.squeeze().numpy())

plt.plot(column_data[train_size:].squeeze(), label='True')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()

true_values = column_data[train_size:].squeeze()
mae = np.mean(np.abs(true_values - predictions))
mapr = np.mean(np.abs((true_values - predictions) / true_values))
rmse = np.sqrt(np.mean((true_values - predictions) ** 2))
r2 = 1 - np.sum((true_values - predictions) ** 2) / np.sum((true_values - np.mean(true_values)) ** 2)
print(f'MAE: {mae:.4f}, MAPR: {mapr:.4f}, RMSE: {rmse:.4f}, R^2: {r2:.4f}')