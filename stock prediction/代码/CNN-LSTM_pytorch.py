import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
data = pd.read_csv('平安银行.csv')
column_to_predict = 'close'  # 需要预测的列名
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_data = self.data.iloc[idx]['close'].values
        target = self.data.iloc[idx][column_to_predict]
        input_data = np.array(input_data)
        return input_data, target

class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        _, (x, _) = self.lstm(x)
        x = x[-1]
        x = self.fc(x)
        return x
def train(model, train_loader, criterion, optimizer):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = criterion(outputs.squeeze(), targets.float())
        loss.backward()
        optimizer.step()
def evaluate(model, test_loader):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs.float())
            predictions.extend(outputs.squeeze().tolist())
            targets.extend(targets.tolist())
    return predictions, targets
batch_size =
epochs =
learning_rate =
train_dataset = CustomDataset(train_data)
test_dataset = CustomDataset(test_data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
model = CNNLSTM()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
for epoch in range(epochs):
    train(model, train_loader, criterion, optimizer)

predictions, targets = evaluate(model, test_loader)

mae = np.mean(np.abs(np.array(predictions) - np.array(targets)))
mape = np.mean(np.abs((np.array(predictions) - np.array(targets)) / np.array(targets))) * 100
rmse = np.sqrt(np.mean(np.square(np.array(predictions) - np.array(targets))))
ssr = np.sum(np.square(np.array(predictions) - np.array(targets)))
sst = np.sum(np.square(np.array(targets) - np.mean(np.array(targets))))
r_squared = 1 - ssr / sst

print(f'MAE: {mae:.4f}')
print(f'MAPE: {mape:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'R²: {r_squared:.4f}')