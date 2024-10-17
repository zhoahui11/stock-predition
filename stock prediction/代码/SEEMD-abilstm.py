import pandas as pd
import numpy as np
from PyEMD import EEMD
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import torch.nn.functional as F
from matplotlib import rcParams

df = pd.read_csv('上证指数.csv')
column_name = 'close'

column_data = df[column_name].values
cs = CubicSpline(np.arange(len(column_data)), column_data)

interpolated_data = cs(np.arange(len(column_data)))

eemd = EEMD()


imfs = eemd.eemd(interpolated_data)

imfs_data = []

scaler = MinMaxScaler(feature_range=(0, 1))
for imf in imfs:
    imf_scaled = scaler.fit_transform(imf.reshape(-1, 1))
    imfs_data.append(imf_scaled.flatten())

X = np.array(imfs_data).T

y = scaler.fit_transform(column_data.reshape(-1, 1))

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
y_train = torch.Tensor(y_train)
y_test = torch.Tensor(y_test)

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = MyDataset(X_train, y_train)
test_dataset = MyDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.v = nn.Parameter(torch.rand(hidden_size * 2))

    def forward(self, hidden_states):
        attn_tanh = torch.tanh(self.W(hidden_states))
        attn_scores = torch.matmul(attn_tanh, self.v)
        attn_scores = attn_scores.view(-1, attn_scores.size(1))
        attn_weights = F.softmax(attn_scores, dim=1)
        output = torch.sum(hidden_states * attn_weights.unsqueeze(-1), dim=1)
        return output, attn_weights

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_out, _ = self.attention(lstm_out)
        output = self.fc(attention_out)
        return output

input_size = X_train.shape[1]
hidden_size =
output_size =

model = BiLSTM(input_size, hidden_size, output_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss =
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.unsqueeze(0))
        loss = criterion(outputs, targets.unsqueeze(0))
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    train_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}")


model.eval()
y_train_pred = model(X_train.unsqueeze(0)).squeeze().detach().numpy()
y_test_pred = model(X_test.unsqueeze(0)).squeeze().detach().numpy()

y_train_pred = scaler.inverse_transform(y_train_pred.reshape(-1, 1))
y_test_pred = scaler.inverse_transform(y_test_pred.reshape(-1, 1))
y_train_true = scaler.inverse_transform(y_train.numpy().reshape(-1, 1))
y_test_true = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test_true, y_test_pred)
mape = np.mean(np.abs((y_test_true - y_test_pred) / y_test_true)) * 100
rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
r2 = r2_score(y_test_true, y_test_pred)

print(mae)
print(mape)
print(rmse)
print(r2)
