import pandas as pd
import numpy as np
from PyEMD import EEMD
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
df = pd.read_csv('.csv')
column_name = 'close'
column_data = df[column_name].values
eemd = EEMD()
imfs = eemd.eemd(column_data)
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
input_size = X_train.shape[1]
hidden_size =
output_size =
model = AttentionBiLSTM(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs =
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
