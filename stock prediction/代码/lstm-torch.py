import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("比亚迪.csv")
close_prices = data["close"].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(close_prices)
train_size = int(0.8 * len(scaled_prices))
train_data = scaled_prices[:train_size]
test_data = scaled_prices[train_size:]

def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length]
        sequences.append((seq, label))
    return sequences

seq_length = 7
train_sequences = create_sequences(train_data, seq_length)
test_sequences = create_sequences(test_data, seq_length)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
input_size =
hidden_size =
num_layers =
output_size =
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs =
for epoch in range(num_epochs):
    for seq, label in train_sequences:
        optimizer.zero_grad()
        seq = torch.FloatTensor(seq).unsqueeze(0)
        label = torch.FloatTensor(label)
        output = model(seq)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

predictions = []
targets = []
with torch.no_grad():
    for seq, label in test_sequences:
        seq = torch.FloatTensor(seq).unsqueeze(0)
        output = model(seq)
        predictions.append(output.numpy())
        targets.append(label)

predictions = np.array(predictions).squeeze()
targets = np.array(targets)

predictions = scaler.inverse_transform(predictions)
targets = scaler.inverse_transform(targets)

mae = mean_absolute_error(targets, predictions)
rmse = np.sqrt(mean_squared_error(targets, predictions))
mape = np.mean(np.abs((targets - predictions) / targets)) * 100
r2 = r2_score(targets, predictions)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"R²: {r2:.4f}")
