import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

class CNNBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(CNNBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cnn = nn.Conv1d(input_size, hidden_size, kernel_size=2, padding=1)
        self.bilstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        output, _ = self.bilstm(x)
        attention_weights = torch.softmax(self.attention(output), dim=1)
        context_vector = torch.sum(attention_weights * output, dim=1)
        output = self.fc(context_vector)
        return output
df = pd.read_csv('平安银行.csv')
data = df['close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
train_size = int(len(data) * 0.8)
train_data = scaled_data[:train_size, :]
test_data = scaled_data[train_size:, :]
def prepare_data(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, :])
        y.append(data[i, :])
    return np.array(X), np.array(y)

lookback = 1
X_train, y_train = prepare_data(train_data, lookback)
X_test, y_test = prepare_data(test_data, lookback)
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()
input_size =
hidden_size =
num_layers =
output_size =
model = CNNBiLSTM(input_size, hidden_size, num_layers, output_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


num_epochs = 100
batch_size = 64
start_time = time.time()
for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        inputs = X_train[i:i+batch_size]
        targets = y_train[i:i+batch_size]

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
train_time = time.time() - start_time


with torch.no_grad():
    start_time = time.time()
    predictions = model(X_test)
test_time = time.time() - start_time

predictions = scaler.inverse_transform(predictions.numpy())
y_test = scaler.inverse_transform(y_test.numpy())

mae = mean_absolute_error(y_test, predictions)
mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print('Training Time: {:.4f} seconds'.format(train_time))
print('Testing Time: {:.4f} seconds'.format(test_time))
print('MAE: ', mae)
print('MAPE: ', mape)
print('RMSE: ', rmse)
print('R²: ', r2)