# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#
# # Define the CNN-BiLSTM model
# class CNNBiLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(CNNBiLSTM, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.cnn = nn.Conv1d(input_size, hidden_size, kernel_size=2, padding=1)
#         self.bilstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
#         self.attention = nn.Linear(hidden_size * 2, 1)
#         self.fc = nn.Linear(hidden_size * 2, output_size)
#
#     def forward(self, x):
#         x = x.permute(0, 2, 1)
#         x = self.cnn(x)
#         x = x.permute(0, 2, 1)
#         output, _ = self.bilstm(x)
#         attention_weights = torch.softmax(self.attention(output), dim=1)
#         context_vector = torch.sum(attention_weights * output, dim=1)
#         output = self.fc(context_vector)
#         return output
#
# # Read the data from file
# df = pd.read_csv('沪深300.csv')
#
# # Select only the 'Close' column
# data = df['close'].values.reshape(-1, 1)
#
# # Normalize the data
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_data = scaler.fit_transform(data)
#
# # Split data into train and test sets
# train_size = int(len(data) * 0.8)
# train_data = scaled_data[:train_size, :]
# test_data = scaled_data[train_size:, :]
#
# # Prepare the training data
# def prepare_data(data, lookback):
#     X, y = [], []
#     for i in range(lookback, len(data)):
#         X.append(data[i-lookback:i, :])
#         y.append(data[i, :])
#     return np.array(X), np.array(y)
#
# lookback = 1
# X_train, y_train = prepare_data(train_data, lookback)
# X_test, y_test = prepare_data(test_data, lookback)
#
# # Convert data to PyTorch tensors
# X_train = torch.from_numpy(X_train).float()
# y_train = torch.from_numpy(y_train).float()
# X_test = torch.from_numpy(X_test).float()
# y_test = torch.from_numpy(y_test).float()
#
# # Create the model
# input_size = 1
# hidden_size = 32
# num_layers = 2
# output_size = 1
# model = CNNBiLSTM(input_size, hidden_size, num_layers, output_size)
#
# # Define the loss function and optimizer
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001)
#
# # Train the model
# num_epochs = 200
# batch_size = 64
# for epoch in range(num_epochs):
#     for i in range(0, len(X_train), batch_size):
#         inputs = X_train[i:i+batch_size]
#         targets = y_train[i:i+batch_size]
#
#         # Forward pass
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     if (epoch + 1) % 10 == 0:
#         print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
#
# # Make predictions on the test data
# with torch.no_grad():
#     predictions = model(X_test)
#
# # Inverse transform the predictions and actual values
# predictions = scaler.inverse_transform(predictions.numpy())
# y_test = scaler.inverse_transform(y_test.numpy())
#
# # Calculate evaluation metrics
# mae = mean_absolute_error(y_test, predictions)
# mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
# rmse = np.sqrt(mean_squared_error(y_test, predictions))
# r2 = r2_score(y_test, predictions)
#
# # Print the evaluation results
# print('MAE: ', mae)
# print('MAPE: ', mape)
# print('RMSE: ', rmse)
# print('R²: ', r2)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time  # 导入time模块

# Define the CNN-BiLSTM model
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

# Read the data from file
df = pd.read_csv('平安银行.csv')

# Select only the 'Close' column
data = df['close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Split data into train and test sets
train_size = int(len(data) * 0.8)
train_data = scaled_data[:train_size, :]
test_data = scaled_data[train_size:, :]

# Prepare the training data
def prepare_data(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, :])
        y.append(data[i, :])
    return np.array(X), np.array(y)

lookback = 1
X_train, y_train = prepare_data(train_data, lookback)
X_test, y_test = prepare_data(test_data, lookback)

# Convert data to PyTorch tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

# Create the model
input_size = 1
hidden_size = 32
num_layers = 2
output_size = 1
model = CNNBiLSTM(input_size, hidden_size, num_layers, output_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Train the model
num_epochs = 100
batch_size = 64
start_time = time.time()  # 记录训练开始时间
for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        inputs = X_train[i:i+batch_size]
        targets = y_train[i:i+batch_size]

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
train_time = time.time() - start_time  # 计算训练时间

# Make predictions on the test data
with torch.no_grad():
    start_time = time.time()  # 记录测试开始时间
    predictions = model(X_test)
test_time = time.time() - start_time  # 计算测试时间

# Inverse transform the predictions and actual values
predictions = scaler.inverse_transform(predictions.numpy())
y_test = scaler.inverse_transform(y_test.numpy())

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, predictions)
mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

# Print the evaluation results and times
print('Training Time: {:.4f} seconds'.format(train_time))
print('Testing Time: {:.4f} seconds'.format(test_time))
print('MAE: ', mae)
print('MAPE: ', mape)
print('RMSE: ', rmse)
print('R²: ', r2)