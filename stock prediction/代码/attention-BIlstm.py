import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
# from PyEMD import signal_processing as EMD
from PyEMD import EEMD
from PyEMD import EMD
from sklearn.preprocessing import MinMaxScaler

def EEMD(data, ensemble_size=50):
    ensemble = np.zeros_like(data)
    for _ in range(ensemble_size):
        noise = 0.2 * np.random.randn(*data.shape)
        ensemble += EMD.emd(data + noise)
    return ensemble / ensemble_size
class BiLSTMAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, attention_dim, output_dim):
        super(BiLSTMAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=1, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(2 * hidden_dim, attention_dim)
        self.out = nn.Linear(2 * hidden_dim, output_dim)
        self.v = nn.Parameter(torch.rand(attention_dim))
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_scores = torch.tanh(self.attention(lstm_out))
        attention_weights = torch.matmul(attention_scores, self.v)
        weights = F.softmax(attention_weights, dim=1)
        weighted_output = torch.sum(lstm_out * weights, dim=1)
        return self.out(weighted_output)
def load_data(file_path, column_name, seq_len, batch_size):
    df = pd.read_csv(file_path)
    data = df[column_name].values.reshape(-1, 1)
    data = EEMD(data)  # 对数据进行EEMD分解
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    x, y = [], []
    for i in range(len(data) - seq_len - 1):
        x.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    x = np.array(x)
    y = np.array(y)
    x = torch.tensor(x, dtype=torch.float32)  # [seq_len, batch_size, 1]
    y = torch.tensor(y, dtype=torch.float32)  # [batch_size, 1]
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, scaler
def train_and_evaluate(model, dataloader, criterion, optimizer, scaler, test_loader):
    model.train()
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        predictions = model(test_loader.dataset.tensors[0])
        predictions = scaler.inverse_transform(predictions.numpy())
        targets = scaler.inverse_transform(test_loader.dataset.tensors[1].numpy())

        mae = mean_absolute_error(targets, predictions)
        mape = np.mean(np.abs((targets - predictions) / targets))
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        r2 = r2_score(targets, predictions)
    return mae, mape, rmse, r2
def main(file_path, column_name, seq_len, batch_size, hidden_dim, attention_dim, learning_rate, num_epochs):
    dataloader, scaler = load_data(file_path, column_name, seq_len, batch_size)
    train_loader, test_loader = train_test_split(dataloader, test_size=0.1, random_state=42)
    model = BiLSTMAttention(input_dim=1, hidden_dim=hidden_dim, attention_dim=attention_dim, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        mae, mape, rmse, r2 = train_and_evaluate(model, train_loader, criterion, optimizer, scaler, test_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, MAE: {mae:.4f}, MAPE: {mape:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}')

file_path = '平安银行.csv'
column_name = 'close'
seq_len =
batch_size =
hidden_dim =
attention_dim =
learning_rate =
num_epochs =

# 运行主函数
main(file_path, column_name, seq_len, batch_size, hidden_dim, attention_dim, learning_rate, num_epochs)