import numpy as np
import pandas as pd
from pyhht.analysis import EMD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


data = pd.read_csv('平安银行.csv')
x = data.iloc[:, 2].values.reshape(-1, 1)

decomposer = EMD(x)
imfs = decomposer.decompose()


scaler = StandardScaler()
imfs = scaler.fit_transform(imfs.T).T
x = scaler.fit_transform(x)


X_train, X_test, y_train, y_test = train_test_split(imfs, x, test_size=0.2, random_state=0)
train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256)

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

input_size = imfs.shape[1]
hidden_size =
num_layers =
output_size =
lr =
num_epochs =
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BiLSTM(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_preds = []
    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            preds = outputs.detach().cpu().numpy()
            train_preds.extend(preds)
    train_mae = mean_absolute_error(y_train, train_preds)
    train_mape = np.mean(np.abs((y_train - train_preds) / y_train))
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    train_r2 = r2_score(y_train, train_preds)

    test_preds = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            preds = outputs.detach().cpu().numpy()
            test_preds.extend(preds)
    test_mae = mean_absolute_error(y_test, test_preds)
    test_mape = np.mean(np.abs((y_test - test_preds) / y_test))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_r2 = r2_score(y_test, test_preds)

    print(f'Epoch {epoch + 1} | Train MAE: {train_mae:.4f} | Test MAE: {test_mae:.4f}')
    print(f'Epoch {epoch + 1} | Train MAPE: {train_mape:.4f} | Test MAPE: {test_mape:.4f}')
    print(f'Epoch {epoch + 1} | Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}')
    print(f'Epoch {epoch + 1} | Train R2: {train_r2:.4f} | Test R2: {test_r2:.4f}')
