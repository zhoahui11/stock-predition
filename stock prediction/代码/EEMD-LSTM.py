import numpy as np
import pandas as pd
from PyEMD import EEMD
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

file_path = '平安银行.csv'
data = pd.read_csv(file_path)

column_to_decompose = 'close'
signal = data[column_to_decompose].values

def eemd_decomposition(signal):
    eemd = EEMD()
    IMF = eemd.eemd(signal)
    return IMF
IMF = eemd_decomposition(signal)

def create_dataset(signal, look_back=1):
    dataX, dataY = [], []
    for i in range(len(signal)-look_back-1):
        dataX.append(signal[i:(i+look_back), 0])
        dataY.append(signal[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

scaler = MinMaxScaler(feature_range=(0, 1))
IMF_scaled = scaler.fit_transform(IMF.reshape(-1, 1))

look_back = 1
X, y = create_dataset(IMF_scaled, look_back)
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]
model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=)

epochs = 100
for i in range(epochs):
    for seq, labels in zip(X_train, y_train):
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

test_losses = []
with torch.no_grad():
    for seq, labels in zip(X_test, y_test):
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        y_test_pred = model(seq)
        test_loss = loss_function(y_test_pred, labels)
        test_losses.append(test_loss.item())

y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))
y_test_pred_unscaled = scaler.inverse_transform(np.array([x.item() for x in y_test_pred]).reshape(-1, 1))

mae = mean_absolute_error(y_test_unscaled, y_test_pred_unscaled)
rmse = mean_squared_error(y_test_unscaled, y_test_pred_unscaled, squared=False)
mape = np.mean(np.abs((y_test_unscaled - y_test_pred_unscaled) / y_test_unscaled)) * 100
r2 = r2_score(y_test_unscaled, y_test_pred_unscaled)

print("MAE: ", mae)
print("RMSE: ", rmse)
print("MAPE: ", mape)
print("R²: ", r2)
