import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

df = pd.read_csv("上证指数.csv")
data = df["close"].values.reshape(-1, 1)
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
train_input = torch.tensor(train_data[:-1]).unsqueeze(1).float()
train_target = torch.tensor(train_data[1:]).float()
test_input = torch.tensor(test_data[:-1]).unsqueeze(1).float()
test_target = torch.tensor(test_data[1:]).float()

input_size =
hidden_size =
num_layers =
model = GRUModel(input_size, hidden_size, num_layers)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs =
batch_size =
total_steps = len(train_input) // batch_size

for epoch in range(num_epochs):
    for step in range(total_steps):
        start = step * batch_size
        end = start + batch_size
        inputs = train_input[start:end]
        targets = train_target[start:end]

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch[{epoch+1}/{num_epochs}], Loss:{loss.item()}")

model.eval()
with torch.no_grad():
    test_output = model(test_input)

test_output = test_output.numpy()
test_target = test_target.numpy()

mae = mean_absolute_error(test_target, test_output)
mape = np.mean(np.abs((test_target - test_output) / test_target)) * 100
rmse = np.sqrt(mean_squared_error(test_target, test_output))
r2 = r2_score(test_target, test_output)

print(f"MAE: {mae}")
print(f"MAPE: {mape}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")