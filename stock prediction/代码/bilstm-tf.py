import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Read the data from the CSV file
df = pd.read_csv('平安银行.csv')
data = df['close']
# Normalize the data using Min-Max scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

# Define the number of previous days to use for prediction
num_steps = 7

# Split the data into training and test sets
train_size = int(len(data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create sequences of length 'num_steps' for training
X_train, y_train = [], []
for i in range(num_steps, len(train_data)):
    X_train.append(train_data[i - num_steps:i, 0])
    y_train.append(train_data[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape the input data for LSTM (samples, time steps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build the BiLSTM model
model = Sequential()
model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(X_train.shape[1], 1)))
model.add(Bidirectional(LSTM(50)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Prepare the test data
X_test, y_test = [], []
for i in range(num_steps, len(test_data)):
    X_test.append(test_data[i - num_steps:i, 0])
    y_test.append(test_data[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)

# Reshape the input data for LSTM (samples, time steps, features)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Make predictions on the test data
predictions = scaler.inverse_transform(model.predict(X_test))

# Calculate evaluation metrics
mae = mean_absolute_error(scaler.inverse_transform(y_test.reshape(-1, 1)), predictions)
mape = np.mean(np.abs((scaler.inverse_transform(y_test.reshape(-1, 1)) - predictions) / scaler.inverse_transform(y_test.reshape(-1, 1)))) * 100
rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(y_test.reshape(-1, 1)), predictions))
r2 = r2_score(scaler.inverse_transform(y_test.reshape(-1, 1)), predictions)

print("MAE:", mae)
print("MAPE:", mape)
print("RMSE:", rmse)
print("R^2:", r2)
