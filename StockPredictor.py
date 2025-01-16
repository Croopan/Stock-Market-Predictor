import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras._tf_keras.keras.layers import LSTM, Dense, Dropout, Input
from keras._tf_keras.keras.models import Sequential

# 1. Load the data
data = pd.read_csv('stock_data.csv')  # Replace with your stock data CSV file
test = pd.read_csv('test_data.csv')

close_prices = data['Close'].values.reshape(-1, 1)
test_prices = test['Close'].values.reshape(-1,1)

print(close_prices, test_prices)

# 2. Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

sequence_length = 60
x_train, y_train = [], []

for i in range(sequence_length, len(scaled_data)):
    x_train.append(scaled_data[i-sequence_length:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.expand_dims(x_train, axis=2)  # Adding a feature dimension

# 3. Build the RNN model
model = Sequential([
    Input(shape=(x_train.shape[1], 1)),  # Define the input shape here
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1)  # Output layer
])

model.compile(optimizer='adam', loss='mean_squared_error')

# 4. Train the model
model.fit(x_train, y_train, epochs=20, batch_size=32)

# 5. Prepare test data
test_data = scaled_data[-(sequence_length + 100):]  # Last 100 points (for example)
x_test, y_test = [], []

for i in range(sequence_length, len(test_data)):
    x_test.append(test_data[i-sequence_length:i, 0])
    y_test.append(test_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.expand_dims(x_test, axis=2)

# 6. Make predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# 7. Visualize the results
plt.figure(figsize=(12, 6))
plt.plot(y_test, color='blue', label='Actual Prices')
plt.plot(predictions, color='red', label='Predicted Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()