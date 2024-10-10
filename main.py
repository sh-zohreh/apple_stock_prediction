import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input  # Import Input layer

# 1. Download stock data (from Yahoo Finance)
stock_data = yf.download('AAPL', start='2010-01-01', end='2023-01-01')  # Example with Apple stock

# 2. View the data
print(stock_data.head())

# 3. Prepare the data
# Only consider the 'Close' column as the closing price data
close_prices = stock_data['Close'].values.reshape(-1, 1)

# 4. Normalize the data with MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# 5. Split the data into training and testing datasets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# 6. Create sequences for training data
def create_sequences(data, seq_length):
    x, y = [], []
    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

seq_length = 60  # Use the last 60 days to predict the next day
x_train, y_train = create_sequences(train_data, seq_length)
x_test, y_test = create_sequences(test_data, seq_length)

# 7. Reshape the data for LSTM input
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# 8. Build the LSTM model
model = Sequential()

# Using Input layer to define input shape
model.add(Input(shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(units=25))
model.add(Dense(units=1))

# 9. Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, batch_size=64, epochs=10)

# 10. Predict prices on test data
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)  # Transform data back to original scale

# 11. Analyze results
plt.figure(figsize=(16, 8))
plt.plot(stock_data.index[train_size+seq_length:], close_prices[train_size+seq_length:], color='blue', label='Actual Prices')
plt.plot(stock_data.index[train_size+seq_length:], predictions, color='red', label='Predicted Prices')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()