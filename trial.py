import requests
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Replace with your CryptoCompare API key
API_KEY = 'YOUR_API_KEY'
URL = 'https://min-api.cryptocompare.com/data/v2/histoday'

def get_crypto_data(symbol, start_date, end_date):
    params = {
        'fsym': symbol,
        'tsym': 'USD',
        'limit': 2000,  # Adjust if needed
        'api_key': API_KEY,
        'toTs': int(datetime.strptime(end_date, '%Y-%m-%d').timestamp()),
        'e': 'CCCAGG'
    }
    response = requests.get(URL, params=params)
    data = response.json()['Data']['Data']
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df[(df['time'] >= start_date) & (df['time'] <= end_date)]
    return df

def create_lagged_features(df, lag=1):
    df[f'lag_{lag}'] = df['close'].shift(lag)
    df = df.dropna()
    return df

# Fetch data for 2023 and 2024
data_2023 = get_crypto_data('BTC', '2023-01-01', '2023-12-31')
data_2024 = get_crypto_data('BTC', '2024-01-01', datetime.now().strftime('%Y-%m-%d'))

# Create lagged features
data_2023 = create_lagged_features(data_2023)
data_2024 = create_lagged_features(data_2024)

# Use lagged features and current close price as target
features = ['lag_1']
X_2023 = data_2023[features]
y_2023 = data_2023['close']
X_2024 = data_2024[features]
y_2024 = data_2024['close']

# Train-test split for 2023 data
X_train, X_test, y_train, y_test = train_test_split(X_2023, y_2023, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set and 2024 data
y_test_pred = model.predict(X_test)
y_2024_pred = model.predict(X_2024)

# Calculate R^2 score for 2024 data
r22= r2_score(y_test, y_test_pred)
r2 = r2_score(y_2024, y_2024_pred)
print(f'R^2 score for 2024 predictions: {r2:.2f}, {r22:.2f}')

# Plot the actual data for 2023, actual data for 2024, and predicted data for 2024
plt.figure(figsize=(14, 7))

# Plot actual 2023 data
plt.plot(data_2023['time'], data_2023['close'], label='Actual 2023', color='blue')

# Plot actual 2024 data
plt.plot(data_2024['time'], data_2024['close'], label='Actual 2024', color='green')

# Plot predicted 2024 data
plt.plot(data_2024['time'], y_2024_pred, label='Predicted 2024', color='red', linestyle='--')

plt.title('Cryptocurrency Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()
