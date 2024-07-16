import requests
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Replace with your CryptoCompare API key
API_KEY = 'b2261e2089259e08fa16900e6d03ec16d69dd1dae13ce12d3fef99ac7bd0017e'
URL = 'https://min-api.cryptocompare.com/data/v2/histoday'
CURRENT_URL = 'https://min-api.cryptocompare.com/data/price'


def get_crypto_data(symbol, start_date, end_date):
    try:
        start_date_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
        end_date_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
        limit = (end_date_ts - start_date_ts) // (24 * 3600)  # Calculate the number of days

        if limit > 365:
            limit = 365

        params = {
            'fsym': symbol,
            'tsym': 'USD',
            'limit': limit,
            'api_key': API_KEY,
            'toTs': end_date_ts,
            'e': 'CCCAGG'
        }

        response = requests.get(URL, params=params)
        data = response.json()['Data']['Data']
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # Ensure the end date is included
        if df['time'].iloc[-1] != pd.to_datetime(end_date):
            additional_day_params = params.copy()
            additional_day_params['limit'] = 1
            additional_day_params['toTs'] = end_date_ts + 24 * 3600  # Add one more day to include end date
            additional_response = requests.get(URL, params=additional_day_params)
            additional_data = additional_response.json()['Data']['Data']
            additional_df = pd.DataFrame(additional_data)
            additional_df['time'] = pd.to_datetime(additional_df['time'], unit='s')
            df = pd.concat([df, additional_df])

        df = df[(df['time'] >= start_date) & (df['time'] <= end_date)]
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def create_lagged_features(df, lag=1):
    df[f'lag_{lag}'] = df['close'].shift(lag)
    df = df.dropna()
    return df

def get_current_price(symbol):
    try:
        params = {
            'fsym': symbol,
            'tsyms': 'USD',
            'api_key': API_KEY
        }
        response = requests.get(CURRENT_URL, params=params)
        data = response.json()
        return data['USD']
    except Exception as e:
        print(f"Error fetching current price: {e}")
        return None

# Fetch data for 2023 and 2024
data_2023 = get_crypto_data('BTC', '2023-01-01', '2023-12-31')
data_2024 = get_crypto_data('BTC', '2024-01-01', datetime.now().strftime('%Y-%m-%d'))
data_2023.to_csv('data_2023.csv')
data_2024.to_csv('data_2024.csv')

# Create lagged features
data_2023 = create_lagged_features(data_2023, lag=1)
data_2024 = create_lagged_features(data_2024, lag=1)

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

# Cross-validation
cv_scores = cross_val_score(model, X_2023, y_2023, cv=5)
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean CV score: {cv_scores.mean()}')

# Predict on the test set and 2024 data
y_test_pred = model.predict(X_test)
y_2024_pred = model.predict(X_2024)

# Calculate R^2 score and RMSE for 2024 data
r22 = r2_score(y_test, y_test_pred)
r2 = r2_score(y_2024, y_2024_pred)
print(f'R^2 score for 2024 predictions and the the test cases: {r2:.2f}, {r22:.2f}')

# Function to predict the price for a specific date
def predict_price_for_date(model, lagged_price, date):
    input_data = pd.DataFrame({'lag_1': [lagged_price]})
    predicted_price = model.predict(input_data)
    return predicted_price[0]

# Function to predict future prices
def predict_future_price(model, data, future_date):
    latest_date = data['time'].max()
    latest_price = data['close'].iloc[-1]

    future_date = datetime.strptime(future_date, '%Y-%m-%d')
    delta_days = (future_date - latest_date).days

    if delta_days <= 0:
        return "Future date must be after the latest date in the dataset."

    future_price = latest_price
    for _ in range(delta_days):
        future_price = predict_price_for_date(model, future_price, latest_date + timedelta(days=1))
        latest_date += timedelta(days=1)

    return future_price

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

# Retrieve the current price
current_date = datetime.now().strftime('%Y-%m-%d')
current_price = get_current_price('BTC')
if current_price:
    print(f"Current price for {current_date}: ${current_price:.2f}")
else:
    print("Error fetching current price.")

# Predict the price for a user-specified future date
user_date = input("Enter a future date (YYYY-MM-DD): ")
predicted_price = predict_future_price(model, pd.concat([data_2023, data_2024]), user_date)
print(f'Predicted price for {user_date}: ${predicted_price:.2f}')
