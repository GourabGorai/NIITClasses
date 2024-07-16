import tkinter as tk
from tkinter import ttk, messagebox
import requests
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
        messagebox.showerror("Error", f"Error fetching data: {e}")
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
        messagebox.showerror("Error", f"Error fetching current price: {e}")
        return None

def train_and_predict(symbol):
    data_2023 = get_crypto_data(symbol, '2023-01-01', '2023-12-31')
    data_2024 = get_crypto_data(symbol, '2024-01-01', datetime.now().strftime('%Y-%m-%d'))
    data_2023 = create_lagged_features(data_2023, lag=1)
    data_2024 = create_lagged_features(data_2024, lag=1)

    features = ['lag_1']
    X_2023 = data_2023[features]
    y_2023 = data_2023['close']
    X_2024 = data_2024[features]
    y_2024 = data_2024['close']

    X_train, X_test, y_train, y_test = train_test_split(X_2023, y_2023, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    cv_scores = cross_val_score(model, X_2023, y_2023, cv=5)
    y_test_pred = model.predict(X_test)
    y_2024_pred = model.predict(X_2024)

    r2 = r2_score(y_2024, y_2024_pred)
    rmse = mean_squared_error(y_2024, y_2024_pred, squared=False)

    plot_data(data_2023, data_2024, y_2024_pred)
    return model, r2, rmse

def plot_data(data_2023, data_2024, y_2024_pred):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(data_2023['time'], data_2023['close'], label='Actual 2023', color='blue')
    ax.plot(data_2024['time'], data_2024['close'], label='Actual 2024', color='green')
    ax.plot(data_2024['time'], y_2024_pred, label='Predicted 2024', color='red', linestyle='--')
    ax.set_title('Cryptocurrency Price Prediction')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    ax.grid(True)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=8, column=0, columnspan=2, padx=10, pady=10)

def predict_price_for_date(model, lagged_price, date):
    input_data = pd.DataFrame({'lag_1': [lagged_price]})
    predicted_price = model.predict(input_data)
    return predicted_price[0]

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

def on_train():
    symbol = symbol_entry.get().upper()
    model, r2, rmse = train_and_predict(symbol)
    r2_label.config(text=f"R² Score: {r2:.2f}")
    rmse_label.config(text=f"RMSE: {rmse:.2f}")
    predict_button.config(state=tk.NORMAL)
    predict_button.bind("<Button-1>", lambda event, m=model: on_predict(m))

def on_predict(model):
    future_date = future_date_entry.get()
    symbol = symbol_entry.get().upper()
    data_2023 = get_crypto_data(symbol, '2023-01-01', '2023-12-31')
    data_2024 = get_crypto_data(symbol, '2024-01-01', datetime.now().strftime('%Y-%m-%d'))
    data = pd.concat([data_2023, data_2024])
    predicted_price = predict_future_price(model, data, future_date)
    prediction_label.config(text=f"Predicted price for {future_date}: ${predicted_price:.2f}")

def on_current_price():
    symbol = symbol_entry.get().upper()
    current_price = get_current_price(symbol)
    if current_price:
        current_price_label.config(text=f"Current price: ${current_price:.2f}")

# Tkinter GUI setup
root = tk.Tk()
root.title("Cryptocurrency Price Prediction")

ttk.Label(root, text="Cryptocurrency Symbol:").grid(row=0, column=0, padx=10, pady=10)
symbol_entry = ttk.Entry(root)
symbol_entry.grid(row=0, column=1, padx=10, pady=10)

ttk.Button(root, text="Train Model", command=on_train).grid(row=1, column=0, columnspan=2, padx=10, pady=10)

r2_label = ttk.Label(root, text="R² Score: N/A")
r2_label.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

rmse_label = ttk.Label(root, text="RMSE: N/A")
rmse_label.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

ttk.Label(root, text="Future Date (YYYY-MM-DD):").grid(row=4, column=0, padx=10, pady=10)
future_date_entry = ttk.Entry(root)
future_date_entry.grid(row=4, column=1, padx=10, pady=10)

predict_button = ttk.Button(root, text="Predict Future Price")
predict_button.grid(row=5, column=0, columnspan=2, padx=10, pady=10)
predict_button.config(state=tk.DISABLED)

prediction_label = ttk.Label(root, text="Predicted price for future date: N/A")
prediction_label.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

ttk.Button(root, text="Get Current Price", command=on_current_price).grid(row=7, column=0, columnspan=2, padx=10, pady=10)

current_price_label = ttk.Label(root, text="Current price: N/A")
current_price_label.grid(row=8, column=0, columnspan=2, padx=10, pady=10)

root.mainloop()
