# bitcoin
import requests
import numpy as np
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
!pip install gluonts matplotlib pandas
api_endpoint = "https://api.binance.com/api/v3/klines"
symbol = "BTCUSDT"
interval = "1d"
limit = 1000
additional_limit = 1000
repeat = 100

print("Total data request:", limit - 1 + (additional_limit - 1) * repeat)

# Initial data retrieval
params = {'symbol': symbol, 'interval': interval, 'limit': limit}
klines_data = requests.get(api_endpoint, params=params).json()

end_time = klines_data[0][6]

klines_data = klines_data[::-1]
klines_data.pop()

# Fetch additional features (modify as needed)
for i in range(repeat):
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': additional_limit,
        'endTime': end_time
    }

    response = requests.get(api_endpoint, params=params).json()

    if len(response) == 1:
        print("No more data")
        break

    end_time = response[0][6]

    klines_data.extend(response[::-1])
    klines_data.pop()

print("Total data length:", len(klines_data))

# Create DataFrame with additional features
df = pd.DataFrame(klines_data)
df.columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
              'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
              'Taker buy quote asset volume', 'Ignore']

# Select relevant columns and convert to numeric
df = df[[ 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
         'Number of trades', 'Taker buy base asset volume']].apply(pd.to_numeric)

# Calculate additional features (modify as needed)
df['Change'] = 100 * (df['Close'] - df['Open']) / df['Open']

# Initialize the 'MA25' column
df['MA25'] = 0.0

for i in range(len(df) - 25):  # Adjusted loop range
    sum_close = 0
    for j in range(25):
        sum_close += df['Close'][i + j]
    df.loc[i, 'MA25'] = sum_close / 25

# Drop rows where moving average has not been computed yet (due to the initial window size)
df = df.dropna(subset=['MA25'])    

# Convert 'Open time' to datetime
df['ds'] = pd.to_datetime(df['Close time'], unit='ms')

# Select columns for Prophet model
df_prophet = df[['ds', 'Open', 'High', 'Low', 'Close', 'Volume', 'Change', 'MA25']]

# Rename columns to match Prophet requirements
df_prophet.columns = ['ds', 'Open', 'High', 'Low', 'y', 'Volume', 'Change', 'MA25']

# Initialize the Prophet model
model = Prophet(
    changepoint_prior_scale=10,seasonality_mode = 'additive')

# Fit the model with historical data
model.fit(df_prophet)

# Create a dataframe with future timestamps for prediction
future = model.make_future_dataframe(periods=200)  # Adjust the number of days as needed

# Make predictions
forecast = model.predict(future)

# Plot the forecast
fig = model.plot(forecast)
plt.title('Complete Forecast')

# Zoom in on a specific portion of the plot
start_date = '2022-11-01'  # Replace with your desired start date
end_date = '2025-11-01'    # Replace with your desired end date
plt.xlim(pd.to_datetime(start_date), pd.to_datetime(end_date))
plt.title(symbol)

# Show the plot
plt.show()
# Initialize the 'MA25' column
df['MA7'] = 0.0

for i in range(len(df) - 7):  # Adjusted loop range
    sum_close = 0
    for j in range(7):
        sum_close += df['Close'][i + j]
    df.loc[i, 'MA7'] = sum_close / 7

# Drop rows where moving average has not been computed yet (due to the initial window size)
df = df.dropna(subset=['MA7'])

df['MA14'] = 0.0

for i in range(len(df) - 14):  # Adjusted loop range
    sum_close = 0
    for j in range(14):
        sum_close += df['Close'][i + j]
    df.loc[i, 'MA14'] = sum_close / 14

# Drop rows where moving average has not been computed yet (due to the initial window size)
df = df.dropna(subset=['MA14'])    

df['MA28'] = 0.0

for i in range(len(df) - 28):  # Adjusted loop range
    sum_close = 0
    for j in range(28):
        sum_close += df['Close'][i + j]
    df.loc[i, 'MA28'] = sum_close / 28

# Drop rows where moving average has not been computed yet (due to the initial window size)
df = df.dropna(subset=['MA28'])    

df['MA49'] = 0.0

for i in range(len(df) - 49):  # Adjusted loop range
    sum_close = 0
    for j in range(49):
        sum_close += df['Close'][i + j]
    df.loc[i, 'MA49'] = sum_close / 49

# Drop rows where moving average has not been computed yet (due to the initial window size)
df = df.dropna(subset=['MA49'])    


df['MA99'] = 0.0

for i in range(len(df) - 99):  # Adjusted loop range
    sum_close = 0
    for j in range(99):
        sum_close += df['Close'][i + j]
    df.loc[i, 'MA99'] = sum_close / 99

# Drop rows where moving average has not been computed yet (due to the initial window size)
df = df.dropna(subset=['MA99'])    



# Convert 'Open time' to datetime
df['ds'] = pd.to_datetime(df['Close time'], unit='ms')

# Select columns for Prophet model
df_prophet = df[['ds', 'Open', 'High', 'Low', 'Close', 'Volume', 'Change', 'MA7', 'MA14', 'MA28', 'MA49', 'MA99']]

# Rename columns to match Prophet requirements
df_prophet.columns = ['ds', 'Open', 'High', 'Low', 'y', 'Volume', 'Change', 'MA7', 'MA14', 'MA28', 'MA49', 'MA99']





# Initialize the Prophet model
model = Prophet(
    changepoint_prior_scale=10,seasonality_mode = 'additive')

# Fit the model with historical data
model.fit(df_prophet)

# Create a dataframe with future timestamps for prediction
future = model.make_future_dataframe(periods=200)  # Adjust the number of days as needed

# Make predictions
forecast = model.predict(future)

# Plot the forecast
fig = model.plot(forecast)
plt.title('Complete Forecast')

# Zoom in on a specific portion of the plot
start_date = '2024-08-01'  # Replace with your desired start date
end_date = '2025-11-01'    # Replace with your desired end date
plt.xlim(pd.to_datetime(start_date), pd.to_datetime(end_date))
plt.title(symbol)

# Show the plot
plt.show()
