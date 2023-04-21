import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from binance.client import Client
from prophet import Prophet


class EthereumForecast:
    def __init__(self, api_key, api_secret):
        self.historical_data = None
        self.current_value = None
        self.client = Client(api_key, api_secret)

    def get_historical_data(self):
        # Defina o intervalo de tempo desejado, no exemplo abaixo é de 1 dia (Client.KLINE_INTERVAL_1DAY).
        klines = self.client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_3DAY, "1 year ago UTC")

        self.historical_data = []
        for row in klines:
            timestamp, open, high, low, close, volume, close_time, quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore = row
            self.historical_data.append([dt.datetime.fromtimestamp(timestamp / 1000.0), float(close)])

        self.historical_data = np.array(self.historical_data)
        if self.historical_data is not None:
            self.historical_data[:, 1] = np.vectorize(np.log)(self.historical_data[:, 1])
            self.current_value = np.exp(self.historical_data[-1, 1])
        else:
            print('Error in fetching data.')

    def run_prophet(self):
        m = Prophet(daily_seasonality=True)
        df = self.historical_data.copy()
        df = np.column_stack((df[:, 0], df[:, 1]))
        df = np.column_stack((df[:, 0], df[:, 1], np.zeros(df.shape[0])))
        df = np.column_stack((df[:, 0], df[:, 1], np.zeros(df.shape[0]), np.zeros(df.shape[0])))
        df = np.column_stack((df[:, 0], df[:, 1], np.zeros(df.shape[0]), np.zeros(df.shape[0]), np.zeros(df.shape[0])))
        df = pd.DataFrame(df, columns=['ds', 'y', 'additive_terms', 'daily', 'additive_terms_upper'])
        m.fit(df)
        future = m.make_future_dataframe(periods=60, freq='min')
        forecast = m.predict(future)
        return forecast.iloc[-1]['yhat'], forecast

    def run_markov(self):
        data = self.historical_data[:, 1]
        mean = np.mean(data)
        variance = np.var(data)
        current_price = self.current_value
        time_period = 0.5
        mean_future_price = current_price * np.exp((mean - 0.5 * variance) * time_period)
        return mean_future_price

    def plot_forecasts(self, prophet_forecast):
        plt.figure(figsize=(12, 6))
        plt.plot(self.historical_data[:, 0], np.vectorize(np.exp)(self.historical_data[:, 1]), label='Historical Data')
        plt.plot(prophet_forecast['ds'], np.vectorize(np.exp)(prophet_forecast['yhat']), label='Prophet Prediction')
        markov_prediction = self.run_markov()
        markov_future = pd.DataFrame({'ds': prophet_forecast['ds'], 'yhat': np.full(prophet_forecast['yhat'].shape, markov_prediction)})
        plt.plot(markov_future['ds'], np.vectorize(np.exp)(markov_future['yhat']), label='Markov Prediction')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('Historical Data, Prophet Prediction and Markov Prediction')
        plt.legend()
        plt.show()

    def get_recommendation(self):
        prophet_prediction, prophet_forecast = self.run_prophet()
        markov_prediction = self.run_markov()
        current_value = self.current_value
        if current_value < markov_prediction and current_value < prophet_prediction:
            return "Buy"
        elif current_value > markov_prediction and current_value > prophet_prediction:
            return "Sell"
        else:
            return "Hold"


class ForecastComparison:
    def __init__(self, prophet_prediction, markov_prediction, real_value):
        self.prophet_prediction = prophet_prediction
        self.markov_prediction = markov_prediction
        self.real_value = real_value

    def plot(self):
        labels = ['Prophet', 'Markov', 'Real Value']
        values = [self.prophet_prediction, self.markov_prediction, self.real_value]

        plt.bar(labels, values)
        plt.xlabel('Prediction Model')
        plt.ylabel('Value')
        plt.title('Prophet vs Markov vs Real Value')
        plt.show()


class EthereumBuySellRobot:
    def __init__(self, api_key, api_secret):
        self.eth_forecast = EthereumForecast(api_key, api_secret)

    def make_trade_decision(self):
        self.eth_forecast.get_historical_data()
        current_value = self.eth_forecast.current_value
        prophet_prediction, prophet_forecast = self.eth_forecast.run_prophet()
        markov_prediction = self.eth_forecast.run_markov()
        recommendation = self.eth_forecast.get_recommendation()
        return prophet_prediction, prophet_forecast, markov_prediction, current_value, recommendation

    def run(self):
        prophet_prediction, prophet_forecast, markov_prediction, current_value, recommendation = self.make_trade_decision()
        self.eth_forecast.plot_forecasts(prophet_forecast)
        comparison = ForecastComparison(prophet_prediction, markov_prediction, current_value)
        comparison.plot()
        print("Trade decision:", recommendation)
