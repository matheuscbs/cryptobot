import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from binance.client import Client
from prophet import Prophet
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


class EthereumForecast:
    def __init__(self, api_key, api_secret):
        self.historical_data = None
        self.current_value = None
        self.client = Client(api_key, api_secret)

    def get_historical_data(self):
        klines = self.client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1DAY, "1 year ago UTC")
        self.historical_data = []
        for row in klines:
            timestamp, open, high, low, close, volume, close_time, quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore = row
            self.historical_data.append([dt.datetime.fromtimestamp(timestamp / 1000.0), float(close)])
        self.historical_data = np.array(self.historical_data)
        if self.historical_data is not None:
            self.current_value = self.historical_data[-1, 1]
        else:
            print('Error in fetching data.')

    def run_prophet(self):
        m = Prophet(daily_seasonality=True)
        df = pd.DataFrame(self.historical_data, columns=['ds', 'y'])
        m.fit(df)
        future = m.make_future_dataframe(periods=59, freq='D')
        forecast = m.predict(future)
        return forecast['yhat'], forecast

    def run_markov_regime_switching(self):
        data = self.historical_data[:, 1].astype(float)
        log_returns = np.log(data[1:] / data[:-1])

        # Escolha o número de regimes (neste caso, 2)
        model = MarkovRegression(log_returns, k_regimes=2, trend='nc', switching_variance=True)
        res = model.fit()

        predicted_log_returns = res.smoothed_marginal_probabilities.iloc[-1].dot(res.params)

        # Prever o preço futuro em 59 dias
        time_period = 59
        future_price = data[-1] * np.exp(predicted_log_returns * time_period)

        return future_price

    def plot_forecasts(self, prophet_forecast_all):
        prophet_forecast, forecast = prophet_forecast_all

        plt.figure(figsize=(12, 6))
        plt.plot(self.historical_data[:, 0], self.historical_data[:, 1], label='Historical Data')
        plt.plot(forecast['ds'], prophet_forecast, label='Prophet Prediction')

        markov_prediction = self.run_markov_regime_switching()
        markov_start_date = forecast['ds'].iloc[-59]
        markov_end_date = markov_start_date + dt.timedelta(days=58)
        markov_future_dates = pd.date_range(markov_start_date, markov_end_date, freq='D')
        markov_forecast = pd.DataFrame({'ds': markov_future_dates, 'yhat': np.linspace(markov_prediction, markov_prediction, markov_future_dates.shape[0])})

        plt.plot(markov_forecast['ds'], markov_forecast['yhat'], label='Markov Regime-Switching Prediction')

        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('Historical Data, Prophet Prediction, and Markov Regime-Switching Prediction')
        plt.legend()
        plt.show()

    def get_recommendation(self):
        prophet_prediction, _ = self.run_prophet()
        prophet_prediction = prophet_prediction.iloc[-1]
        markov_prediction = self.run_markov_regime_switching()
        current_value = self.current_value
        if current_value < markov_prediction and current_value < prophet_prediction:
            return "Buy"
        elif current_value > markov_prediction and current_value > prophet_prediction:
            return "Sell"
        else:
            return "Hold"


class ForecastComparison:
    def __init__(self, prophet_forecast, markov_forecast, real_value):
        self.prophet_forecast = prophet_forecast
        self.markov_forecast = markov_forecast
        self.real_value = real_value

    def plot(self):
        labels = ['Prophet', 'Markov Regime-Switching', 'Real Value']
        prophet_last_prediction = self.prophet_forecast.iloc[-1]
        markov_last_prediction = self.markov_forecast['yhat'].iloc[-1]
        values = [prophet_last_prediction, markov_last_prediction, self.real_value]

        plt.bar(labels, values)
        plt.xlabel('Prediction Model')
        plt.ylabel('Value')
        plt.title('Prophet vs Markov Regime-Switching vs Real Value')
        plt.show()


class EthereumBuySellRobot:
    def __init__(self, api_key, api_secret):
        self.eth_forecast = EthereumForecast(api_key, api_secret)

    def make_trade_decision(self):
        self.eth_forecast.get_historical_data()
        current_value = self.eth_forecast.current_value
        prophet_forecast, prophet_full_forecast = self.eth_forecast.run_prophet()
        markov_forecast = self.eth_forecast.run_markov_regime_switching()
        recommendation = self.eth_forecast.get_recommendation()
        return prophet_forecast, prophet_full_forecast, markov_forecast, current_value, recommendation

    def run(self):
        prophet_forecast, prophet_full_forecast, markov_forecast, current_value, recommendation = self.make_trade_decision()
        self.eth_forecast.plot_forecasts((prophet_forecast, prophet_full_forecast))
        markov_forecast_df = pd.DataFrame({'ds': prophet_full_forecast['ds'].iloc[-59:], 'yhat': np.full(59, markov_forecast)})
        comparison = ForecastComparison(prophet_forecast, markov_forecast_df, current_value)
        comparison.plot()
        print("Trade decision:", recommendation)
