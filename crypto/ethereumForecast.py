import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from binance.client import Client
from prophet import Prophet
from scipy.linalg import pinv
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


class DataFetcher:
    def __init__(self, api_key, api_secret):
        self.client = Client(api_key, api_secret)

    def get_historical_data(self, symbol, interval, duration):
        klines = self.client.get_historical_klines(symbol, interval, duration)
        data = []

        for row in klines:
            (
                timestamp,
                open,
                high,
                low,
                close,
                volume,
                close_time,
                quote_asset_volume,
                number_of_trades,
                taker_buy_base_asset_volume,
                taker_buy_quote_asset_volume,
                ignore,
            ) = row
            data.append([dt.datetime.fromtimestamp(timestamp / 1000.0), float(close)])

        data_df = pd.DataFrame(data, columns=["ds", "y"])
        data_df = data_df.interpolate(method="time")
        data_df = data_df.fillna(data_df.bfill())

        return data_df


class ExchangeForecast:
    def __init__(self, data_fetcher):
        self.historical_data = None
        self.current_value = None
        self.data_fetcher = data_fetcher

    def fetch_historical_data(self, symbol, interval, duration):
        self.historical_data = self.data_fetcher.get_historical_data(
            symbol, interval, duration
        )
        self.current_value = self.historical_data.iloc[-1, 1]
        if self.historical_data.isnull().sum().sum() != 0:
            raise ValueError("Historical data contains NaN or missing values.")

    def run_prophet(self):
        m = Prophet(daily_seasonality=False, weekly_seasonality=True)
        m.fit(self.historical_data)
        future = m.make_future_dataframe(periods=59, freq="D")
        forecast = m.predict(future)
        return forecast, future

    def run_markov_regime_switching(self, log_returns, k_regimes=2):
        if k_regimes >= len(log_returns):
            raise ValueError(
                "The number of states must be less than the number of observations."
            )
        model = CustomMarkovRegression(
            log_returns, k_regimes=k_regimes, trend="n", switching_variance=True
        )

        print("Shape of log_returns: ", log_returns.shape)
        if log_returns.ndim != 1:
            raise ValueError("log_returns must be a 1-dimensional array")

        res = model.fit()

        smoothed_marginal_probabilities_df = pd.DataFrame(
            res.smoothed_marginal_probabilities
        )
        params_per_regime = res.params.reshape(2, -1)
        predicted_log_returns = smoothed_marginal_probabilities_df.iloc[-1].dot(
            params_per_regime
        )

        avg_predicted_log_returns = np.dot(
            smoothed_marginal_probabilities_df.iloc[-1], predicted_log_returns
        )

        time_period = 59
        future_prices = np.zeros(time_period)
        for i in range(1, time_period + 1):
            future_prices[i - 1] = self.current_value * np.exp(
                avg_predicted_log_returns * i
            )

        return future_prices

    def plot_forecasts(self, prophet_forecast_all, markov_future_prices):
        prophet_full_forecast, future = prophet_forecast_all

        plt.figure(figsize=(12, 6))
        plt.plot(
            pd.to_datetime(self.historical_data.iloc[:, 0]),
            self.historical_data.iloc[:, 1],
            label="Historical Data",
        )
        plt.plot(
            prophet_full_forecast["ds"],
            prophet_full_forecast["yhat"],
            label="Prophet Prediction",
        )

        markov_start_date = future["ds"].iloc[-59]
        markov_end_date = markov_start_date + dt.timedelta(days=58)
        markov_future_dates = pd.date_range(
            markov_start_date, markov_end_date, freq="D"
        )
        markov_forecast = pd.DataFrame(
            {"ds": markov_future_dates, "yhat": markov_future_prices}
        )

        plt.plot(
            markov_forecast["ds"],
            markov_forecast["yhat"],
            label="Markov Regime-Switching Prediction",
        )
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.title(
            "Historical Data, Prophet Prediction, and Markov Regime-Switching Prediction"
        )
        plt.legend()
        plt.show()

    def get_recommendation(self, prophet_forecast, markov_prediction):
        if (
            self.current_value < markov_prediction[0]
            and self.current_value < prophet_forecast["yhat"].tail(1).values[0]
        ):
            return "Buy"
        elif (
            self.current_value > markov_prediction[0]
            and self.current_value > prophet_forecast["yhat"].tail(1).values[0]
        ):
            return "Sell"
        else:
            return "Hold"


class CustomMarkovRegression(MarkovRegression):
    def initial_probabilities(self, A, *args, **kwargs):
        A = np.reshape(A, (self.k_regimes, self.k_regimes))

        if A.shape[0] != A.shape[1]:
            raise ValueError("The transition matrix A must be square.")

        mask = np.isfinite(A).all(axis=1)
        A = A[mask, :]

        try:
            eigvals, eigvecs = np.linalg.eig(np.transpose(A))
            steady_state_vec = eigvecs[:, np.isclose(eigvals, 1)]
            probabilities = steady_state_vec / steady_state_vec.sum()
        except np.linalg.LinAlgError:
            raise RuntimeError(
                "Steady-state probabilities could not be" " constructed."
            )

        return probabilities.flatten()

    def transform_params(self, unconstrained):
        constrained = unconstrained.copy()
        constrained[-self.k_regimes :] = np.exp(constrained[-self.k_regimes :])
        constrained[: -self.k_regimes] = pinv(
            np.eye(self.exog.shape[1] * self.k_regimes)
            - self.Z @ np.kron(np.diag(constrained[-self.k_regimes :]), self.R)
        ).dot(self.exog.T @ self.endog)
        return constrained

    def untransform_params(self, constrained):
        unconstrained = constrained.copy()
        unconstrained[-self.k_regimes :] = np.log(unconstrained[-self.k_regimes :])
        unconstrained[: -self.k_regimes] = pinv(
            np.eye(self.exog.shape[1] * self.k_regimes)
            - self.Z
            @ np.kron(np.diag(np.exp(unconstrained[-self.k_regimes :])), self.R)
        ).dot(self.exog.T @ self.endog)
        return unconstrained

    def loglikeobs(self, params):
        params = np.asarray(params)
        if not self.enforce_stationarity:
            params = self.transform_params(params)
        Z, R, Q, transition_probabilities = self._unpack_params(params)
        (
            self.filtered_marginal_probabilities,
            self.filtered_joint_probabilities,
        ) = self._kalman_filter(Z, R, Q, transition_probabilities)
        (
            self.smoothed_marginal_probabilities,
            self.smoothed_joint_probabilities,
        ) = self._kalman_smoother(Z, R, Q, transition_probabilities)
        loglike = np.sum(np.log(np.sum(self.filtered_joint_probabilities, axis=0)))
        return loglike


class ForecastComparison:
    def __init__(self, prophet_forecast, markov_forecast, real_value):
        self.prophet_forecast = prophet_forecast
        self.markov_forecast = markov_forecast
        self.real_value = real_value

    def plot(self):
        labels = ["Prophet", "Markov Regime-Switching", "Real Value"]
        prophet_last_prediction = self.prophet_forecast["yhat"].iloc[-1]
        markov_last_prediction = self.markov_forecast.iloc[-1]["yhat"]
        values = [prophet_last_prediction, markov_last_prediction, self.real_value]

        plt.bar(labels, values)
        plt.xlabel("Prediction Model")
        plt.ylabel("Value")
        plt.title("Prophet vs Markov Regime-Switching vs Real Value")
        plt.show()


class ExchangeBuySellRobot:
    def __init__(self, api_key, api_secret):
        data_fetcher = DataFetcher(api_key, api_secret)
        self.exchange_forecast = ExchangeForecast(data_fetcher)

    def make_trade_decision(self):
        symbol = "BTCUSDT"  # símbolo do Bitcoin
        interval = Client.KLINE_INTERVAL_5MINUTE  # intervalo de 5 minutos
        duration = "3 days ago UTC"  # dados dos últimos 3 dias
        self.exchange_forecast.fetch_historical_data(symbol, interval, duration)
        current_value = self.exchange_forecast.current_value
        prophet_forecast, future = self.exchange_forecast.run_prophet()
        log_returns = pd.Series(
            self.exchange_forecast.historical_data["y"]
            / self.exchange_forecast.historical_data["y"].shift(1)
        ).dropna()
        log_returns = log_returns.reset_index(drop=True)

        if log_returns.isnull().sum() != 0:
            raise ValueError("Log returns contains NaN or missing values.")

        markov_forecast = self.exchange_forecast.run_markov_regime_switching(
            log_returns
        )

        recommendation = self.exchange_forecast.get_recommendation(
            prophet_forecast, markov_forecast
        )

        return prophet_forecast, future, markov_forecast, current_value, recommendation

    def run(self):
        (
            prophet_forecast,
            future,
            markov_forecast,
            current_value,
            recommendation,
        ) = self.make_trade_decision()

        self.exchange_forecast.plot_forecasts(
            (prophet_forecast, future), markov_forecast
        )
        markov_forecast_df = pd.DataFrame(
            {"ds": future["ds"].iloc[-59:], "yhat": markov_forecast}
        )
        comparison = ForecastComparison(
            prophet_forecast, markov_forecast_df, current_value
        )
        comparison.plot()
        print("Trade decision:", recommendation)
