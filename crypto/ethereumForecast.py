import requests
import json
import datetime as dt
import numpy as np
from fbprophet import Prophet
import pandas as pd

class EthereumForecast:
    def __init__(self):
        self.historical_data = None
        self.current_value = None

    def get_historical_data(self):
    # API endpoint for 1 year of daily Ethereum prices in USD
        url = 'https://api.coingecko.com/api/v3/coins/ethereum/market_chart?vs_currency=usd&days=365'
        response = requests.get(url)
        if response.status_code == 200:
            self.historical_data = json.loads(response.content)['prices']
            self.historical_data = [[dt.datetime.fromtimestamp(row[0]/1000.0), row[1]] for row in self.historical_data]
            self.historical_data = np.array(self.historical_data)
            if self.historical_data is not None:
                if len(self.historical_data) > 0 and isinstance(self.historical_data[0][1], float):
                    self.historical_data = np.column_stack((self.historical_data[:,0], np.log(self.historical_data[:,1])))
                    self.current_value = np.exp(self.historical_data[-1,1])
            else:
                print('Error in fetching data. Status code: ', response.status_code)

    def run_prophet(self):
        m = prophet(daily_seasonality=True)
        df = self.historical_data.copy()
        df[:,1] = np.log(df[:,1])
        df = np.column_stack((df[:,0], df[:,1]))
        df = np.column_stack((df[:,0], df[:,1], np.zeros(df.shape[0])))
        df = np.column_stack((df[:,0], df[:,1], np.zeros(df.shape[0]), np.zeros(df.shape[0])))
        df = np.column_stack((df[:,0], df[:,1], np.zeros(df.shape[0]), np.zeros(df.shape[0]), np.zeros(df.shape[0])))
        df = pd.DataFrame(df, columns=['ds', 'y', 'additive_terms', 'daily', 'additive_terms_upper'])
        m.fit(df)
        future = m.make_future_dataframe(periods=30, freq='min')
        forecast = m.predict(future)
        return forecast.iloc[-1]['yhat']

    def run_markov(self):
        data = self.historical_data[:,1]
        mean = np.mean(data)
        variance = np.var(data)
        current_price = self.current_value
        time_period = 0.5  # 30 minutes in hours
        mean_future_price = current_price * np.exp(mean * time_period)
        return mean_future_price

    def get_recommendation(self):
        prophet_prediction = self.run_prophet()
        markov_prediction = self.run_markov()
        current_value = self.current_value
        if current_value < markov_prediction and current_value < prophet_prediction:
            return "Buy"
        elif current_value > markov_prediction and current_value > prophet_prediction:
            return "Sell"
        else:
            return "Hold"

class EthereumBuySellRobot:
    def __init__(self):
        self.eth_forecast = EthereumForecast()

    def make_trade_decision(self):
        self.eth_forecast.get_historical_data()
        current_value = self.eth_forecast.current_value
        recommendation = self.eth_forecast.get_recommendation()
        decision = None
        if recommendation == "Buy":
            decision = Buy(current_value)
        elif recommendation == "Sell":
            decision = Sell(current_value)
        else:
            decision = Hold(current_value)
        return decision

    def run(self):
        decision = self.make_trade_decision()
        # aqui deve ser implementada a lógica para realizar a operação de compra ou venda
        print("Trade decision:", decision)

if __name__ == '__main__':
    robot = EthereumBuySellRobot()
    robot.run()