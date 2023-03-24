from ethereumForecast import EthereumForecast

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