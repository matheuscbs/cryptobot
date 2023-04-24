import os

from dotenv import load_dotenv

from crypto.ethereumforecast import ExchangeBuySellRobot

load_dotenv()

api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_API_SECRET")

if __name__ == '__main__':
    robot = ExchangeBuySellRobot(api_key, api_secret)
    robot.run()
