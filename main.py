import asyncio
import aiohttp
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

WINDOW = 60
THRESHOLD = 0.01
INTERVAL = '1m'


async def get_price_history(symbol, interval, start_time, end_time):
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&startTime={start_time}&endTime={end_time}'
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            json = await response.json()
    return [float(x[4]) for x in json]


async def main():
    while True:
        now = datetime.utcnow()
        end_time = int(now.timestamp() * 1000)
        start_time = int((now - timedelta(minutes=WINDOW)).timestamp() * 1000)

        eth_prices_task = asyncio.create_task(get_price_history('ETHUSDT', INTERVAL, start_time, end_time))
        btc_prices_task = asyncio.create_task(get_price_history('BTCUSDT', INTERVAL, start_time, end_time))

        eth_prices = np.array(await eth_prices_task).reshape(-1, 1)
        btc_prices = np.array(await btc_prices_task).reshape(-1, 1)

        model = LinearRegression()
        model.fit(btc_prices, eth_prices)

        eth_adjusted = eth_prices - (model.coef_ * btc_prices + model.intercept_)
        eth_return = (eth_adjusted[-1] - eth_adjusted[0]) / eth_prices[-1]
        eth_return = eth_return[0].round(5)

        if abs(eth_return) >= THRESHOLD:
            print(f'{now}: ETH has moved by {eth_return * 100}%')

        await asyncio.sleep(60)

if __name__ == '__main__':
    asyncio.run(main())
