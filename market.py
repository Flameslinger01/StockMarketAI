# import
import extract_data as ed
import gymnasium as gym
from gymnasium import spaces
from gymnasium import utils
import numpy as np
from numpy.random import default_rng
from gymnasium.envs.registration import register

register(
     id="stockmarketAI-v0",
     entry_point="market:marketEnv",
)

class marketEnv(gym.Env):
    # Initialization
    def __init__(self, numberOfDays=2, numberOfStocks=3, data='data/minute.csv'):
        self.data = data
        self.numberOfDays = numberOfDays
        self.numberOfStocks = numberOfStocks
        self.fullMarket, self.tickerList = ed.generateStockList(data, self.np_random.integers(0,numberOfDays-1), 17)
        self.startingMoney = np.random.randint(30000, 40000)
        self.currentMoney = self.startingMoney
        self.currentTime, self.stockWorth = 0, 0
        self.numShares = np.zeros([len(self.fullMarket),1], np.float32)
        self.buyPrice = np.zeros([len(self.fullMarket),1], np.float32)
        rng = default_rng()
        self.stockIndexes = rng.choice(len(self.fullMarket), size=numberOfStocks, replace=False)
        # Observation Space
        self.observation_space = gym.spaces.flatten_space(spaces.Dict(
            {
                "botData": spaces.Box(np.array([25000, 0]), np.array([3.4028235e+38, 390]), (2,), np.float32), # current money, current time
                "investmentData": spaces.Box(np.zeros([len(self.fullMarket)], np.float32), np.full_like(np.zeros([len(self.fullMarket)]), [3.4028235e+38]), (len(self.fullMarket),), np.float32), # numshares
                "stocks": spaces.Box(0, 3.4028235e+38, (len(self.fullMarket),5), np.float32), # Open High Low Close Volume
            }
        ))

        # Action Space

        self.action_space = gym.spaces.flatten_space(spaces.Dict(
            {
                # Stock Name is key, action is box with how many shares to do something with (negative is sell, 0 is hold, positive is buy)
                stock.ticker: spaces.Box(-self.numShares[self.fullMarket.index(stock)],(max(0,self.currentMoney-25000))/(np.float32(stock.open[self.currentTime])*np.float32(numberOfStocks)),dtype=np.float32) # minimum of how many shares had, max of above minimum divided by stock price further divided by number of stocks 
                for stock in self.fullMarket if self.fullMarket.index(stock) in self.stockIndexes
            }
        ))

    def _get_obs(self):
        # set up stock history
        open = [[np.float32(stock.open[self.currentTime])] for stock in self.fullMarket]
        close = [[np.float32(stock.close[self.currentTime])] for stock in self.fullMarket]
        low = [[np.float32(stock.low[self.currentTime])] for stock in self.fullMarket]
        high = [[np.float32(stock.high[self.currentTime])] for stock in self.fullMarket]
        volume = [[np.float32(stock.volume[self.currentTime])] for stock in self.fullMarket]
        test2 = [[[self.currentMoney]], [[self.currentTime]], self.numShares.tolist(), open, close, low, high, volume]
        test2 = [val for sublist in test2 for val in sublist]
        test2 = [val for sublist in test2 for val in sublist]
        return np.array(test2, dtype= np.float32)
    
 
    def _get_info(self):
        self.stockWorth = sum((np.float32(stock.open[self.currentTime]) * self.numShares[self.fullMarket.index(stock)]) for stock in self.fullMarket)
        return {
            "timeLeft": (390 - self.currentTime),
            "profit": (self.currentMoney + self.stockWorth) - self.startingMoney
        }

    # Reset 
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        # Variables
        self.fullMarket, self.tickerList = ed.generateStockList(self.data, self.np_random.integers(0,self.numberOfDays-1), 17)
        self.startingMoney = np.random.randint(30000, 40000)
        self.currentMoney = self.startingMoney
        self.currentTime, self.stockWorth = 0, 0
        self.numShares = np.zeros([len(self.fullMarket),1], np.float32)
        self.buyPrice = np.zeros([len(self.fullMarket),1], np.float32)
        rng = default_rng()
        self.stockIndexes = rng.choice(len(self.fullMarket), size=self.numberOfStocks, replace=False)

        # Action Space
        self.action_space = gym.spaces.flatten_space(spaces.Dict(
            {
                # Stock Name is key, action is box with how many shares to do something with (negative is sell, 0 is hold, positive is buy)
                stock.ticker: spaces.Box(-self.numShares[self.fullMarket.index(stock)],(max(0,self.currentMoney-25000))/(np.float32(stock.open[self.currentTime])*np.float32(self.numberOfStocks)),dtype=np.float32) # minimum of how many shares had, max of above minimum divided by stock price further divided by number of stocks 
                for stock in self.fullMarket if self.fullMarket.index(stock) in self.stockIndexes
            }
        ))

        # Gather observations
        observation = self._get_obs()
        info = self._get_info()

        return observation, {}
        
    # Step 
    def step(self, action_dict):
        # Actions
        for stock, action in action_dict.items():
            self.numShares[self.tickerList.index(stock)] += np.float32(action)
            self.currentMoney -= np.float32(self.fullMarket[self.tickerList.index(stock)].close[int(self.currentTime)].item()) * np.float32(action.item())
        reward = 0 

        self.action_space = gym.spaces.flatten_space(spaces.Dict(
            {
                # Stock Name is key, action is box with how many shares to do something with (negative is sell, 0 is hold, positive is buy)
                stock.ticker: spaces.Box(-self.numShares[self.fullMarket.index(stock)],(max(0,self.currentMoney-25000))/(np.float32(stock.open[self.currentTime])*np.float32(self.numberOfStocks)),dtype=np.float32) # minimum of how many shares had, max of above minimum divided by stock price further divided by number of stocks 
                for stock in self.fullMarket if self.fullMarket.index(stock) in self.stockIndexes
            }
        ))
        terminated = False
        truncated = False
        reward += 1 if self.stockWorth + self.currentMoney - self.startingMoney > 0 else -1
        if self.currentTime == 390:
            terminated = True
        if self.currentMoney < 25000:
            truncated = True
        
        self.currentTime += 1
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, {}


market = marketEnv(2,3, 'data/minute.csv')
#temp = 0
#market._get_obs()
"""
while(True):
    for i in range(100):
        test = market.action_space.sample()
        # print(market.numShares)
        # print(market.action_space)
        market.step(test)
    if((market.currentMoney+market.stockWorth - market.startingMoney) > temp):
        print(str(market.currentMoney+market.stockWorth - market.startingMoney))
        temp = market.currentMoney+market.stockWorth - market.startingMoney
    market.reset()
    if(temp > 100):
        break


print("Starting Money: " + str(market.startingMoney) + ", Current Money: " + str(market.currentMoney+market.stockWorth))
print("Profit: " + str(market.currentMoney+market.stockWorth - market.startingMoney))
"""