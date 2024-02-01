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
    def __init__(self, numberOfDays, numberOfStocks, data):
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
        self.observation_space = spaces.Dict(
            {
                "botData": spaces.Box(np.array([25000, 0]), np.array([3.4028235e+38, 390]), (2,), np.float32), # current money, current time
                "investmentData": spaces.Box(np.zeros([numberOfStocks,2], np.float32), np.full_like(np.zeros([numberOfStocks,2]), [3.4028235e+38, 3.4028235e+38]), (numberOfStocks,2), np.float32), # numshares, buy price
                "stocks": spaces.Box(0, 3.4028235e+38, (numberOfStocks,5), np.float32), # Open High Low Close Volume
            }
        )

        # Action Space

        self.action_space = spaces.Dict(
            {
                # Stock Name is key, action is box with how many shares to do something with (negative is sell, 0 is hold, positive is buy)
                stock.ticker: spaces.Box(-self.numShares[self.fullMarket.index(stock)],(max(0,self.currentMoney-25000))/(np.float32(stock.open[self.currentTime])*np.float32(numberOfStocks)),dtype=np.float32) # minimum of how many shares had, max of above minimum divided by stock price further divided by number of stocks 
                for stock in self.fullMarket if self.fullMarket.index(stock) in self.stockIndexes
            }
        )

    def _get_obs(self):
        # set up stock history
        open = [[stock.open[i] for i in range(self.currentTime+1)] for stock in self.fullMarket]
        close = [[stock.close[i] for i in range(self.currentTime+1)] for stock in self.fullMarket]
        low = [[stock.low[i] for i in range(self.currentTime+1)] for stock in self.fullMarket]
        high = [[stock.high[i] for i in range(self.currentTime+1)] for stock in self.fullMarket]
        volume = [[stock.volume[i] for i in range(self.currentTime+1)] for stock in self.fullMarket]
        return { "botData" : np.array([self.currentMoney, self.currentTime]), "investmentData" :np.array([self.numShares,self.buyPrice]), "stocks" : np.array([open, close, low, high, volume]) }
    
 
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
        self.action_space = spaces.Dict(
            {
                # Stock Name is key, action is box with how many shares to do something with (negative is sell, 0 is hold, positive is buy)
                stock.ticker: spaces.Box(-self.numShares[self.fullMarket.index(stock)],(max(0,self.currentMoney-25000))/(np.float32(stock.open[self.currentTime])*np.float32(self.numberOfStocks)),dtype=np.float32) # minimum of how many shares had, max of above minimum divided by stock price further divided by number of stocks 
                for stock in self.fullMarket if self.fullMarket.index(stock) in self.stockIndexes
            }
        )

        # Gather observations
        observation = self._get_obs()
        info = self._get_info()

        return observation, info
        
    # Step 
    def step(self, action_dict):
        # Actions
        for stock, action in action_dict.items():
            self.numShares[self.tickerList.index(stock)] += np.float32(action)
            self.currentMoney -= np.float32(self.fullMarket[self.tickerList.index(stock)].close[int(self.currentTime)].item()) * np.float32(action.item())
        reward = 0 

        self.action_space = spaces.Dict(
            {
                # Stock Name is key, action is box with how many shares to do something with (negative is sell, 0 is hold, positive is buy)
                stock.ticker: spaces.Box(-self.numShares[self.fullMarket.index(stock)],(max(0,self.currentMoney-25000))/(np.float32(stock.open[self.currentTime])*np.float32(self.numberOfStocks)),dtype=np.float32) # minimum of how many shares had, max of above minimum divided by stock price further divided by number of stocks 
                for stock in self.fullMarket if self.fullMarket.index(stock) in self.stockIndexes
            }
        )
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

        return observation, reward, terminated, truncated, info

