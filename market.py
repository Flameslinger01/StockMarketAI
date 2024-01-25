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
     entry_point="market.envs:marketEnv",
     max_episode_steps=500,
)

fullMarket = ed.generateStockList('data/minute.csv', 0, 17)
numberOfDays = 2
numberOfStocks = 3
startingMoney = np.random.randint(30000, 40000)
currentMoney = startingMoney
currentTime = 4
numShares = np.zeros([numberOfStocks,1], np.float32)
numShares[1] = 100
buyPrice = np.zeros([numberOfStocks,1], np.float32)
rng = default_rng()
stockIndexes = rng.choice(len(fullMarket), size=numberOfStocks, replace=False)
for stock in fullMarket:
    print(fullMarket.index(stock))

open = [[stock.open[i] for i in range(currentTime+1)] for stock in fullMarket]
close = [[stock.close[i] for i in range(currentTime+1)] for stock in fullMarket]
low = [[stock.low[i] for i in range(currentTime+1)] for stock in fullMarket]
high = [[stock.high[i] for i in range(currentTime+1)] for stock in fullMarket]
volume = [[stock.volume[i] for i in range(currentTime+1)] for stock in fullMarket]
test = np.array([open, close, low, high, volume])

#print(test)
#print(test[fullMarket.index('AAPL')][0][0]) # Apple, Open, Minute 0

action_space = spaces.Dict(
            {
                stock.ticker: spaces.Box(-numShares[fullMarket.index(stock)],(currentMoney-25000)/(np.float32(stock.open[currentTime])*np.float32(numberOfStocks)),dtype=np.float32)
                for stock in fullMarket if fullMarket.index(stock) in stockIndexes
            }
        )
for stock, action in action_space.sample().items():
    print(stock, action)
print(action_space.sample()) #[fullMarket[0].ticker]

class marketEnv(gym.Env):
    # Initialization
    def __init__(self, numberOfDays, numberOfStocks):
        self.fullMarket = ed.generateStockList('data/minute.csv', self.np_random.integers(0,self.numberOfDays-1,1,int), 17)
        self.startingMoney = self.np_random.integers(30000, 40000, 1, np.float32)
        self.currentMoney = self.startingMoney
        self.currentTime = 0
        self.numShares = np.zeros([self.numberOfStocks,1], np.float32)
        self.buyPrice = np.zeros([self.numberOfStocks,1], np.float32)
        self.numberOfDays = numberOfDays
        self.numberOfStocks = numberOfStocks
        # Observation Space
        self.observation_space = spaces.Dict(
            {
                "botData": spaces.Box(np.array([25000, 0]), np.array([3.4028235e+38, 390]), (2,), np.float32), # current money, current time
                "investmentData": spaces.Box(np.zeros([numberOfStocks,2], np.float32), np.full_like(np.zeros([numberOfStocks,2]), [3.4028235e+38, 3.4028235e+38]), (numberOfStocks,2), np.float32), # numshares, buy price
                "stocks": spaces.Box(0, 3.4028235e+38, (numberOfStocks,5), np.float32), # Open High Low Close Volume
            }
        )

        # Action Space
        """
        self.action_space = spaces.Dict(
            {
                 "buy": spaces.Box([0, 1],[numberOfStocks, 3.4028235e+38],(2,), np.float32), # what stock, how many shares
                 "sell": spaces.Box([0, -3.4028235e+38],[1001, -1],(2,), np.float32), #what stock, how many shares
                 "wait": np.ndarray([0,0])
            }
        )
        """
        self.action_space = spaces.Dict(
            {
                # Stock Name is key, action is box with how many shares to do something with (negative is sell, 0 is hold, positive is buy)
                stock.ticker: spaces.Box(-self.numShares[self.fullMarket.index(stock)],(self.currentMoney-25000)/np.float32(stock.open[self.currentTime])*np.float32(numberOfStocks),dtype=np.float32) # minimum of how many shares had, max of above minimum divided by stock price further divided by number of stocks 
                for stock in self.fullMarket
            }
        )

    def _get_obs(self):
        # set up stock history
        open = [[stock.open[i] for i in range(self.currentTime+1)] for stock in self.fullMarket]
        close = [[stock.close[i] for i in range(self.currentTime+1)] for stock in self.fullMarket]
        low = [[stock.low[i] for i in range(self.currentTime+1)] for stock in self.fullMarket]
        high = [[stock.high[i] for i in range(self.currentTime+1)] for stock in self.fullMarket]
        volume = [[stock.volume[i] for i in range(self.currentTime+1)] for stock in self.fullMarket]
        return { "botData" : np.array([self.currentMoney, self.currentTime]) , "investmentData" :np.array([self.numShares,self.buyPrice]), "stocks" : np.array([open, close, low, high, volume]) }
    
 
    def _get_info(self):
        stockWorth = sum(self.numShares*self.fullMarket[self.currentTime])
        return {
            "timeLeft": (390 - self.currentTime),
            "profit": (self.currentMoney + stockWorth) - self.startingMoney
        }

    # Reset 
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        # Variables
        self.fullMarket = ed.generateStockList('data/minute.csv', self.np_random.integers(0,self.numberOfDays-1,1), 17)
        self.startingMoney = self.np_random.integers(30000, 40000, 1, np.float32)
        self.currentMoney = self.startingMoney
        self.currentTime = 0
        self.numShares = np.zeros([self.numberOfStocks,1], np.float32)
        self.buyPrice = np.zeros([self.numberOfStocks,1], np.float32)

        # Gather observations
        observation = self._get_obs()
        info = self._get_info()

        return observation, info
        
    # Step 
    def step(self, action_dict):
        # Actions
        for stock, action in action_dict.items():
            """"""

        terminated = False
        truncated = False
        reward += 1 if sum(self.fullMarket[self.currentTime]) - sum(self.buyPrice) > 0 else -1
        if self.currentTime == 390:
            terminated = True
        if self.currentMoney < 25000:
            truncated = True
        
        self.currentTime += 1
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info