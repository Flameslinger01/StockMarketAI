# import
import extract_data as ed
import csv
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
        self.currentTime, self.stockWorth, self.reward = 0, 0, 0
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
        self.low, self.high = [],[]
        for stock in self.fullMarket:
            if self.fullMarket.index(stock) in self.stockIndexes:
                self.low.append(-self.numShares[self.fullMarket.index(stock)].item())
                self.high.append(max(0,self.currentMoney-25000)/(np.float32(stock.open[self.currentTime])*np.float32(self.numberOfStocks)))
        self.action_space = spaces.Box(-25, 25,(self.numberOfStocks,),np.float32)

    def action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (-1, 1) to (low, high)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = action * scale_factor + reloc_factor
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (low, high) to (-1, 1)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = (action - reloc_factor) / scale_factor
        action = np.clip(action, -1.0, 1.0)

        return action
    def _get_obs(self):
        # set up stock history
        open = [[np.float32(stock.open[self.currentTime])] for stock in self.fullMarket]
        close = [[np.float32(stock.close[self.currentTime])] for stock in self.fullMarket]
        low = [[np.float32(stock.low[self.currentTime])] for stock in self.fullMarket]
        high = [[np.float32(stock.high[self.currentTime])] for stock in self.fullMarket]
        volume = [[np.float32(stock.volume[self.currentTime])] for stock in self.fullMarket]
        test2 = [[[self.currentMoney]], [[self.currentTime]], self.numShares.tolist(), self.buyPrice.tolist(),open, close, low, high, volume]
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
        self.stockIndexes.sort()

        # Action Space
        self.low, self.high = [],[]
        for stock in self.fullMarket:
            if self.fullMarket.index(stock) in self.stockIndexes:
                self.low.append(-self.numShares[self.fullMarket.index(stock)].item())
                self.high.append(max(0,self.currentMoney-25000)/(np.float32(stock.open[self.currentTime])*np.float32(self.numberOfStocks)))

        # Gather observations
        observation = self._get_obs()
        info = self._get_info()

        return np.array(observation, dtype=np.float32)
        
    # Step 
    def step(self, action_dict):
        # Actions
        action_dict = self.action(action_dict)
        #print(self.numShares)
        j = 0
        for i in range(len(self.fullMarket)):
            if i in self.stockIndexes:
                old = self.buyPrice[i] * self.numShares[i]
                self.numShares[i] += np.clip(np.float32(action_dict[j]),self.low[j],self.high[j])
                self.currentMoney -= np.float32(self.fullMarket[i].close[int(self.currentTime)].item()) * np.clip(np.float32(action_dict[j]),self.low[j],self.high[j])
                if self.numShares[i] > 0: self.buyPrice[i] = np.float32((old + np.float32(self.fullMarket[i].close[int(self.currentTime)].item()) * np.clip(np.float32(action_dict[j]),self.low[j],self.high[j]))/self.numShares[i])
                #print(self.buyPrice[i])
                j += 1
        #print()
    
        # Action Space
        self.low, self.high = [],[]
        for stock in self.fullMarket:
            if self.fullMarket.index(stock) in self.stockIndexes:
                self.low.append(-self.numShares[self.fullMarket.index(stock)].item())
                self.high.append(max(0,self.currentMoney-25000)/(np.float32(stock.open[self.currentTime])*np.float32(self.numberOfStocks)))
        #self.reward = 0
        terminated = False
        truncated = False
    
        if(sum(self.numShares) == 0):
            self.reward += 1
        else:
            self.reward -= 1
        if self.currentTime == 360:
            #print(self.currentMoney)
            terminated = True
        
        self.currentTime += 1
        observation = self._get_obs()
        info = self._get_info()
        return observation, self.reward, terminated, truncated, {}

"""
market = marketEnv()
#print(market._get_obs().shape)
temp = 0
growth = 1
k = 0
file1 = open("MyFile.csv", "w") 
while(True):
    k += 1
    growth += 0.05
    for i in range(100):
        test = market.action_space.sample()
        #print(market.currentMoney)
        #print(market.numShares)
        #print(market.action_space)
        #print(test)
        market.step(test)
    if((market.currentMoney+market.stockWorth - market.startingMoney)*growth > temp):
        file1.write(str((market.currentMoney+market.stockWorth - market.startingMoney).item() * growth) + ", " + str(k) + "\n")
        print(str((market.currentMoney+market.stockWorth - market.startingMoney).item() * growth) + ", " + str(k))
        temp = (market.currentMoney+market.stockWorth - market.startingMoney) * growth/2
        real = temp*2
    if(real > 17000):
        break
    market.reset()

print("Starting Money: " + str(market.startingMoney) + ", Current Money: " + str(market.currentMoney+market.stockWorth))
print("Profit: " + str(market.currentMoney+market.stockWorth - market.startingMoney))
file1.close()
"""