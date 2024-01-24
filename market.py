# import
import extract_data as ed
import gymnasium as gym
from gymnasium import spaces
from gymnasium import utils
import numpy as np



class marketEnv(gym.Env):
    # Initialization
    def __init__(self, numberOfDays, numberOfStocks):
        self.numberOfDays = numberOfDays
        self.numberOfStocks = numberOfStocks
        # Observation Space
        self.observation_space = spaces.Dict(
            {
                "botData": spaces.Box(np.array([25000, 0]), np.array([3.4028235e+38, 390]), (2,), float), # current money, current time
                "investmentData": spaces.Box(np.zeros([numberOfStocks,2], float), np.full_like(np.zeros([numberOfStocks,2]), [3.4028235e+38, 3.4028235e+38]), (numberOfStocks,2), float), # shares, buy price
                "stocks": spaces.Box(0, 3.4028235e+38, (numberOfStocks,5), float), # Open High Low Close Volume
            }
        )

        # Action Space
        self.action_space = spaces.Dict(
            {
                 "buy": spaces.Box([0, 1],[1001, 3.4028235e+38],(2,), float), # what stock, how many shares
                 "sell": spaces.Box([0, 0],[1001, 3.4028235e+38],(2,), float), #what stock, how many shares
                 "wait": np.ndarray([0,0])
            }

        )

    # Get observations from environment

    def _get_obs(self):
        """
        test
        """
    # Reset 
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Variables
        self.fullMarket = ed.generateStockList('data/minute.csv', self.np_random.integers(0,1,1,int), 17)
        self.currentMoney = self.np_random.integers(30000, 40000, 1, float)
        self.currentTime = 0
        self.investmentData = np.zeros([self.numberOfStocks,2], float)
        
        



    def step(self, action):
        stock_price = "investmentData" 
        
        