# import
import extract_data as ed
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class marketEnv(gym.Env):
    def __init__(self, stockList):
        """sedfsef"""

        # Observation Space
        self.observation_space = spaces.Dict(
            {
                "botData": spaces.Box([25000 0], [3.4028235e+38 390], (2,), float32), # current money, current time
                "investmentData": spaces.Box([0 0], [3.4028235e+38 3.4028235e+38], (2,4), float32), # shares, buy price
                "currentStock": spaces.Box(0, 3.4028235e+38, (5,4), float32), # Open High Low Close Volume
            }
        )
        # Action Space
        self.action_space = spaces.Dict(4)
        
        self.bot_commands = spaces.Dict(
            {
                "buy": spaces.Box([],[],(), float32), # what stock, how many shares 
            }
        )
    
