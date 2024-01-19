# import
import extract_data as ed
import gym
from gym import spaces
import numpy as np


class marketEnv(gym.Env):

    def __init__(self, stockList):
        """sedfsef"""

        # Observation Space
        self.observation_space = spaces.Dict(
            {
                "stockList": spaces.Dict()
            }
        )
        
    
