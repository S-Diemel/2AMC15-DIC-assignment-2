"""Random Agent.

This is an agent that takes a random action from the available action space.
"""
import random
import numpy as np
import time
from agents import BaseAgent


class GreedyAgent(BaseAgent):
    """Agent that performs a random action every time. """
    def update(self, state: tuple[int, int], reward: float, action):
        #time.sleep(1)
        pass

    def take_action(self, state: tuple[int, int]) -> int:
        x, y, speed, orientation, steps_fw, steps_fw_left, steps_fw_right, dx, dy = state
        if steps_fw>1:
            if speed == 0:
                action = 0
            else:
                action = 4


        else:
            if steps_fw_left > 1 and steps_fw_right > 1:
                action = random.choice([2,3])
            elif steps_fw_left > 1:
                action = 2
            elif steps_fw_right > 1:
                action = 3
            else:
                if speed==0:
                    action = random.choice([2,3])
                else:
                    action = 1
        #print(action)
        return action




