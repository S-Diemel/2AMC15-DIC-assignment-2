"""Random Agent.

This is an agent that takes a random action from the available action space.
"""
from random import randint
import numpy as np
import time
from agents import BaseAgent


class RandomAgent(BaseAgent):
    """Agent that performs a random action every time. """
    def update(self, state: tuple[int, int], reward: float, action):
        time.sleep(1)
        pass

    def take_action(self, state: tuple[int, int]) -> int:
        return randint(0, 3)