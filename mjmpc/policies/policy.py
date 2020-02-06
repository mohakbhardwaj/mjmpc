"""Base class for policies"""
from abc import ABC, abstractmethod
import numpy as np


class Policy(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        pass

    @abstractmethod
    def get_action(self, obs: np.ndarray = None) -> np.ndarray:
        """Returns action given observation"""
        pass

    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def get_action_seq(self, obs: np.ndarray = None, horizon: int = 1) -> np.ndarray:
        pass