"""Select random action"""
#!/usr/bin/env python
import  numpy as np
from .policy import Policy

class RandomPolicy(Policy):
    def __init__(self, action_lows: np.ndarray, action_highs: np.ndarray, batch_size: int):
        super(RandomPolicy, self).__init__(batch_size)
        self._action_lows = action_lows
        self._action_highs = action_highs
        self._num_actions = action_highs.shape[0]        

    def get_action(self, obs: np.ndarray = None) -> np.ndarray:
        """
            Returns a single randomly sampled action
        """
        action = np.random.uniform(self._action_lows, self._action_highs, size=(self.batch_size, self._num_actions))
        return action.reshape(self._num_actions,)

    def get_action_seq(self, obs: np.ndarray = None, horizon: int = 1) -> np.ndarray:
        """
            Return a sequence of randomly sampled actions
        """
        return np.random.uniform(self._action_lows, self._action_highs, size=(self.batch_size, horizon, self._num_actions))

    def reset(self):
        pass
