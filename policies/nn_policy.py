"""Policy parameterized by a neural network"""
#!/usr/bin/env python
import numpy as np
from .policy import Policy

class NNPolicy(Policy):
    def __init__(self, model, batch_size):
        self.model = model
        super(NNPolicy, self).__init__(batch_size)

    def get_action(self, obs: np.ndarray = None) -> np.ndarray:
        """Returns randomly sampled action"""
        action = self.model(obs)
        return action

    def reset(self):
        raise NotImplementedError("To be implemented")