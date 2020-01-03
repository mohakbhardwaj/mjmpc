#!/usr/bin/env python
import numpy as np
from .policy import  Policy
import control


class MPCPolicy(Policy):
    def __init__(self, controller_type, param_dict, batch_size=1):
        super(MPCPolicy, self).__init__(batch_size)
        if controller_type == "mppi":
            self.controller = control.MPPI(**param_dict)
        else:
            raise(NotImplementedError, "Controller type does not exist")

    def get_action(self, obs: np.ndarray = None)-> np.ndarray:
        action = self.controller.step()
        return action

    def reset(self):
        self.controller.reset()
    
    def get_action_seq(self, obs: np.ndarray = None, horizon: int = 1)-> np.ndarray:
        action = self.controller.step()
        return action