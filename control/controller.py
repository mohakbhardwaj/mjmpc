#!/usr/bin/env python
from abc import ABC, abstractmethod
import numpy as np


def scale_ctrl(ctrl, action_low_limit, action_up_limit):
    if len(ctrl.shape) == 1:
        ctrl = ctrl[np.newaxis, :, np.newaxis]
    act_half_range = (action_up_limit - action_low_limit) / 2.0
    act_mid_range = (action_up_limit + action_low_limit) / 2.0
    ctrl = np.clip(ctrl, -1.0, 1.0)
    return act_mid_range[np.newaxis, :, np.newaxis] + ctrl * act_half_range[np.newaxis, :, np.newaxis]

class Controller(ABC):
    def __init__(self,
                 num_actions,
                 action_lows,
                 action_highs,
                 get_state_fn,
                 set_state_fn,
                 terminal_cost_fn=None):

        self.num_actions = num_actions
        self.action_lows = action_lows
        self.action_highs = action_highs
        self.get_state_fn = get_state_fn
        self.set_state_fn = set_state_fn
        self.terminal_cost_fn = terminal_cost_fn

    @abstractmethod
    def step(self) -> np.ndarray:
        pass

    def set_terminal_cost_fn(self, fn):
        self.terminal_cost_fn = fn


    # def set_get_state_fn(self, fn):
    #     self.get_state_fn = fn
