#!/usr/bin/env python
import numpy as np
from .policy import  Policy
import mjmpc.control as control


class MPCPolicy(Policy):
    def __init__(self, controller_type, param_dict, batch_size=1):
        super(MPCPolicy, self).__init__(batch_size)
        if controller_type == "cem":
            self.controller = control.CEM(**param_dict)
        elif controller_type == "dmd":
            self.controller = control.DMDMPC(**param_dict)
        elif controller_type == "ilqr":
            self.controller = control.ILQR(**param_dict)
        elif controller_type == "mppi":
            self.controller = control.MPPI(**param_dict)
        elif controller_type == "pfmpc":
            self.controller = control.PFMPC(**param_dict)
        elif controller_type == "random_shooting":
            self.controller = control.RandomShooting(**param_dict)
        elif controller_type == "random_shooting_nn":
            self.controller = control.RandomShootingNN(**param_dict)
        elif controller_type == "softq":
            self.controller = control.SoftQMPC(**param_dict)
        # elif controller_type == "sac_mpc":
            # self.controller = control.SACMPC(**param_dict)
        else:
            raise NotImplementedError("Controller type does not exist")

    def get_action(self, state, calc_val=False):
        action, value = self.controller.optimize(state, calc_val)
        return action, value

    def reset(self):
        self.controller.reset()
    