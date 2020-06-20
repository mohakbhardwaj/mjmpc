from mjmpc.control.controller import Controller

from mjmpc.utils import helpers

class SoftQMPC(Controller):
    def __init__(self, 
                 d_state,
                 d_action,                
                 action_lows,
                 action_highs,
                 horizon,
                 base_action,
                 gamma,
                 n_iters,
                 set_sim_state_fn=None,
                 get_sim_state_fn=None,
                 sim_step_fn=None,
                 sim_reset_fn=None,
                 rollout_fn=None,
                 sample_mode='mean',
                 batch_size=1,
                 seed=0):

        super(SoftQMPC, self).__init__(d_state,
                                    d_action,
                                    action_lows, 
                                    action_highs,
                                    horizon,
                                    gamma,  
                                    n_iters,
                                    set_sim_state_fn,
                                    get_sim_state_fn,
                                    sim_step_fn,
                                    sim_reset_fn,
                                    rollout_fn,
                                    sample_mode,
                                    batch_size,
                                    seed)