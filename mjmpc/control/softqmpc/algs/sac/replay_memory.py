import random
import numpy as np

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class ReplayMemoryTraj(ReplayMemory):
    def __init__(self, capacity):
        super(ReplayMemoryTraj, self).__init__(capacity)
    
    def push(self, trajectories):
        for k in trajectories.keys():
            if k is not 'infos':
                trajectories[k] = np.concatenate(trajectories[k], axis=0)
        num_elements = trajectories["observations"].shape[0]
        for i in range(num_elements):
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            obs = trajectories["observations"][i]
            action = trajectories["actions"][i]
            reward = -1.0 * trajectories["costs"][i]
            next_obs = trajectories["next_observations"][i]
            done = trajectories["dones"][i]
            self.buffer[self.position] = (obs, action, reward, next_obs, done)
            self.position = (self.position + 1) % self.capacity      
