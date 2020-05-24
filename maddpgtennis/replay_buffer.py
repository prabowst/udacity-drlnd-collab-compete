import numpy as np
import random
import torch
from collections import deque, namedtuple

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ReplayBuffer:

    '''Replay buffer to store experience tuples'''

    def __init__(self, n_action, buffer_size, batch_size, seed=0):

        '''Initialize ReplayBuffer class

        Params:
            n_action        : size of the action
            buffer_size     : size of buffer
            batch_size      : size of training and sample batch
            seed            : random seed number
        '''

        self.n_action = n_action
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        labels = ['state', 'action', 'reward', 'state_', 'done']
        self.experience = namedtuple('Experience', field_names=labels)
        self.seed = random.seed(seed)

    def add(self, state, action, reward, state_, done):

        '''Append new experience to memory

        Params:
            state       : current state of the environment
            action      : action taken by the agents
            reward      : reward given to the agent based on the action
            state_      : new state of the environment after action resolved
            done        : status of the environment
        '''

        exp = self.experience(state, action, reward, state_, done)
        self.memory.append(exp)

    def sample_replay(self):

        '''Take random sample of experience from the batches available within
           the replay buffer

        Return:
            tuple of states, actions, rewards, next states and dones
        '''

        experiences = random.sample(self.memory, self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        states_ = torch.from_numpy(np.vstack([e.state_ for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, states_, dones)

    def __len__(self):

        '''Return current size of memory'''

        return len(self.memory)
