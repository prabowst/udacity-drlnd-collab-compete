import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):

    '''Actor Network model'''

    def __init__(self, n_state, n_action, seed=0):

        '''Initialize the Actor policy

        Params:
            n_state     : dimension of state space
            n_action    : dimension of action
            seed        : seed random number
        '''

        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        fc1_units = 256
        fc2_units = 128

        self.fc1 = nn.Linear(n_state, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, n_action)
        self.reset_params()

    def reset_params(self):

        '''Reset the network's parameters'''

        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):

        '''Network maps states to actions with batchnorm1d

        Params:
            state       : state of the environment

        Return:
            x           : output of the network
        '''

        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

class Critic(nn.Module):

    '''Critic Network model'''

    def __init__(self, n_state, n_action, seed=0):

        '''Initialize the Critic policy

        Params:
            n_state     : dimension of state space
            n_action    : dimension of action
            seed        : seed random number
        '''

        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        fc1_units = 256
        fc2_units = 128

        self.fc1 = nn.Linear(n_state, fc1_units)
        self.fc2 = nn.Linear(fc1_units+n_action, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_params()

    def reset_params(self):

        '''Reset the network's parameters'''

        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):

        '''Network maps states to actions with batchnorm1d

        Params:
            state       : state of the environment

        Return:
            x           : output of the network
        '''

        xs = F.relu(self.fc1(state))
        x_cat = torch.cat((xs, action), dim=-1)
        x = F.relu(self.fc2(x_cat))
        x = self.fc3(x)
        return x
