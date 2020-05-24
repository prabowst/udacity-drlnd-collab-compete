import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

from maddpgtennis import Actor, Critic, ReplayBuffer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DDPG:

    '''Define a single DDPG agent'''

    def __init__(self, n_state, n_action, n_agents, lr_act, lr_crit,
                 agent_no, seed=0):

        '''Initialize the DDPG agent

        Params:
            n_state     : number of states
            n_action    : size of action
            n_agents    : number of agents
            lr_act      : learning rate of the actor network
            lr_crit     : learning rate of the critic network
            agent_no    : number of the agent
            seed        : random seed number
        '''

        self.n_state = n_state
        self.n_action = n_action
        self.n_agents = n_agents
        self.lr_act = lr_act
        self.lr_crit = lr_crit
        self.agent_no = agent_no
        self.seed = random.seed(seed)

        self.act_local = Actor(self.n_state, self.n_action).to(device)
        self.act_target = Actor(self.n_state, self.n_action).to(device)
        self.crit_local = Critic(self.n_state, self.n_action).to(device)
        self.crit_target = Critic(self.n_state, self.n_action).to(device)

        self.act_opt = optim.Adam(self.act_local.parameters(), lr=self.lr_act)
        self.crit_opt = optim.Adam(self.crit_local.parameters(),
                                   lr=self.lr_crit, weight_decay=0)

        self.buffer_size = int(1e6)
        self.batch_size = 128
        self.memory = ReplayBuffer(self.n_action, self.buffer_size,
                                   self.batch_size)

    def action(self, state):

        '''Grab the action of individual agent

        Params:
            state       : current state of the environment

        Return:
            action      : clipped action between -1 and 1 for continuous space
        '''

        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        self.act_local.eval()
        with torch.no_grad():
            action = self.act_local(state).squeeze().cpu().data.numpy()
        self.act_local.train()
        return np.clip(action, -1, 1)
