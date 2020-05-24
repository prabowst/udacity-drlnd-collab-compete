import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random

from collections import namedtuple, deque
from maddpgtennis import DDPG, ReplayBuffer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class MADDPG:

    '''Multi Agent DDPG'''

    def __init__(self, n_state, n_action, n_agents, seed=0):

        '''Initialize the MADDPG class

        Params:
            n_state     : size of observation (state-space)
            n_action    : size of action
            n_agents    : how many agents are trained
            seed        : seed random number
        '''


        self.n_state = n_state
        self.n_action = n_action
        self.n_agents = n_agents

        self.lr_act = 1e-4
        self.lr_crit = 3e-4
        self.agents_ = []
        for i in range(self.n_agents):
            agent = DDPG(self.n_state, self.n_action, self.n_agents,
                         self.lr_act, self.lr_crit, i)
            self.agents_.append(agent)

        self.buffer_size = int(1e6)
        self.batch_size = 128
        self.t_step = 0
        self.gamma = 0.99
        self.updateTargetNet = 1 # 20
        self.num_learn = 1 # 15
        self.noise_weight = 1
        self.noise_decay = 0.997
        self.noise_min = 0.1
        self.noise_update = 1

    def add_noise(self):

        '''Adding normal distribuion noise sample to be used later for agents'
           actions in learning

        Return:
            noise       : noise in tensor form
        '''

        random_dist = np.random.normal(0, 1, self.n_action)
        noise = self.noise_weight * random_dist
        return torch.from_numpy(noise).float().to(device)

    def action_(self, states):

        '''Action of the agents

        Params:
            states      : current state of the environment

        Return:
            actions     : calculate action for each agent on top of noise
        '''

        agents_observation = torch.from_numpy(states).float().to(device)

        actions = []
        for i in range(self.n_agents):
            agent = self.agents_[i]
            noise = self.add_noise().to(device)
            action = agent.action(agents_observation[i]) + noise
            actions.append(action)

        return actions

    def step(self, state, action, reward, state_, done):

        '''Gather memory for experience replay and check learning conditions
           as well as update the noise parameters

        Params:
            state       : current state of the environment
            action      : action taken by the agents
            reward      : reward given to the agent based on the action
            state_      : new state of the environment after action resolved
            done        : status of the environment
        '''

        for i in range(self.n_agents):
            self.agents_[i].memory.add(state[i], action[i], reward[i],
                                       state_[i], done[i])

        if self.t_step % self.noise_update == 0:
            self.noise_weight = max(self.noise_weight*self.noise_decay,
                                    self.noise_min)
        if len(self.agents_[0].memory) > self.batch_size:
            if self.t_step % self.updateTargetNet == 0:
                for _ in range(self.num_learn):
                    for i in range(self.n_agents):
                        agent_exp = self.agents_[i].memory.sample_replay()
                        self.learn(agent_exp, i)
        self.t_step += 1

    def learn(self, agent_exp, agent_no):

        '''Learning steps for the agents

        Params:
            agent_exp   : random sample of memory to be learned from
            agent_no    : unique id of the agent
        '''

        agent = self.agents_[agent_no]
        states, actions, rewards, states_, dones = agent_exp

        actions_ = agent.act_target(states_)
        Q_targets_ = agent.crit_target(states_, actions_)

        Q_targets = rewards + (self.gamma * Q_targets_ * (1 - dones))
        Q_expect = agent.crit_local(states, actions)

        crit_loss = F.mse_loss(Q_expect, Q_targets)
        agent.crit_opt.zero_grad()
        crit_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.crit_local.parameters(), 1)
        agent.crit_opt.step()

        # update actor
        actions_pred = agent.act_local(states)
        actor_loss = -agent.crit_local(states, actions_pred).mean()
        agent.act_opt.zero_grad()
        actor_loss.backward()
        agent.act_opt.step()

        # soft update
        self.soft_update(agent.crit_local, agent.crit_target)
        self.soft_update(agent.act_local, agent.act_target)

    def soft_update(self, local, target, TAU=0.01):

        '''Soft update in updating the networks parameter

        Params:
            local       : local network
            target      : target network
            TAU         : constant tau that governs the soft-update
        '''
        
        for local_param, target_param in zip(local.parameters(),
                                             target.parameters()):
            target_param.data.copy_(TAU * local_param.data + \
                                    (1.0-TAU) * target_param.data)
