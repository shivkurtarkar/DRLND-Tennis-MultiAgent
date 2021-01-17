import numpy as np
import random
from collections import namedtuple, deque

import torch

from ddpg_agent import Agent,ReplayBuffer 

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 512        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters

LEARN_EVERY =1
LEARN_N_TIMES =1

class MADDPGAgent:
    def __init__(self, state_size, action_size, num_agents, random_seed):
        
        self.state_size = state_size
        self.action_size = action_size
        self.random_seed = random.seed(random_seed)
        
        self.agents = [ Agent(state_size, action_size, random_seed)]*num_agents
        self.shared_memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        

    def step(self, states, actions, rewards, next_states, dones, step):
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.shared_memory.add(state, action, reward, next_state, done)
               
        if len(self.shared_memory) > BATCH_SIZE and step % LEARN_EVERY == 0:
            for _ in range(LEARN_N_TIMES):
                for agent in self.agents:
                    experiences = self.shared_memory.sample()
                    agent.learn(experiences, GAMMA)
    
    def act(self, states, add_noise=True):
        actions = []
        for state, agent in zip(states, self.agents):
            state = np.expand_dims(state, axis=0)
            action = agent.act(state) 
            action = np.reshape(action, newshape=(-1))
            actions.append(action)            
        actions = np.stack(actions)
        return actions
    
    def save_weights(self):
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_'+str(i)+'.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_'+str(i)+'.pth')
    def load_weights(self):
        for i, agent in enumerate(self.agents):
            agent.actor_local.load_state_dict(torch.load('checkpoint_actor_'+str(i)+'.pth'))
            agent.critic_local.load_state_dict(torch.load('checkpoint_critic_'+str(i)+'.pth'))
            
    def reset(self):
        for agent in self.agents:
            agent.reset()
        