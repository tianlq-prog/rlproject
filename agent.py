from ipdb import set_trace
import pommerman
from pommerman import agents
import sys
import gym
import time
import random
import numpy as np
from collections import namedtuple
from collections import Counter
import multiprocessing as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

S_statespace = 3
S_actionspace = 6
class A2CNet(nn.Module):
    def __init__(self,gpu=True):
        super(A2CNet, self).__init__()
        self.gamma             = 0.99   
        self.entropy_coef      = 0.01  
        self.lr                = 0.001

        self.conv1 = nn.Conv2d(S_statespace, 64, 3, stride=1, padding=1)
        #self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        #self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        #self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        #self.maxp4 = nn.MaxPool2d(2, 2)

        self.encoder1 = nn.Linear(14483,1000)
        self.encoder2 = nn.Linear(1000,200)
        self.encoder3 = nn.Linear(200,50)
        
        self.critic_linear = nn.Linear(83, 1)
        self.actor_lstm = nn.LSTM(50, S_actionspace,2,batch_first=True)
        self.actor_out = nn.Linear(S_actionspace, S_actionspace)

        torch.nn.init.xavier_uniform_(self.encoder1.weight)
        torch.nn.init.xavier_uniform_(self.encoder2.weight)
        torch.nn.init.xavier_uniform_(self.encoder3.weight)
        torch.nn.init.xavier_uniform_(self.critic_linear.weight)

        self.optimizer   = optim.Adam(self.parameters(), lr=self.lr)
        self.device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")

    def forward(self, x,raw, hx,cx):
        batch_size,timesteps, C, H, W = x.size()
        x = x.view(batch_size * timesteps, C, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(batch_size,timesteps, -1)
        x = torch.cat((x,raw),-1)
        x = F.relu(self.encoder1(x))
        x = F.relu(self.encoder2(x))
        x = F.relu(self.encoder3(x))#.permute(1, 0, 2)
        #critic
        value = self.critic_linear(raw)
        #actor
        out,(hx,cx) = self.actor_lstm(x,(hx,cx))
        action = self.actor_out(out)
        return action,value, hx, cx

    def get_lstm_reset(self,batch=1):
        hx = torch.zeros(2, batch, 6) 
        cx = torch.zeros(2, batch, 6)
        return hx.to(self.device),cx.to(self.device)

    def discount_rewards(self, _rewards):
        R = 0
        gamma = self.gamma
        rewards = []
        for r in _rewards[::-1]:
            R = r + gamma * R
            rewards.insert(0, R)
        
        return rewards #torch.from_numpy(rewards).to(self.device)

class RLAgent(agents.BaseAgent):

    def __init__(self, model,train=True):
        super(RLAgent, self).__init__()
        self.model     = model
        self.states    = []
        self.rawstates    = []
        self.actions   = []
        self.action_history   = np.zeros(6)
        self.values    = []
        self.probs     = []
        self.stochastic = train
        self.hn, self.cn = self.model.get_lstm_reset()

    def act(self,state,action_space):
        obs,raw = self.observe(state,self.action_history)
        obs = torch.from_numpy(obs).float().to(self.model.device).unsqueeze(0).unsqueeze(0)
        raw = torch.from_numpy(raw).float().to(self.model.device).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            probs,val,self.hn,self.cn = self.model(obs,raw,self.hn,self.cn)
            probs_softmaxed = F.softmax(probs, dim=-1)

            if self.stochastic: 
                action = Categorical(probs_softmaxed).sample().item()
            else: 
                action = probs_softmaxed.max(2, keepdim=True)[1].item()

        self.actions.append(action)
        self.states.append(obs)
        self.rawstates.append(raw)
        self.probs.append(probs)
        self.values.append(val)
        self.action_history[:-1] = self.action_history[1:]
        self.action_history[-1] = action
        return action

    def clear(self):
        del self.states[:]
        del self.rawstates[:]
        del self.actions[:]
        del self.probs[:]
        del self.values[:]
        self.hn, self.cn = self.model.get_lstm_reset()
        self.action_history   = np.zeros(6)

        return self.states,self.rawstates, self.actions, self.probs, self.values

    def observe(self,state,action_history):
        obs_width = 5 #choose uneven number
        obs_radius = obs_width//2
        board = state['board']
        blast_strength = state['bomb_blast_strength']
        bomb_life = state['bomb_life']
        pos = np.asarray(state['position'])
        board_pad = np.pad(board,(obs_radius,obs_radius),'constant',constant_values=1)
        BS_pad = np.pad(blast_strength,(obs_radius,obs_radius),'constant',constant_values=0)
        life_pad = np.pad(bomb_life,(obs_radius,obs_radius),'constant',constant_values=0)
        #centered, padded board
        board_cent = board_pad[pos[0]:pos[0]+2*obs_radius+1,pos[1]:pos[1]+2*obs_radius+1]
        bomb_BS_cent = BS_pad[pos[0]:pos[0]+2*obs_radius+1,pos[1]:pos[1]+2*obs_radius+1]
        bomb_life_cent = life_pad[pos[0]:pos[0]+2*obs_radius+1,pos[1]:pos[1]+2*obs_radius+1]
        ammo = np.asarray([state['ammo']])
        my_BS = np.asarray([state['blast_strength']])

        #note: on the board, 0: nothing, 1: unbreakable wall, 2: wall, 3: bomb, 4: flames, 6,7,8: pick-ups:  11,12 and 13: enemies
        out = np.empty((3,11+2*obs_radius,11+2*obs_radius),dtype=np.float32)
        out[0,:,:] = board_pad
        out[1,:,:] = BS_pad
        out[2,:,:] = life_pad
        #get raw surroundings
        raw = np.concatenate((board_cent.flatten(),bomb_BS_cent.flatten()),0)
        raw = np.concatenate((raw,bomb_life_cent.flatten()),0)
        raw = np.concatenate((raw,ammo),0)
        raw = np.concatenate((raw,my_BS),0)
        raw = np.concatenate((raw,action_history),0)

        return out,raw

class Randomagent(agents.BaseAgent):
        def __init__(self): super(Randomagent, self).__init__()
        def act(self, obs, action_space): 
            return random.randint(1,4) 
    

def normal_env(agent_list):
    env = gym.make('PommeFFACompetition-v0')
    
    for id, agent in enumerate(agent_list):
        assert isinstance(agent, agents.BaseAgent)
        agent.init_agent(id, env.spec._kwargs['game_type'])

    env.set_agents(agent_list)
    env.set_init_game_state(None)
    env.set_render_mode('human')
    return env
