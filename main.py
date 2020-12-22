from ipdb import set_trace
from agent import *
import pommerman
import colorama
from pommerman import agents
from collections import Counter
import os
import sys
import time
import random
import os
import math
import os
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--gpu')
parser.add_argument('--use')
parser.add_argument('--resume',action='store_true')
args=parser.parse_args()
if args.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

class World():
    def __init__(self, init_gmodel = True):
        if init_gmodel: 
            self.gmodel = A2CNet(gpu = (args.gpu is not None)) # Global model
            if args.gpu is not None:
                self.gmodel=self.gmodel.cuda()

        self.model = A2CNet(gpu = False)     # Agent (local) model
        if args.resume:
            self.model.load_state_dict(torch.load("model/A2C.weights"))
            self.gmodel.load_state_dict(torch.load("model/A2C.weights"))
        self.rlagent = RLAgent(self.model)
        self.ragent = Randomagent()

        self.agent_list = [
            self.rlagent, 
            agents.SimpleAgent(), 
            agents.SimpleAgent(), 
            agents.SimpleAgent()
        ]
        self.env = normal_env(self.agent_list) #naked_env
        fmt = {
            'int':   self.color_sign,
            'float': self.color_sign
        }
        np.set_printoptions(formatter=fmt,linewidth=300)
        pass

    def color_sign(self, x):
        if x == 0:    c = colorama.Fore.LIGHTBLACK_EX
        elif x == 1:  c = colorama.Fore.BLACK
        elif x == 2:  c = colorama.Fore.BLUE
        elif x == 3:  c = colorama.Fore.RED
        elif x == 4:  c = colorama.Fore.RED
        elif x == 10: c = colorama.Fore.YELLOW
        else:         c = colorama.Fore.WHITE
        x = '{0: <2}'.format(x)
        return f'{c}{x}{colorama.Fore.RESET}'

def do_rollout(env, rlagent, do_print = False):
    done, state = False, env.reset()
    rewards, dones   = [], []
    states,rawstates, actions,  probs, values = rlagent.clear()
    length=0

    while (not done) or (length>500):
        if do_print:
            time.sleep(0.1)
            os.system('clear')
            print(state[0]['board'])
            
        action = env.act(state)
        state, reward, done, info = env.step(action)
        length+=1
        if reward[0] == -1: done = True
        rewards.append(reward[0])
        dones.append(done)
    
    return (states.copy(), 
            rawstates.copy(),
            actions.copy(), 
            rewards, dones, 
            probs.copy(), 
            values.copy())

def gmodel_train(gmodel, states,rawstates,  actions, rewards, gae):
    probs=[]
    values=[]
    for i in range(len(states)):
        state,rawstate = torch.stack(states[i],1).squeeze(2).to(gmodel.device),torch.stack(rawstates[i],1).squeeze(2).to(gmodel.device) 
        hns,cns=gmodel.get_lstm_reset()
        gmodel.train()
        ps, vs, _, _ = gmodel(state,rawstate, hns, cns)
        probs.append(ps.squeeze(0))
        values.append(vs.squeeze(0))
    
    probs=torch.cat(probs,axis=0)
    values=torch.cat(values,axis=0)
    prob      = F.softmax(probs, dim=-1)
    log_prob  = F.log_softmax(probs, dim=-1)
    entropy   = -(log_prob * prob).sum(1)

    log_probs = log_prob[range(0,len(actions)), actions]
    advantages = torch.tensor(rewards).to(gmodel.device) - values.squeeze(1)
    value_loss  = advantages.pow(2)*0.5
    policy_loss = -log_probs*torch.tensor(gae).to(gmodel.device) - gmodel.entropy_coef*entropy
    
    gmodel.optimizer.zero_grad()
    pl = policy_loss.sum()
    vl = value_loss.sum()
    loss = pl+vl
    loss.backward()
    gmodel.optimizer.step()
    
    return loss.item(), pl.item(), vl.item(),entropy.mean().item()

def unroll_rollouts(gmodel, list_of_full_rollouts):
    gamma = gmodel.gamma
    tau   = 1

    states,rawstates, actions, rewards,gae =[], [], [], [], []
    for (s,raw, a, r, d, p, v) in list_of_full_rollouts:
        states.append(s)
        rawstates.append(raw)
        actions.extend(a)
        rewards.extend(gmodel.discount_rewards(r))

        # Calculate GAE
        last_i, _gae, __gae = len(r) - 1, [], 0
        for i in reversed(range(len(r))):
            next_val = v[i+1] if i != last_i else 0
            delta_t = r[i] + gamma*next_val - v[i]
            __gae = __gae * gamma * tau + delta_t
            _gae.insert(0, __gae)

        gae.extend(_gae)
    
    return states,rawstates, actions, rewards, gae

def train(world):
    model, gmodel = world.model, world.gmodel
    rlagent, env     = world.rlagent, world.env
    if os.path.exists("training.txt"): os.remove("training.txt")

    rr = -1
    ROLLOUTS_PER_BATCH = 10
    total_rollout=0
    for i in range(40000):
        total_rollout+=ROLLOUTS_PER_BATCH
        full_rollouts = [do_rollout(env, rlagent) for _ in range(ROLLOUTS_PER_BATCH)]
        last_rewards = [roll[3][-1] for roll in full_rollouts]
        not_discounted_rewards = [roll[3] for roll in full_rollouts]
        states,rawstates,actions, rewards, gae = unroll_rollouts(gmodel, full_rollouts)
        #gmodel.gamma = 0.9+0.1/(1+math.exp(-0.0003*(i-20000))) # adaptive gamma
        l, pl, vl,et = gmodel_train(gmodel, states,rawstates, actions, rewards, gae)
        rr = rr * 0.99 + np.mean(last_rewards)/ROLLOUTS_PER_BATCH * 0.01
        epilength=len(actions)
        text= '%d\tepilen%d\tgamma:%.3f\tavgrw:%.3f\twin:%d\tdrw:%.3f\tploss:%.3f\tvloss:%.3f\tentropy:%.3f'%(total_rollout,epilength,gmodel.gamma,rr,last_rewards.count(1),np.mean(rewards),pl,vl,et)
        print(text)
        print(Counter(actions))
        ROLLOUTS_PER_BATCH = min(max(1,100//epilength),3)
        with open("training.txt", "a") as f: 
            print(text,file=f)
        model.load_state_dict(gmodel.state_dict())
        if i >= 10 and (i % 10) == 0: torch.save(gmodel.state_dict(), "model/A2C.weights") 


def eval(world, init_gmodel = False):
    env = world.env
    model = world.model
    rlagent = world.rlagent
    rlagent.stochastic = True

    do_print = True
    done = None
    reward = 0
    last_reward = [0,0,0,0]
    
    while True:
        model.load_state_dict(torch.load("model/A2C.weights"))

        done, state, _ = False, env.reset(), rlagent.clear()
        t = 0
        while not done:
            if do_print:
                time.sleep(0.1)
                os.system('clear')
                print(state[0]['board'])
                print("\n\n")
                print("Probs: \t", rlagent.probs[-1] if len(rlagent.probs) > 0 else [])
                print("Val: \t", rlagent.values[-1] if len(rlagent.values) > 0 else None)
                print("\nLast reward: ", last_reward, "Time", t)

            env.render()
            action = env.act(state)
            state, reward, done, info = env.step(action)
            if reward[0] == -1: 
                last_reward = reward
                break
            t+=1

locals()[args.use](World())
