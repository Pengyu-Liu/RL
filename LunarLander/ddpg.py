import numpy as np
import matplotlib.pyplot as plt
import gym

import torch
import torch.nn as nn

import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#------------------------- Network ----------------------------
class Actor_Network(nn.Module):
    def __init__(self,n_states,n_action):
        super(Actor_Network,self).__init__()
        self.fc1 = nn.Linear(n_states,hidden)
        self.fc1.weight.data.normal_(0,0.1)
        self.out = nn.Linear(hidden,n_action)
        self.out.weight.data.normal_(0,0.1)
    def forward(self,x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.out(x)
        actions_value = torch.nn.functional.tanh(x)
        return actions_value

class Critic_Network(nn.Module):
    def __init__(self,n_states,n_action):
        super(Critic_Network,self).__init__()
        self.fcs = nn.Linear(n_states,hidden)
        self.fcs.weight.data.normal_(0,0.1)
        self.fca = nn.Linear(n_action,hidden)
        self.fca.weight.data.normal_(0,0.1)
        self.out = nn.Linear(hidden,1)
        self.out.weight.data.normal_(0,0.1)
    def forward(self,s,a):
        x1 = self.fcs(s)
        x2 = self.fca(a)
        x3 = torch.nn.functional.relu(x1+x2)
        actions_value = self.out(x3)
        return actions_value

#------------------------- DDPG class ----------------------------
class DDPG(object):
    def __init__(self, n_action, n_states):
        self.n_action, self.n_states = n_action, n_states
        # for storing memory, initialize
        self.memory = np.zeros((memory_size, n_states * 2 + n_action + 1), dtype=np.float32)
        # for counting when updating memory
        self.counter = 0
        # four networks
        self.Actor_eval = Actor_Network(n_states,n_action)
        self.Actor_target = Actor_Network(n_states,n_action)
        self.Critic_eval = Critic_Network(n_states,n_action)
        self.Critic_target = Critic_Network(n_states,n_action)
        
        # optimizers
        self.critic_train = torch.optim.Adam(self.Critic_eval.parameters(),lr=lr_critic)
        self.actor_train = torch.optim.Adam(self.Actor_eval.parameters(),lr=lr_actor)
        # loss function for critic
        self.loss_td = nn.MSELoss()

    def store_transition(self, s, a, r, s_):
        # replay memory
        transition = np.hstack((s, a, [r], s_))
        index = self.counter % memory_size
        self.memory[index, :] = transition
        self.counter += 1

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        return self.Actor_eval(x)[0].detach()

    def learn(self):
        # sample data from batch, extract s,a,r,s_ accordingly 
        indices = np.random.choice(memory_size, size=batch_size)
        bt = self.memory[indices, :]
        bs = torch.FloatTensor(bt[:, :self.n_states])
        ba = torch.FloatTensor(bt[:, self.n_states: self.n_states + self.n_action])
        br = torch.FloatTensor(bt[:, -self.n_states - 1: -self.n_states])
        bs_ = torch.FloatTensor(bt[:, -self.n_states:])

        a = self.Actor_eval(bs)
        q = self.Critic_eval(bs,a)
        loss_actor = -torch.mean(q)
        self.actor_train.zero_grad()
        loss_actor.backward()
        self.actor_train.step()

        # Compute the target q value
        a_ = self.Actor_target(bs_)  # This network does not update the parameters in time, is used to predict the action in Critic Q_target
        q_ = self.Critic_target(bs_,a_)  # This network does not update the parameters in time, is used to give the Gradient ascent strength when the Actor updates the parameters
        q_target = br + gamma*q_
        
        # computer current q value
        q_current = self.Critic_eval(bs,ba)

        # Compute critic loss
        critic_loss = self.loss_td(q_target,q_current)
        # optimize the critic loss
        self.critic_train.zero_grad()
        critic_loss.backward()
        self.critic_train.step()

        # softly update target critic and actor networks by using eval to reach all attributes
        for x1,x2 in zip(self.Critic_target.parameters(),self.Critic_eval.parameters()):
            x1.data.data.mul_((1-tau))
            x1.data.add_(tau*x2.data)

        for x1,x2 in zip(self.Actor_target.parameters(),self.Actor_eval.parameters()):
            x1.data.data.mul_((1-tau))
            x1.data.add_(tau*x2.data)

#---------------- adjustable hyperparameters ----------------------

number_episodes = 1000
number_steps_per_episode =300

lr_actor = 0.001
lr_critic = 0.002
gamma = 0.99
tau = 0.002

memory_size = 10000
batch_size = 32
hidden = 128

sigma = 3 
#--------------------------- main -------------------------------
env = gym.make('LunarLanderContinuous-v2')
env = env.unwrapped

n_states = env.observation_space.shape[0]
n_action = env.action_space.shape[0]

ddpg = DDPG(n_action, n_states)

cul_reward_list = []

t0 = time.time()
for ep in range(number_episodes):
    s = env.reset()
    cul_reward = 0
    sigma *= 0.995 # decaying noise
    for st in range(number_steps_per_episode):

        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, sigma), -1, 1) # noise
        s_, r, done, info = env.step(a)

        ddpg.store_transition(s, a, r, s_) # memory

        if ddpg.counter > memory_size:
            ddpg.learn()

        s = s_
        
        cul_reward += r
        if done or st==number_steps_per_episode-1:
            print('Episode:', ep, ' Reward:', int(cul_reward))
            cul_reward_list.append(cul_reward)
            break


plt.plot(cul_reward_list)
plt.title('Performance for all algorithms')
plt.xlabel('episode')
plt.ylabel('cumulative reward')
plt.legend()
plt.savefig('allresult.png',dpi=200)
print('Running time: ', time.time() - t0)
#np.savetxt('result01.txt',np.array(cul_reward_list))