import gym
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
Hyperparameters:
    lr                  # learning rate
    epsilon             # greedy policy
    gamma               # reward discount
    target_update_iter  # target update frequency
    replay_memory_size
    batch_size
    hidden              # nodes in the hidden layer
"""
lr = 0.01
epsilon = 0.9
gamma = 0.7
target_update_iter = 100
replay_memory_size = 1000
batch_size = 32
hidden = 128

# environment
env = gym.make('CartPole-v1')
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

# common used way to create a neutral network
class net(torch.nn.Module):
    def __init__(self, ):
        super(net, self).__init__()
        self.fc1 = torch.nn.Linear(n_states, hidden)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = torch.nn.Linear(hidden, n_actions)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, xx):
        xx = self.fc1(xx)
        xx = torch.nn.functional.relu(xx)
        actions_value = self.out(xx)
        return actions_value

# define a DQN
# (refer to tensorflow version and rewrite it to pytorch:
# Title: Balancing a CartPole System with Reinforcement Learning - A Tutorial
# Author: Swagat Kumar)
class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = net(), net()
        # for target updating
        self.learn_step_counter = 0
        # for storing memory
        self.memory_counter = 0
        # initialize memory
        self.memory = np.zeros((replay_memory_size, n_states * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = torch.nn.MSELoss()

    def choose(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # greedy
        if np.random.uniform() < epsilon:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]  # return the argmax
        else:  # random
            action = np.random.randint(0, n_actions)
            action = action
        return action

    def store(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % replay_memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % target_update_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample from batch
        sample_index = np.random.choice(replay_memory_size, batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :n_states])
        b_a = torch.LongTensor(b_memory[:, n_states:n_states + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, n_states + 1:n_states + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -n_states:])

        # choose q_eval according to b_a
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()  # detach but not back-propagate
        q_target = b_r + gamma * q_next.max(1)[0].view(batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        # update eval_net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn = DQN()

# dynamically showing the plot
# (refer to official tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
def plot_durations():
    plt.figure(2)
    plt.clf()
    plt.xlabel('Episode')
    plt.ylabel('Duration')

    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


plot_x_data, plot_y_data = [], []
print('\n...')
for i_episode in range(400):
    s = env.reset()
    score = 0
    total_score = 0
    total_r = 0
    for t in count():
        env.render()
        # choose action
        a = dqn.choose(s)
        # return
        s_, r, done, info = env.step(a)
        # accumulate the reward
        total_r += r
        # define score for better training the nets
        x, x_dot, theta, theta_dot = s_
        score1 = (env.x_threshold - abs(x+0.02*x_dot)) / env.x_threshold
        score2 = (env.theta_threshold_radians - abs(theta+0.02*theta_dot)) / env.theta_threshold_radians
        score = score1*score2
        # store transition
        dqn.store(s, a, score, s_)

        if dqn.memory_counter > replay_memory_size:
            dqn.learn()
            if done:
                print('Ep: ', i_episode,
                      '| Total reward: ', total_r)
        if done:
            plot_durations()
            break
        s = s_
    plot_x_data.append(i_episode)
    plot_y_data.append(total_r)
    plt.plot(plot_x_data, plot_y_data)
print('Complete')
env.close()
plt.ioff()
plt.show()