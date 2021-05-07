import numpy as np
import matplotlib.pyplot as plt
import copy
import gym


# discretize continuous observation
def discretize_state(obs):
	discretized = []
	for i in range(len(obs)):
		lower_bound = max(env.observation_space.low[i],-1e3)
		higher_bound = min(env.observation_space.high[i],1e3)
		scaling = ((obs[i] - lower_bound) / (higher_bound - lower_bound))
		discretized.append(min(int(scaling*n_bins[i]),n_bins[i]-1))
	return tuple(discretized)

# moving average of N points
def aver(x,N):
    n = np.ones(N,dtype=float)
    weights = n/N
    aver_x = np.convolve(weights, x, mode='valid')
    return aver_x

# zero initialization
def Qtable_init_0():
    return np.zeros(n_bins+(env.action_space.n,))

# random initialization
def Qtable_init_rand():
    return np.random.uniform(low=-0.05,high=0.05,size=n_bins+(env.action_space.n,))

# Q tabular learning algorithm
def Tabular_Q(lr,discount_factor,init_name,episode):
    if init_name=='zero':
        # initialize Q-table
        Q_table = Qtable_init_0()
    else:
        Q_table = Qtable_init_rand()

    reward_list = []
    epsilon_list = []
    for i_episode in range(episode):
        observation = env.reset()
        discrete_state,done = discretize_state(observation),False
        reward_total = 0
        t = 0
        epsilon = 3*max(0,0.3-np.log10(i_episode*0.0015+1))
        epsilon_list.append(epsilon)
        while done==False:
            t += 1
            if np.random.random() > epsilon:
                action = np.argmax(Q_table[discrete_state])
            else:
                action = np.random.randint(0, env.action_space.n)

            observation, reward, done, info = env.step(action)

            reward_total += reward
            new_discrete_state = discretize_state(observation)
            
            reward_new = (1-abs(observation[0]/env.observation_space.high[0])) *  (1-abs(observation[2]/env.observation_space.high[2])) 
            Q_table[discrete_state][action] = (1-lr)*Q_table[discrete_state][action] + lr*(reward_new + discount_factor*np.max(Q_table[new_discrete_state]) )

            discrete_state = copy.deepcopy(new_discrete_state)

            if done:
                reward_list.append(reward_total)
                #print("Episode {} finished with {} rewards".format(i_episode,reward_total))
                break

            #env.render()

    return reward_list

#---------------------------------- main script start ----------------------
env = gym.make('CartPole-v1')

## our final chosen experiment parameter set
print('\nour finally chosen parameter set chosen experiment')
plt.figure(figsize=(10,4))
n_bins = (4,2,4,2)
result = Tabular_Q(lr=0.1,discount_factor=0.9,init_name='zeros',episode=10000)
plt.plot(result,label='ordinary',alpha=0.5)
plt.plot(aver(result,20),label='moving average 20',alpha=0.5)
print('average of last 1000 episodes cumulative rewards',np.mean(result[-1000:-1]))

plt.title('our finally chosen parameter set chosen experiment')
plt.xlabel('episode')
plt.ylabel('cumulative reward')
plt.legend()
plt.savefig('fig_final_exper.png',dpi=200)

## make action with random choice, no algorithm supported, can be called "baseline accuracy" 
print('\n0th experiment: make action with random choice, no algorithm supported, can be called "baseline accuracy"')
rewards_list = [] # to record all episode results
for i_episode in range(2000):
        observation = env.reset()
        done = False
        reward_total = 0
        while done==False:
            # take a random action
            observation, reward, done, info = env.step(env.action_space.sample())
            reward_total += reward

        rewards_list.append(reward_total)
plt.figure(figsize=(6,4))
plt.plot(rewards_list,alpha=0.5)
plt.xlabel('episode')
plt.ylabel('cumulative reward')
plt.title('baseline accuracy experiment with random action')
plt.savefig('fig_baseline.png',dpi=200)
print('average cumulative reward of baseline experiment',np.mean(rewards_list))


## 1st experiment, Q-table discretization choice
print('\n1st experiment, Q-table discretization choice')

ran_seed_list = 46456813,79841163,4311548
for i_ran,ran_seed_ in enumerate(ran_seed_list):
    plt.figure(figsize=(6,4))
    np.random.seed(ran_seed_)
    n_bins = (4,2,4,2)
    result = Tabular_Q(lr=0.1,discount_factor=0.9,init_name='zeros',episode=2000)
    plt.plot(aver(result,20),label='4,2,4,2',alpha=0.5)
    print('n_bins = (4,2,4,2), average rewards of the last 1000 episode',np.mean(result[1000:]))
    
    n_bins = (4,4,4,4)
    result = Tabular_Q(lr=0.1,discount_factor=0.9,init_name='zeros',episode=2000)
    plt.plot(aver(result,20),label='4,4,4,4',alpha=0.5)
    print('n_bins = (4,4,4,4), average rewards of the last 1000 episode',np.mean(result[1000:]),'\n')
    
    plt.title('Q-table discretization choice experiment')
    plt.xlabel('episode')
    plt.ylabel('cumulative reward')
    plt.legend()
    plt.savefig('fig_distable'+str(i_ran)+'.png',dpi=200)


## 2nd experiment, Q table initialization experiment
print('\n2nd experiment, Q table initialization experiment')

ran_seed_list = 116813,791163,4311548
for i_ran,ran_seed_ in enumerate(ran_seed_list):
    plt.figure(figsize=(6,4))
    np.random.seed(ran_seed_)
    n_bins = (4,2,4,2)
    result = Tabular_Q(lr=0.1,discount_factor=0.9,init_name='zeros',episode=2000)
    plt.plot(aver(result,20),label='zero',alpha=0.5)
    print('initialization = zeros, average rewards of the last 1000 episode',np.mean(result[1000:]))
    
    n_bins = (4,2,4,2)
    result = Tabular_Q(lr=0.1,discount_factor=0.9,init_name='random',episode=2000)
    plt.plot(aver(result,20),label='random',alpha=0.5)
    print('initialization = random, average rewards of the last 1000 episode',np.mean(result[1000:]),'\n')
    
    plt.title('Q table initialization experiment')
    plt.xlabel('episode')
    plt.ylabel('cumulative reward')
    plt.legend()
    plt.savefig('fig_init'+str(i_ran)+'.png',dpi=200)


## 3rd experiment, learning rate and discount factor space
print('\n3rd experiment, learning rate and discount factor space')
plt.figure(figsize=(6,4))
n_bins = (4,2,4,2)
lr_list = [0.1,0.2,0.3]
disfactor_list = [0.8,0.9]
rewards_list = []
for lr in lr_list:
    for discount_factor in disfactor_list:
        result = Tabular_Q(lr=lr,discount_factor=discount_factor,init_name='zeros',episode=2000)
        plt.plot(aver(result,20),label='lr=%.2f,df=%.2f'%(lr,discount_factor),alpha=0.5)
        print('lr=%.2f,df=%.2f'%(lr,discount_factor),np.mean(result[1000:]))

plt.title('experiment with learning rate and discount factor')
plt.ylabel('cumulative rewards')
plt.xlabel('episode')
plt.legend()
plt.savefig('fig_lrdf.png',dpi=200)

env.close()

## Probability of Trial: epsilon curve
epsilon_list = []
for x in range(2000):
    epsilon = 3*max(0,0.3-np.log10(x*0.0015+1))
    epsilon_list.append(epsilon)
plt.figure(figsize=(6,4))
plt.plot(epsilon_list)
plt.title('Probability of Trial')
plt.xlabel('episode')
plt.ylabel('epsilon')
plt.savefig('fig_epsilon.png',dpi=200)