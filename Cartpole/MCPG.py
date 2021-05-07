#Monte-Carlo policy gradient
import numpy as np
import gym
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Input
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
import random

#generate an action from the policy network given a state
def get_action(policy_network,state):
    input_data=tf.convert_to_tensor([state.astype('float32')])
    output_data=policy_network(input_data)
    #make out put into a distribution
    dist = tfp.distributions.Categorical(probs=output_data, dtype=tf.float32)
    #take an action
    action = dist.sample()
    return int(action.numpy()[0])

def get_log_prob(policy_network,state,action):
    #get the log probability of the action in the state
    input_data=tf.convert_to_tensor([state.astype('float32')])
    output_data=policy_network(input_data)
    dist = tfp.distributions.Categorical(probs=output_data, dtype=tf.float32)
    return  dist.log_prob(action)

#loss function
def custom_loss(log_prob,reward):
    loss=tf.convert_to_tensor(log_prob)*reward
    return -tf.reduce_sum(loss)

#learning rate: 0.001, 0.00003
lr=0.0003
#discount factor: 0.9, 0.95, 0.99
gamma=0.95
episode=3000
max_step=600
#update policy per update number of episodes: recommend 1
update=1
env=gym.make('CartPole-v1')
#max deviation of location
location=env.observation_space.high[0]
#max deviation of angle
angle=env.observation_space.high[2]

#plolicy network
policy_network=Sequential()
policy_network.add(Dense(128, input_shape=(4,),activation='relu'))
policy_network.add(Dropout(0.5))
policy_network.add(Dense(128, activation="relu"))
policy_network.add(Dense(env.action_space.n, activation="softmax"))
policy_network.summary()
opt=Adam(lr)

d=0
averg_rewards=np.zeros(episode//update)
averg_loss=np.zeros(episode//update)
discount_rewards=[]
states=[]
actions=[]
for i_episode in range(episode):
    observation=env.reset()
    rewards=[]
    my_rewards=[]
    for t in range(max_step):
        #monte carlo sample
        #env.render()
        #store observation
        states.append(observation.copy())
        #take an action
        action=get_action(policy_network,observation)
        #interact with the enviroment
        observation, reward, done, info = env.step(action)
        #store action and reward of this action
        #our way of calculating reward
        my_rewards.append((1-abs(observation[0]/location)*(1-abs(observation[2])/angle)))
        rewards.append(reward)
        actions.append(action)
        if done:
            break
    #actually, this is the cumulative reward
    averg_rewards[d]+=sum(rewards)
    for i in range(len(rewards)):
        for j in range(1,len(rewards)-i):
            #discount rewards
            #use the following line if you want to use reward directly returned by env
            #rewards[i]+=rewards[i+j]*np.power(gamma,j).astype('float32')
            my_rewards[i]+=my_rewards[i+j]*np.power(gamma,j).astype('float32')
        #store discount reward of every episode
        #if use rewards to calculate the discount reward in the previous step, please use this line to pass values to discount reward
        #discount_rewards.append(rewards[i])
        discount_rewards.append(my_rewards[i])

    #update policy_network per update episodes
    if (i_episode+1)%update == 0:
        #normalization: don't recommend
        #discount_rewards=(discount_rewards-np.mean(discount_rewards))/(np.std(discount_rewards)+1e-10)
        log_probs=[]
        with tf.GradientTape() as tape:
            for i_state, i_action in zip(states,actions):
                #loss need to be calculated in tape
                log_prob=get_log_prob(policy_network,i_state,i_action)
                log_probs.append(log_prob)
            loss=custom_loss(log_probs,discount_rewards)/update
        grads=tape.gradient(loss, policy_network.trainable_variables)
        #update weights
        opt.apply_gradients(zip(grads,policy_network.trainable_variables))
        averg_rewards[d]= averg_rewards[d]/update
        averg_loss[d]= loss.numpy()
        discount_rewards=[]
        states=[]
        actions=[]
        print('iteration', d, 'average reward', averg_rewards[d])
        d+=1

env.close()
np.savetxt('averg{:}_lr{:}_gm{:}_loss.txt'.format(update,lr,gamma),averg_loss)
np.savetxt('averg{:}_lr{:}_gm{:}_rewards.txt'.format(update,lr,gamma),averg_rewards)
plt.figure()
plt.plot(np.arange(1,d+1)*update,averg_rewards)
plt.xlabel('episode')
plt.ylabel('average rewards')
plt.savefig('averg{:}_lr{:}_gm{:}_training_curve.png'.format(update,lr,gamma),dpi=150)
