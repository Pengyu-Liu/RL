#PPO for LunarLander
import numpy as np
import gym
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#tensorflow 2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Input
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp


#Actor network for discrete version
class Actor(tf.keras.Model):
    def __init__(self):
        super(Actor, self).__init__()
        #self.network=Sequential()
        #input state size is 8 for lunarlander
        self.dense0=Dense(64, input_shape=(8,),activation='relu')
        self.dense1=Dense(64,activation='relu')
        #check action space: 4 for dicrete lunarlander. Output probability of 4 action choices
        self.outlayer=Dense(4, activation='softmax')

    def call(self,state):
        input_data=tf.convert_to_tensor(np.array([state]))
        x=self.dense0(input_data)
        x=self.dense1(x)
        return self.outlayer(x)

    def get_log_prob(self,state,action):
    #get the log probability of the action of a state
        input_data=tf.convert_to_tensor(np.array([state]))
        x=self.dense0(input_data)
        x=self.dense1(x)
        output_data=self.outlayer(x)
        dist = tfp.distributions.Categorical(probs=output_data, dtype=tf.float32)
        return  dist.log_prob(action)


#Critic network
class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        #self.network=Sequential()
        #input state size
        self.dense2=Dense(64, input_shape=(8,),activation='relu')
        self.dense3=Dense(64,activation='relu')
        #predict a value
        self.outlayer2=Dense(1, activation=None)

    def call(self,state):
        x=self.dense2(state)
        x=self.dense3(x)
        return self.outlayer2(x)

#ppo agent of discrete action space
class PPO_agent:
    def __init__(self,lr_actor, lr_critic):
        self.actor=Actor()
        self.critic=Critic()
        #optimizer and learning rate
        self.actor_opt=Adam(lr_actor)
        self.critic_opt=Adam(lr_critic)

    def get_action(self,state):
        #sample an action given a state
        output_data=self.actor(state)
        #make out put into a distribution, though softmax already gives the distribution
        dist = tfp.distributions.Categorical(probs=output_data, dtype=tf.float32)
        #take an action
        action = dist.sample()
        return int(action.numpy()[0])

#continuous Actor
class Actor_continuous(tf.keras.Model):
    def __init__(self):
        super(Actor_continuous, self).__init__()
        #self.network=Sequential()
        #input state size
        self.dense4=Dense(64, input_shape=(8,),activation='relu')
        self.dense5=Dense(64,activation='relu')
        #2 pairs of mu and sigma for two normal distributions
        self.mu=Dense(2, activation='tanh')
        self.sigma=Dense(2,activation='softplus')

    def call(self,state):
        input_data=tf.convert_to_tensor(np.array([state]))
        x=self.dense4(input_data)
        x=self.dense5(x)
        mu=self.mu(x)
        sigma=self.sigma(x)
        return mu, sigma

    def get_log_prob(self,state,action):
    #get the log probability of the action in the state
        mu,sigma=self.call(state)
        mu,sigma=mu[0],sigma[0]
        dist = tfp.distributions.Normal(loc=tf.reshape(mu,(len(mu),)),scale=tf.reshape(sigma,(len(sigma),)))
        #sum the two log probabilities
        return  tf.reduce_sum(dist.log_prob(action))

#continous PPO agent
class PPO_agent_con:
    def __init__(self,lr_actor, lr_critic,):
        self.actor=Actor_continuous()
        self.critic=Critic()
        self.actor_opt=Adam(lr_actor)
        self.critic_opt=Adam(lr_critic)

    def get_action(self,state):
        mu,sigma=self.actor(state)
        mu,sigma=mu[0],sigma[0]
        dist = tfp.distributions.Normal(loc=tf.reshape(mu,(len(mu),)),scale=tf.reshape(sigma,(len(sigma),)))
        #take an action
        action = dist.sample()
        return action.numpy()

def Actor_loss(old_logprob,new_probs,advantage,epsilon):
    #calculate loss of actor network
    advantage=tf.convert_to_tensor(advantage,dtype=tf.float32)
    #normalize advantange
    advantage = (advantage - tf.math.reduce_mean(advantage)) / (tf.math.reduce_std(advantage) + 1e-6)
    ratio=tf.math.exp(new_probs-old_logprob)
    surrogate1=ratio*advantage
    #clip
    loss=-tf.reduce_mean(tf.math.minimum(surrogate1,tf.clip_by_value(ratio,1-epsilon,1+epsilon)*advantage))
    return loss

def Critic_loss(returns,V):
    #critic loss: mse of V_target and V_predicted
    loss=tf.keras.losses.mean_squared_error(returns, V)
    return loss

def compute_advantage(gamma, lambd, buffer_reward,buffer_value,buffle_done):
    #calculate advantage
    advantage=np.zeros(len(buffer_reward))
    a_t=0
    for i in reversed(range(len(buffer_reward))):
        #start from the last step to save computation time
        delta=buffer_reward[i]+(gamma*buffer_value[i+1]-buffer_value[i])*(1-buffer_done[i])
        a_t=delta+gamma*lambd*a_t
        advantage[i]=a_t
    V_target=advantage+buffer_value[:-1]
    return advantage, V_target

#main program-------------------------------------------------------------------
#learning rate: actor and critic
lr_actor=3e-3
lr_critic=3e-3
#discount factor
gamma=0.99
#clip factor
epsilon=0.2
#smoothing factor in advantage function calculation
lambd=1
#the number of training episodes
episode=500
#length of trajectory used in updating
legth_T=32
#max step for each episode: found very useful
max_step=300

#discret action environment
# env=gym.make("LunarLander-v2")
# ppo=PPO_agent(lr_actor,lr_critic)
#continuous action evironment
env=gym.make("LunarLanderContinuous-v2")
ppo=PPO_agent_con(lr_actor,lr_critic)

#n_steps=0
total_reward=np.zeros(episode)
learn_iteraion=0
#start training--------------------------------------------------------
for i_episode in range(episode):
    state=env.reset()
    #env.render()
    done=False
    buffer_state, buffer_action, buffer_reward, buffer_done= [],[],[],[]
    while not done and len(buffer_reward)<max_step:
        #rollout
        #store state
        buffer_state.append(state.copy())
        #take an action
        action=ppo.get_action(state)
        #env.render()
        #interact with the enviroment
        state, reward, done, _ = env.step(action)
        #n_steps+=1
        buffer_action.append(action)
        buffer_reward.append(reward)
        buffer_done.append(done)
    #total reward of this episode
    total_reward[i_episode]=sum(buffer_reward)
    indices=np.arange(len(buffer_reward),dtype='int')
    #shuffle sampled trajectory: SGD
    np.random.shuffle(indices)
    epoch=len(buffer_action)//legth_T
    #divide the full trajectory into small groups with a length of legth_T (except for the last group)
    epoch_start=np.arange(0,len(buffer_reward),legth_T)
    #old probability
    old_logprob=np.zeros(len(buffer_reward))
    #value
    buffer_value=np.zeros(len(buffer_reward)+1)
    for i,(i_state, i_action) in enumerate(zip(buffer_state,buffer_action)):
        buffer_value[i]=ppo.critic(np.array([i_state])).numpy()
        old_logprob[i]=ppo.actor.get_log_prob(i_state,i_action)
    #one more value: the state after taking the last action
    buffer_value[-1]=ppo.critic(np.array([state])).numpy()
    #calculate advantage
    advantage, V_target=compute_advantage(gamma, lambd, buffer_reward,buffer_value,buffer_done)
    #make list to array
    buffer_action=np.asarray(buffer_action)
    buffer_state=np.asarray(buffer_state)
    #update policy epoch+1 times
    for idx in epoch_start:
        learn_iteraion+=1
        ind=np.array(indices[idx:idx+legth_T],dtype='int')
        sample_state=buffer_state[ind]
        sample_action=buffer_action[ind]
        sample_old_logprob=old_logprob[ind]
        sample_target=V_target[ind]
        sample_advantage=advantage[ind]
        new_logprob=[]
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            #calculate new log prob
            for i_state, i_action in zip(sample_state,sample_action):
                new_logprob.append(ppo.actor.get_log_prob(i_state,i_action))
            new_logprob=tf.reshape(new_logprob,(len(new_logprob),))
            #calculate predicted values
            V=ppo.critic(sample_state)
            V=tf.reshape(V,(len(V),))
            critic_loss=Critic_loss(tf.stop_gradient(sample_target),V)
            actor_loss=Actor_loss(sample_old_logprob,new_logprob,sample_advantage,epsilon)
        #update networks
        grads1 = tape1.gradient(actor_loss, ppo.actor.trainable_variables)
        grads2 = tape2.gradient(critic_loss, ppo.critic.trainable_variables)
        ppo.actor_opt.apply_gradients(zip(grads1, ppo.actor.trainable_variables))
        ppo.critic_opt.apply_gradients(zip(grads2, ppo.critic.trainable_variables))

    buffer_state, buffer_action, buffer_reward, buffer_done= [],[],[],[]
    print('episode', i_episode,'epoch', epoch,'total reward', total_reward[i_episode])

env.close()
np.savetxt('total_reward.txt',total_reward)
plt.figure()
plt.plot(np.arange(episode),total_reward)
plt.xlabel('episode')
plt.ylabel('rewards')
plt.savefig('training_curve.png',dpi=300)
print('learn_iteraion',learn_iteraion)
