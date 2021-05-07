import numpy as np
import gym
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import multiprocessing
import threading
tf.disable_v2_behavior()

"""
Hyper parameters:
    lr                    # learning rate
    gamma                 # discount factor
    n_workers             # number of workers (2 to cpu.count)
    update_global         # update global network frequency
"""
lr = 0.0005
gamma = 0.99
update_global = 5
# n_workers = 4
n_workers = multiprocessing.cpu_count()


MAX_GLOBAL_EP = 2000
# For global network
GLOBAL_NET_SCOPE = 'Global_Net'
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0
# Environment of game
GAME = 'LunarLander-v2'
env = gym.make(GAME)
N_S = env.observation_space.shape[0]
N_A = env.action_space.n


# Network class can be used for both actor and critic
class ACNet(object):
    def __init__(self, scope, globalac=None):
        # Global network
        if scope == GLOBAL_NET_SCOPE:
            with tf.variable_scope(scope):
                # Shape: None is for batch, S is for observation of every state
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self._build_net(N_A)
                # Parameters for actor and critic
                self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        else:  # Local network to calculate loss
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')
                self.a_prob, self.v = self._build_net(N_A)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))   # TD square to avoid negative number

                with tf.name_scope('a_loss'):
                    # Log_probability
                    log_prob = tf.reduce_sum(tf.log(self.a_prob) * tf.one_hot(self.a_his, N_A, dtype=tf.float32),
                                             axis=1, keep_dims=True)
                    exp_v = log_prob * td
                    # Encourage exploration
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob), axis=1,
                                             keep_dims=True)
                    self.exp_v = 0.001 * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('local_grad'):
                    # Calculate the grad
                    self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                    self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

                with tf.name_scope('sync'):
                    # Sync the worker and global
                    with tf.name_scope('pull'):  # pull to local
                        self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalac.a_params)]
                        self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalac.c_params)]
                    with tf.name_scope('push'):  # push to global
                        self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalac.a_params))
                        self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalac.c_params))

    def _build_net(self, n_a):
        w_init = tf.random_normal_initializer(0., .01)  # Initialize the tensor with normal distribution
        # Critic: output the state value to calculate TD
        with tf.variable_scope('critic'):
            cell_size = 64
            s = tf.expand_dims(self.s, axis=1,
                               name='timely_input')  # [time_step, feature] => [time_step, batch, feature]
            rnn_cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(cell_size)
            self.init_state = rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
            outputs, self.final_state = tf.nn.dynamic_rnn(
                cell=rnn_cell, inputs=s, initial_state=self.init_state, time_major=True)
            cell_out = tf.reshape(outputs, [-1, cell_size], name='flatten_rnn_outputs')  # joined state representation
            l_c = tf.layers.dense(cell_out, 200, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        # Actor: output action and variance
        with tf.variable_scope('actor'):
            cell_out = tf.stop_gradient(cell_out, name='c_cell_out')
            l_a = tf.layers.dense(cell_out, 300, tf.nn.relu6, kernel_initializer=w_init, name='la')
            a_prob = tf.layers.dense(l_a, n_a, tf.nn.softmax, kernel_initializer=w_init, name='ap')
        return a_prob, v

    # Local grads applies to global net
    def update_global(self, feed_dict):
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)

    def pull_global(self):
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    # Select action w.r.t the actions prob
    def choose_action(self, s, cell_state):
        prob_weights, cell_state = SESS.run([self.a_prob, self.final_state], feed_dict={self.s: s[np.newaxis, :],
                                                                                        self.init_state: cell_state})
        prob_weights = prob_weights / prob_weights.sum()
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())
        return action, cell_state


class Worker(object):
    def __init__(self, name, globalac):
        self.env = gym.make(GAME)
        self.name = name
        self.AC = ACNet(name, globalac)

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        # Buffer for s, a, r, used to update n_steps
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            ep_t = 0
            # Zero rnn state at beginning
            rnn_state = SESS.run(self.AC.init_state)
            # Keep rnn state for updating global net
            keep_state = rnn_state.copy()
            while True:
                # self.env.render()
                a, rnn_state_ = self.AC.choose_action(s, rnn_state)
                s_, r, done, info = self.env.step(a)
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)
                # Update global and assign to local net
                if total_step % update_global  == 0 or done:
                    if done:
                        v_s_ = 0  # Terminal
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :], self.AC.init_state: rnn_state_})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:  # Reverse it
                        v_s_ = r + gamma * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(
                        buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                        self.AC.init_state: keep_state,
                    }

                    self.AC.update_global(feed_dict)

                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()
                    keep_state = rnn_state_.copy()  # Replace keep_state as the new initial rnn s_

                s = s_
                total_step += 1
                rnn_state = rnn_state_  # Rnn state renew
                ep_t += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # Record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)
                    if not self.env.unwrapped.lander.awake:
                        solve = '| Land'
                    else:
                        solve = '| ----'
                    print(
                        self.name,
                        "Episode:", GLOBAL_EP,
                        solve,
                        "| Cumulative Reward: %i" % GLOBAL_RUNNING_R[-1],
                    )
                    GLOBAL_EP += 1
                    break


if __name__ == "__main__":
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(lr, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(lr, name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)
        workers = []
        # Create worker
        for i in range(n_workers):
            i_name = 'Worker_%i' % i
            workers.append(Worker(i_name, GLOBAL_AC))

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)   # Create thread
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)   # Thread is managed uniformly by COORD

    # Save the list of the cumulative_rewards
    np.savetxt('l_a=0.0005,l_c=0.001,g=0.99,u_f=4,n_w=4(3).txt', GLOBAL_RUNNING_R)
    # Plot
    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.show()
