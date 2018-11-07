from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import board
import gym
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
count = 3
def checkdl():
    print('DL')
class HiddenLayer:
    def __init__(self, M1, M2, f = tf.nn.relu, use_bias = True):
        self.W = tf.Variable(tf.random_normal(shape=(M1, M2)))
        self.params = [self.W]
        self.use_bias = use_bias
        if use_bias:
            self.b = tf.Variable(np.zeros(M2).astype(np.float32))
            self.params.append(self.b)
        self.f = f

    def forward(self, X):
        if self.use_bias:
            a = tf.matmul(X, self.W) + self.b
        else:
            a = tf.matmul(X, self.W)
        return self.f(a)

class DQN:
    def __init__(self, D, K, hidden_layer_sizes, gamma, max_experiences = 10000, min_experiences = 100, batch_sz = 32):
        self.K = K
        self.layers = []
        M1 = D
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            self.layers.append(layer)
            M1 = M2

        layer = HiddenLayer(M1, K, lambda x: x)
        self.layers.append(layer)
        
        self.params = []
        for layer in self.layers:
            self.params += layer.params
        
        self.X = tf.placeholder(tf.float32, shape=(None, 8, 8), name='X')
        self.X_flat = tf.reshape(self.X, [-1, 64], name='X_flat')
        self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
        self.actions = tf.placeholder(tf.int32, shape=(None, D), name='actions')

        Z = self.X_flat
        for layer in self.layers:
            Z = layer.forward(Z)
        Y_hat = Z
        self.predict_op = Y_hat

        '''
        selected_action_values = tf.reduce_sum(
            Y_hat * tf.one_hot(self.actions, K),
            reduction_indices = [1]
        )
        '''     
        cost = tf.reduce_mean(tf.square(self.G - self.predict_op))
        self.train_op = tf.train.RMSPropOptimizer(0.001).minimize(cost)

        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.batch_sz = batch_sz
        self.gamma = gamma

    def set_session(self, session):
        self.session = session

    def copy_from(self, other):
        ops = []
        my_params = self.params
        other_params = other.params
        for p, q in zip(my_params, other_params):
            actual = self.session.run(q)
            op = p.assign(actual)
            ops.append(op)
        self.session.run(ops)

    def predict(self, X):
        #print(X)
        return self.session.run(self.predict_op, feed_dict = {self.X: [X]})[0]

    def train(self, target_network):
    # sample a random batch from buffer, do an iteration of GD
        if len(self.experience['s']) < self.min_experiences:
      # don't do anything if we don't have enough experience
            return

    # randomly select a batch
        idx = np.random.choice(len(self.experience['s']), size=self.batch_sz, replace=False)
     # print("idx:", idx)
        states = [self.experience['s'][i] for i in idx]
        actions = [self.experience['a'][i] for i in idx]
        rewards = [self.experience['r'][i] for i in idx]
        next_states = [self.experience['s2'][i] for i in idx]
        dones = [self.experience['done'][i] for i in idx]
        next_Q = [np.max(target_network.predict(next_states[i])) for i in range(len(idx))]
        #print(len(next_Q))
        #print(len(rewards))
        #print(dones)
        for i in range(len(idx)):
            next_Q = target_network.predict(states[i])
            if dones[i]:
                next_Q[actions[i]] = rewards[i]
            else:
                action = target_network.sample_action(next_states[i])
                next_Q[actions[i]] = rewards[i] + target_network.gamma * next_Q[target_network.ij_to_a(action[0],action[1])]                
        '''
        targets = [r + self.gamma*next_q if not done else r for r, next_q, done in zip(rewards, next_Q, dones)]

        #targets = [r + self.gamma*next_q if not done else r for r, next_q, done in zip(rewards, next_Q, dones)]
        for i in idx:
            next_Q = target_network.predict(state[i])
            if done[i]:
                next_Q[actions[i]] = rewards[i]
            else:
           
    # call optimizer
        for i in range(len(idx)):
            self.session.run(
            self.train_op,
            feed_dict={
                self.X: [states[i]],
                self.G: [targets[i]],
                self.actions: [actions[i]]
            }
            )
        '''
    def add_experience(self, s, a, r, s2, done):
        if len(self.experience['s']) >= self.max_experiences:
            self.experience['s'].pop(0)
            self.experience['a'].pop(0)
            self.experience['r'].pop(0)
            self.experience['s2'].pop(0)
            self.experience['done'].pop(0)
        self.experience['s'].append(s)
        self.experience['a'].append(a)
        self.experience['r'].append(r)
        self.experience['s2'].append(s2)
        self.experience['done'].append(done)

    def sample_action(self,env, eps, tile):
        state = env.get_state()
        possible_moves = env.check_possible_moves(tile)
        if not possible_moves:
            return None
        if np.random.random() < eps:
            choice = np.random.choice(len(possible_moves))
            next_move = possible_moves[choice]
            return next_move[0], next_move[1]
        else:
            targets = np.empty(len(possible_moves))
            for i in range(len(possible_moves)):
                targets[i] = int(possible_moves[i][0]*8 + possible_moves[i][1])
            Qs = self.predict(state)

            #print(Qs.shape)
            index = np.argsort(Qs)
            #print(Qs)
            for action in reversed(index):
                if action in targets:
                    break
            print(action)
            i = int(action/env.LENGTH)
            j = action%env.LENGTH
            return i, j

    def ij_to_a(self, i, j):
        return i*8 + j
class Agent:
    def __init__(self, env, color, sizes, gamma):
        self.env = env
        self.observation = None
        self.prev_observation = None
        self.reward = None
        self.totalreward = None
        if color == 'black':
            self.tile = -1
        else:
            self.tile = 1
        self.sizes = sizes
        self.gamma = gamma
        #D = len(env.reset())
        D = K = env.LENGTH**2
        sizes = [200, 200]
        self.model = DQN(D, K, sizes, gamma)
        self.tmodel = DQN(D, K, sizes, gamma)
        init = tf.global_variables_initializer()
        session = tf.InteractiveSession()
        session.run(init)
        self.model.set_session(session)
        self.tmodel.set_session(session)

def play_one(env, black, white, eps, gamma, copy_period, render = False):
    black.observation = white.observation = env.reset()
    black.totalreward = white.totalreward = 0
    current_player = None
    totalreward = 0
    iters = 0
    while not env.game_over():
        if render:
            env.draw_board()
        if current_player == black:
            current_player = white
            other_player = black
        else:
            current_player = black
            other_player = white
        action = current_player.model.sample_action(env, eps, current_player.tile)
        env.draw_board()
        #print(action)
        if action == None:
            continue
        current_player.prev_observation = current_player.observation
        env.put_tile(current_player.tile, action[0], action[1])
        current_player.observation = env.get_state()
        current_player.reward = env.reward(current_player.tile)
        current_player.totalreward += current_player.reward
        current_player.model.add_experience(current_player.prev_observation, current_player.model.ij_to_a(action[0], action[1]), current_player.reward, current_player.observation, env.game_over())
        current_player.model.train(current_player.tmodel)
        iters += 1
        if iters % copy_period == 0:
            current_player.tmodel.copy_from(current_player.model)
    

    return black.totalreward, white.totalreward

def main():
    env = board.Board()
    gamma = 0.99
    copy_period = 10
    N = 5
    black_agent = Agent(env, 'black', [200, 200], 0.99)
    white_agent = Agent(env, 'white', [200, 200], 0.99)
    totalrewards_black = totalrewards_white = np.empty(N)
    for n in range(N):
        eps = 1.0/np.sqrt(n+1)
        totalreward = totalrewards_black[n], totalrewards_white[n] = (play_one(env,black_agent, white_agent, eps, gamma, copy_period))
        if n % 1 == 0:
            print("episode:", n, "total reward:", totalreward, "eps:", eps)
    print("reward for last 100 episodes:", totalrewards_black[-100:].mean(), totalrewards_white[-100:].mean())
    play_one(env, black_agent, white_agent, eps = 0, gamma =1,copy_period = 1000000000000, render = True)

if __name__ == '__main__':
    main()