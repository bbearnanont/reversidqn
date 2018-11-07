import board
import tensorflow as tf
import numpy as np
import os
import sys

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
    def __init__(self, input_size, output_size, hidden_layer_sizes, lr, gamma, max_experiences = 10000, min_experiences = 100, batch_size = 32):
        self.output_size = output_size
        self.layers = []
        self.lr = lr
        m1 = input_size
        for m2 in hidden_layer_sizes:
            layer = HiddenLayer(m1, m2)
            self.layers.append(layer)
            m1 = m2
        #last layer
        layer = HiddenLayer(m1, output_size, lambda x: x)
        self.layers.append(layer)

        self.params = []
        for layers in self.layers:
            self.params += layer.params
        self.X = tf.placeholder(tf.float32, shape=(None, 8, 8), name ='X')
        self.X_flat = tf.reshape(self.X, [-1, input_size], name ='X_flat')
        self.G = tf.placeholder(tf.float32, shape=(None), name = 'G')
        #self.actions = tf.placeholder(tf.int32, shape=(None, output_size) name = 'actions')
        Z = self.X_flat
        for layer in self.layers:
            Z = layer.forward(Z)
        self.predict_op = Z
        self.loss = tf.reduce_mean(tf.square(self.G - self.predict_op))
        self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
        self.experience = {'s':[], 'a':[], 'r':[], 's2':[], 'done':[]}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.batch_size = batch_size
        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        self.model_name = "{}.ckpt".format(os.path.splitext(os.path.basename(__file__))[0])
        self.gamma = gamma
        self.saver = tf.train.Saver()
    def set_session(self, session):
        self.session = session
    '''
    def copy_from(self, other):
        ops = []
        my_params = self.params
        other_params = other.params
        for p, q in zip(my_params, other_params):
        actual = self.session.run(q)
        op = p.assign(actual)
        ops.append(op)
        self.session.run(ops)
    '''
    def predict(self, X):
    #        print(X)
        return self.session.run(self.predict_op, feed_dict={self.X: [X]})   
    
    def train(self,tile):
        if len(self.experience['s']) < self.min_experiences:
            return
        state_minibatch = []
        y_minibatch = []
        idx = np.random.choice(len(self.experience['s']),size = self.batch_size, replace = False)
        states = [self.experience['s'][i] for i in idx]
        actions = [self.experience['a'][i] for i in idx]
        rewards = [self.experience['r'][i] for i in idx]
        next_states = [self.experience['s2'][i] for i in idx]
        done = [self.experience['done'][i] for i in idx]

        for i in range(len(idx)):
            y_i = self.predict(states[i])
            if done[i]:
                y_i[0][actions[i]] = rewards[i]
            else:
                action, qvalue = self.sample_action(next_states[i],0, tile)
                if action == None:
                    qvalue = y_i[0][actions[i]]
                y_i[0][actions[i]] = rewards[i] + self.gamma * qvalue
            #state_minibatch.append(states[i])
            #y_minibatch.append([y_i[0][i]])
            self.session.run(self.train_op, feed_dict={self.X: [states[i]],self.G:rewards[i] ,self.predict_op:y_i})
        #self.current_loss = self.sess.run(self.loss, feed_dict={self.x: state_minibatch, self.y: y_minibatch})

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

    def sample_action(self, state, eps, tile):
        env = board.Board()
        env.board = state
        possible_moves = env.check_possible_moves(tile)
        if possible_moves == False:
            return None, None
        if np.random.random() < eps:
            choice = np.random.choice(len(possible_moves))
            next_move = possible_moves[choice]
            return env.ij_to_flat(next_move[0], next_move[1]), None
        else:
            targets = np.empty(len(possible_moves))
            for i in range(len(possible_moves)):
                targets[i] = env.ij_to_flat(possible_moves[i][0], possible_moves[i][1])
            Qs = self.predict(state)
            index = np.argsort(Qs)
            for action in reversed(index[0]):
                #print(action)
                if action in targets:
                    break
            return action, Qs[0][action]
    
    def save_model(self):
        self.saver.save(self.session, os.path.join(self.model_dir, self.model_name))

    def load_model(self, model_path=None):
        if model_path:
            # load from model_path
            self.saver.restore(self.session, model_path)
        else:
            # load from checkpoint
            checkpoint = tf.train.get_checkpoint_state(self.model_dir)
            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore(self.session, checkpoint.model_checkpoint_path)
