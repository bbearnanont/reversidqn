from board import Board
import tensorflow as tf
import numpy
import os
import sys
import numpy as np
from model2 import HiddenLayer, DQN
if __name__ == '__main__':
    env = Board()
    lr = 0.001
    gamma = 0.99
    N = 2000
    players = []
    board_size = env.LENGTH**2
    players.append(DQN(board_size, board_size, [200, 200], lr, gamma))
    players.append(DQN(board_size, board_size, [200, 200], lr, gamma))
    wincount = np.zeros(len(players))
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)
    for player in players:
        player.set_session(session)
    
    for n in range(N):
        eps = 1.0/np.sqrt(n+1)
        observation = env.reset()
        iters = 0
        while not env.game_over():
            #env.draw_board()
            #print(env.game_over())
            for i in range(len(players)):
                tile = -1 if i%2 == 0 else 1
                action, _ = players[i].sample_action(env.get_state(), eps, tile)
                #print('action=',action)
                prev_observation = observation
                if action is not None:
                    x, y = env.flat_to_ij(action)
                #print(x,y)
                    env.put_tile(tile, x, y)
                observation = env.get_state()
                reward = env.reward(tile)
                players[i].add_experience(prev_observation, action, reward, observation, env.game_over())
                players[i].train(tile)
                iters += 1
                if env.game_over():
                    #env.draw_board()
                    if env.winner == -1:
                        wincount[0]+=1
                    elif env.winner == 1:
                        wincount[1]+=1
                    print('Episode',n,'finished!')
                    print('wincount of black:',wincount[0],'wincount of white:',wincount[1])
    players[1].save_model()
            