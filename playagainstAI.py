
import argparse
import tensorflow as tf
from board import Board
from model2 import DQN

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path")
    parser.add_argument("-s", "--save", dest="save", action="store_true")
    parser.set_defaults(save=False)
    args = parser.parse_args()

    # environmet, agent
    env = Board()
    board_size = env.LENGTH**2
    agent = DQN(board_size, board_size, [200, 200], 0, 0)
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)
    agent.set_session(session)
    agent.load_model(args.model_path)
    env.reset()
    while not env.game_over():
        #env.draw_board()
        for i in range(2):
            env.draw_board()        
            tile = -1 if i%2 == 0 else 1
            if i ==1:
                action, _ = agent.sample_action(env.get_state(), 0, tile)
                if action is not None:
                    x, y = env.flat_to_ij(action)
                    env.put_tile(tile, x, y)
                    print('White on',x, y)
                else:
                    print("No possible move, Skip turn!")
                    continue
            else:
                while True:
                    possible_moves = env.check_possible_moves(tile)
                    if not possible_moves:
                        print("No possible move, Skip turn!")
                        break
                    else:
                        print('Possible moves: ', possible_moves)
                        move = input("Enter coordinates i, j for your next move (ex:0,1):")
                        i, j = move.split(',')
                        i = int(i)
                        j = int(j)
                        if env.is_valid_move(tile, i, j):
                            env.put_tile(tile, i, j)
                            print('Black on',i,j)
                            break
                        else:
                            print("Valid Input!!")
    print(env.get_score())
                