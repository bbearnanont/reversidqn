import numpy as np

class Board:
    def __init__(self, LENGTH = 8):
        self.LENGTH = LENGTH
        self.board = np.zeros((LENGTH, LENGTH))
        self.BLACK = -1
        self.WHITE = 1
        self.ended = False
        self.score = {}
        self.winner = None
    def reset(self):
        self.winner = None
        self.ended = False
        self.board = np.zeros((self.LENGTH, self.LENGTH))
        middle = int(self.LENGTH/2)
        self.board[middle-1, middle-1] = self.BLACK 
        self.board[middle, middle] = self.BLACK
        self.board[middle-1, middle] = self.WHITE
        self.board[middle, middle-1] = self.WHITE
        self.get_score()
        return self.board

    def get_state(self):
        return self.board
 
    def get_score(self):
        count_black = 0
        count_white = 0
        for temp in range(self.LENGTH**2):
            if self.board[int(temp/self.LENGTH)][int(temp%self.LENGTH)] == -1:
                count_black+=1
            elif self.board[int(temp/self.LENGTH)][int(temp%self.LENGTH)] ==1:
                count_white+=1
        self.score = {-1:count_black,1:count_white}

        #unique, counts = np.unique(self.board, return_counts=True)
        #print(unique, counts)
        #self.score = {-1:dict(zip(unique, counts))[-1],1:dict(zip(unique, counts))[1]}
        return self.score

    def is_OnBoard(self, i, j):
        return i >= 0 and i < self.LENGTH and j >= 0 and j < self.LENGTH

    def is_valid_move(self, tile, i, j):
        tiles_to_flip = []
        if not self.is_OnBoard(i, j) or self.board[i, j] != 0:
            return False
        othertile = tile * (-1)
        #check all direction for flanked opponent's tiles or row
        for xdir, ydir in ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)):
            x, y = i, j
            x += xdir
            y += ydir
            if self.is_OnBoard(x, y) and self.board[x][y] == othertile:
                if not self.is_OnBoard(x, y):
                    continue
                while self.board[x][y] == othertile:
                    x += xdir
                    y += ydir
                    if not self.is_OnBoard(x, y):
                        break
                if not self.is_OnBoard(x, y):
                    continue
                if self.board[x][y] == tile:
                    while True:
                        x -= xdir
                        y -= ydir
                        if x == i and y == j:
                            break
                        tiles_to_flip.append([x, y])
        #return opponent's tiles to flip
        return tiles_to_flip    

    def game_over(self, force_recalculate=False):
        #return True if game is over
        #otherwise return False
        #set winner
        if not force_recalculate and self.ended:
            return self.ended
        
        #check possible move on both side
        #if there is any possible move on bothside = game is not over yet
        for i in range(self.LENGTH):
            for j in range(self.LENGTH):
                if self.is_valid_move(self.BLACK, i, j):
                    return False
                if self.is_valid_move(self.WHITE, i, j):
                    return False
        self.get_score()
        #set black win if black > white
        if self.score[self.BLACK] > self.score[self.WHITE]:
            self.winner = self.BLACK
        #set white win if white > black
        elif self.score[self.BLACK] < self.score[self.WHITE]:
            self.winner = self.WHITE
        #draw if white == black
        else:
            self.winner = None
        self.ended = True
        return True
    
    def reward(self, tile):
        if not self.game_over():
            return 0
        
        #only get a reward when archieve victory
        return 1 if self.winner == tile else 0
    
        #check draw if score is equal and no possible move on both side
    
    def is_draw(self):
        return self.ended and self.winner is None

    def put_tile(self, tile, i, j):
        #put tile on the board (it must pass the poilicy of the game first)
        #by is_valid_move(tile, i, j)
        tiles_to_flip = self.is_valid_move(tile, i, j)
        for n in range(len(tiles_to_flip)):
            self.board[tiles_to_flip[n][0], tiles_to_flip[n][1]] *= -1
        self.board[i, j] = tile
        #updating score after put the tile and flip flanked tile
        self.get_score()
        return self.board

    def check_possible_moves(self, tile):
            possible_moves = []
            for i in range(self.LENGTH):
                for j in range(self.LENGTH):
                    if self.is_valid_move(tile, i, j):
                        possible_moves.append((i, j))
            if possible_moves:
                return possible_moves
            else:
                return False
    
    def draw_board(self):
        print("   0j   1j   2j   3j   4j   5j   6j   7j")
        print("----------------------------------------")
        for i in range(self.LENGTH):
            print(i,'i|',end='',sep='')
            for j in range(self.LENGTH):
                if self.board[i,j] == self.BLACK:
                    print("B    ", end="")
                elif self.board[i,j] == self.WHITE:
                    print("W    ", end="")
                else:
                    print("0    ", end="")
                
            print("")
            print("----------------------------------------")

    def flat_to_ij(self, flat):
        if flat == 0:
            return 0, 0
        i = int(flat / self.LENGTH)
        j = flat % self.LENGTH
        return i, j
    
    def ij_to_flat(self, i, j):
        return i*self.LENGTH + j

if __name__ == '__main__':
    env = Board()
    env.reset()
    env.draw_board()
    print(env.get_score())
    print(env.game_over())
    #print(env.check_possible_moves(1))
    #print(env.get_score())
