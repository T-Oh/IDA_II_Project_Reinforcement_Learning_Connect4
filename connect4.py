#adapted from https://github.com/KeithGalli/Connect4-Python/blob/master/connect4.py



import numpy as np
#import pygame
#import sys
#import math

BLUE = (0,0,255)
BLACK = (0,0,0)
RED = (255,0,0)
YELLOW = (255,255,0)



class connect4:
    
    def __init__(self,row_count=6,column_count=7):
        #init the class object
        self.ROW_COUNT = row_count
        self.COLUMN_COUNT = column_count
        self.MAX_TURNS=row_count*column_count
        self.board = np.zeros((row_count,column_count)) #board is initilaized with 0
        self.turn=1
        self.won=0
        
        
    def reset(self,start_turn=1):
        #reset to empty board with given start_turn
        self.board=np.zeros((self.ROW_COUNT,self.COLUMN_COUNT))
        self.turn=start_turn
        self.won=0
        
    def execute(self,action,turn=1):
        #uptdate board with given action and turn and returns reward
        #tile where piece is placed is set to 1 or 2 respectively depending on value of self.turn (the parameter turn is an unused rudiment )
        self.board[int(action[0])][int(action[1])] = self.turn
        self.turn=self.turn%2+1
        #turn is only applied if agent plays: positive reward for win, negative for loose, neutral otherwise
        reward=0
        if self.winning_move(turn):
            reward=10
        elif self.winning_move(turn%2+1):
            reward=-10
        return self.board,reward    

    
    def is_valid_location(self, col):
        #check if location is valid
        return self.board[self.ROW_COUNT-1][col] == 0

    def get_next_open_row(self, col):
        #gets next open row for given column
        for r in range(self.ROW_COUNT):
            if self.board[r][col] == 0:
                return r
            
    def print_board(self):
        print(np.flip(self.board, 0))
        
    def get_actions(self):
        #get possible actions for board, returns [-1,-1] if no possible actions or board is won
        actions=np.array([-1,-1])
        if self.winning_move(1) or self.winning_move(2):    #if someone won the game return no actions (-1)
            return actions
        for i in range(self.COLUMN_COUNT):
            if self.board[self.ROW_COUNT-1,i] != 0: #empty tile is 0
                continue
            else:
                row=self.get_next_open_row(i)
                if np.array_equal(actions,np.array([-1,-1])):
                    actions=np.array([[row,i]])
                else:
                    actions=np.append(actions,[[row,i]],axis=0)     
        return actions

    def winning_move(self,turn):
        #check if player 'turn' won the game
    	# Check horizontal locations for win
    	for c in range(self.COLUMN_COUNT-3):
    		for r in range(self.ROW_COUNT):
    			if self.board[r][c] == turn and self.board[r][c+1] == turn and self.board[r][c+2] == turn and self.board[r][c+3] == turn:
    				return True
    
    	# Check vertical locations for win
    	for c in range(self.COLUMN_COUNT):
    		for r in range(self.ROW_COUNT-3):
    			if self.board[r][c] == turn and self.board[r+1][c] == turn and self.board[r+2][c] == turn and self.board[r+3][c] == turn:
    				return True
    
    	# Check positively sloped diaganols
    	for c in range(self.COLUMN_COUNT-3):
    		for r in range(self.ROW_COUNT-3):
    			if self.board[r][c] == turn and self.board[r+1][c+1] == turn and self.board[r+2][c+2] == turn and self.board[r+3][c+3] == turn:
    				return True
    
    	# Check negatively sloped diaganols
    	for c in range(self.COLUMN_COUNT-3):
    		for r in range(3, self.ROW_COUNT):
    			if self.board[r][c] == turn and self.board[r-1][c+1] == turn and self.board[r-2][c+2] == turn and self.board[r-3][c+3] == turn:
    				return True

    def check_end(self):
        #checks if one of the players has won or if the board is full
        if self.winning_move(1):
            return 1
        if self.winning_move(2):
            return 2
        actions=self.get_actions()
        if np.all(actions==-1):
            return 0
        return -1 

    def random_action(actions):
        #selects a random action out of given actions
        N_poss=len(actions)    #number of possibilities
        j=np.random.choice(range(N_poss)) 
        action=actions[j]
        return action 
    
    def random_play(self):
        #executes a random action depending on board and turn of self 
        #returns -1 if succesful otherwise the winner BEFORE the random play or 0 if board is fuull
        actions=self.get_actions()
        if np.all(actions==-1):
            if self.winning_move(1):
                return 1
            elif self.winning_move(2):
                return 2
            else:
                return 0
        action=connect4.random_action(actions)
        self.execute(action)
        return -1
        