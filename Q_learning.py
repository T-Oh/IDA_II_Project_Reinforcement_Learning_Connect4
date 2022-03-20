# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 14:16:03 2022
game from https://github.com/KeithGalli/Connect4-Python/blob/master/connect4.py

@author: tobia
"""
import sys
sys.path.append(".")
import numpy as np
import time
import pickle
from connect4 import connect4
from QValues import QValues
#import pygame
#import sys
#import math




#QLearning for connect 4, base taken from ReinforcementLearning exercie II   
def q_learning(env, q=None, nr_episodes=100, epsilon=0.1, alpha=0.1, gamma=0.98):
    #only player 1 learning, player 2 playing random
    if not q:
        q = QValues()

    for episode in range(nr_episodes):
        env.reset()   
        if episode%2==1:    #so that the agent learns as first and second player
            env.reset(2)    #set start turn to 2 in case random player starts -> so that agent is always p1
            actions=env.get_actions()
            action=connect4.random_action(actions)
            env.execute(action)
        agent_turn=env.turn
        for i in range(env.MAX_TURNS):
            state = np.array(env.board)
            actions = env.get_actions()
            #print(actions)
            if np.all(actions==-1): #-1 means no actions can be taken
                if env.winning_move(agent_turn):
                    q.wins=np.append(q.wins,[[episode,1]],axis=0)
                elif env.winning_move(agent_turn%2+1):
                    q.wins=np.append(q.wins,[[episode,2]],axis=0)
                else:
                    q.wins=np.append(q.wins,[[episode,0]],axis=0)
                break # final state reached

            action = q.epsilon_greedy(state, actions, epsilon)
            q_old = q.get_value(state, action)   
            #print('Q_OLD: ',q_old)
            _, reward = env.execute(action) #take action
            #env.print_board()
            #print('Player',env.turn)
            
            '''player 2'''
            actions=env.get_actions()
            #print(actions)
            if np.any(actions!=-1): #if random player can still move he moves
                p2_action=connect4.random_action(actions)
                env.execute(p2_action)
            if env.winning_move(2): reward=-10
            
            '''continue q_learning'''
            next_state = env.board
            next_max_action = q.max_action(next_state, env.get_actions())
            q_next = q.get_value(next_state, next_max_action)
            #print('Q_NEXT: ',q_next)
            q_new = q_old + alpha * (reward + gamma * q_next - q_old)
            #print('Q_NEW: ',q_new)
            #print(state,action)
            q.set_value(state, action, q_new)
        print('\nEpisode: ',episode)
        print('\n')
        #env.print_board()
        #print(q.values[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    return q 
 
print('start\n')
      
env = connect4()    #set environment to connect 4
q = q_learning(env, nr_episodes=70000)  #start q_learning 


#save policy (q lookup table)
f=open('D:/Dokumente/Studium/Master/IDA2/Project/p2r_sr_ep70000_rw10_rl10_r0.pkl','wb')
pickle.dump(q,f)
f.close()
#print('\nWins:\n')
#print(q.wins)
wins=q.wins[:,1]-1
print(sum(wins))    

            


        
    
            
        


