# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 11:32:33 2022

@author: tobia
"""
import sys
sys.path.append(".")
import numpy as np
import pickle
from QValues import QValues
from connect4 import connect4
from scipy import stats

#from connect4 import connect4

def test_q(q,env,N_tests=1000):
    wins=np.array([0])
    for i in range(N_tests):
        env.reset()   
        '''if i%2==1:    #so that the agent is tested as first and second player -> even when p1 is 2nd player his 'turn' will be 1 for easier evaluation
            env.reset(2)
            actions=env.get_actions()
            action=connect4.random_action(actions)
            env.execute(action)'''
        agent_turn=env.turn     #agent turn is always 1 (refers to piece in the board not that its actual first turn)
        for i in range(env.MAX_TURNS):
            state = np.array(env.board)
            actions = env.get_actions()
            #print(actions)
            if np.all(actions==-1): #-1 means no actions can be taken
                if env.winning_move(agent_turn):
                    wins=np.append(wins,[1])
                elif env.winning_move(agent_turn%2+1):
                    wins=np.append(wins,[2])
                else:
                    wins=np.append(wins,[0])
                break # final state reached

            action = q.max_action(state, actions, learning=False)
            env.execute(action) #take action
            
            '''player 2'''
            actions=env.get_actions()
            #print(actions)
            if np.any(actions!=-1): #if random player can still move he moves
                p2_action=connect4.random_action(actions)
                env.execute(p2_action)
    return wins
 

    
env=connect4()

###########Trained with P2: random, Start: P1 ##########################
#load file
f=open('D:\Dokumente\Studium\Master\IDA2\Project/p2r_sp1_ep100_rw10_rl10_r0.pkl','rb')
q=pickle.load(f)
f.close()

wins=test_q(q,env,1000)     #test
split_wins=np.ndarray([10]) #split the test in 10 sets of 100 for t-test
for i in range(10):
    _ , p1_wins=np.unique(wins[100*(i):100*(i+1)-1],return_counts=True) #get the wins of p1 in each split
    split_wins[i]=p1_wins[1]   
p1_wins=sum(split_wins)     #total wins of p1 in the 1000 tests
p_sp1=np.array(stats.ttest_1samp(split_wins/100,0.50)[1])
p2r_sp1_counts=np.array(p1_wins/1000)
std=np.std(split_wins/100)



for i in [1000,10000,80000]:
    filename='D:\Dokumente\Studium\Master\IDA2\Project/p2r_sp1_ep'+str(i)+'_rw10_rl10_r0.pkl'
    f=open(filename,'rb')
    q=pickle.load(f)
    f.close()
    
    wins=test_q(q,env,1000)
    split_wins=np.ndarray([10]) #split the test in 10 sets of 100 for t-test
    for i in range(10):
        values , p1_wins=np.unique(wins[100*(i):100*(i+1)-1],return_counts=True) #get the wins of p1 in each split
        split_wins[i]=p1_wins[values==1]    
    p1_wins=sum(split_wins)     #total wins of p1 in the 1000 tests
    p_sp1=np.append(p_sp1,stats.ttest_1samp(split_wins/100,0.50)[1])
    p2r_sp1_counts=np.append(p2r_sp1_counts,np.array(p1_wins/1000))
    std=np.append(std,np.std(split_wins/100))
p2r_sp1_err=std    
    
    
##################p2 random start random trained bots##################
'''f=open('D:\Dokumente\Studium\Master\IDA2\Project/random_start/p2r_sr_ep100_rw10_rl10_r0.pkl','rb')
q=pickle.load(f)
f.close()

wins=test_q(q,env,1000)     #test
split_wins=np.ndarray([10]) #split the test in 10 sets of 100 for t-test
for i in range(10):
    _ , p1_wins=np.unique(wins[100*(i):100*(i+1)-1],return_counts=True) #get the wins of p1 in each split
    split_wins[i]=p1_wins[1]   
p1_wins=sum(split_wins)     #total wins of p1 in the 1000 tests
p_sr=np.array(stats.ttest_1samp(split_wins/100,0.50)[1])
p2r_sr_counts=np.array(p1_wins/1000)
std=np.std(split_wins/100)



for i in [1000,10000,70000]:
    filename='D:\Dokumente\Studium\Master\IDA2\Project/random_start/p2r_sr_ep'+str(i)+'_rw10_rl10_r0.pkl'
    f=open(filename,'rb')
    q=pickle.load(f)
    f.close()
    wins=test_q(q,env,1000)
    split_wins=np.ndarray([10]) #split the test in 10 sets of 100 for t-test
    for i in range(10):
        values , p1_wins=np.unique(wins[100*(i):100*(i+1)-1],return_counts=True) #get the wins of p1 in each split
        split_wins[i]=p1_wins[values==1]    
    p1_wins=sum(split_wins)     #total wins of p1 in the 1000 tests
    p_sr=np.append(p_sr,stats.ttest_1samp(split_wins/100,0.50)[1])
    p2r_sr_counts=np.append(p2r_sr_counts,np.array(p1_wins/1000))
    std=np.append(std,np.std(split_wins/100))
p2r_sr_err=std  '''

print(p2r_sp1_counts)  
print(p2r_sp1_err)
#MCTS data
MCTS_10k_counts=[0,0,0,0.55]
MCTS_10k_std=[0,0,0,0.023]
MCTS_1k_counts=[0,0,0.5,0.5]
MCTS_1k_std=[0,0,0.038,0.034]
MCTS_100_counts=[0,0.52,0.51,0.52]
MCTS_100_std=[0,0.032,0.04,0.023]



    