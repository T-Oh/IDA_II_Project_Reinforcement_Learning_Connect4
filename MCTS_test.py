# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 16:27:38 2022

@author: tobia
"""
import sys
sys.path.append(".")
import numpy as np
import pickle
from connect4 import connect4
from QValues import QValues
from MCTS import MCTS
from scipy import stats
 

def MCTS_test2(MCTS,nr_tests,q_opponent=None):
    #MCTS is always player 1 (this only refers to the token not to who starts)
    random=False                #distinguishes between random play or q play of opponent
    played_on_tree=nr_tests     #used to analyzie how many games are actually completely played on the tree
    play_tree=True              #as long as the nodes exist in the tree this is True
    if q_opponent==None:
        random=True
    wins=[]         #counts wins of MCTS
    env=connect4()  #set up environemnt
    #testing
    for i in range(nr_tests):
        env.reset()
        node_id=1
        if play_tree==False:
            played_on_tree-=1   
            play_tree=True
        win=-1
        #if p2 starts
        '''if i%2+1==2:         
            env.reset(2)
            if random: 
                env.random_play()
            else:
                actions=env.get_actions()
                action=q_opponent.max_action(env.board,actions,learning=False)
                env.execute(action)
            for child in MCTS.tree[2]['child']:
                if (env.board==MCTS.tree[child]['state']).all():
                    node_id=child
                    break;'''
        
        #playing loop
        for j in range(env.MAX_TURNS):

            #############PLAYER 1#####################
            #if still on tree
            if play_tree:
                next_node=MCTS.get_best_node(node_id)
                if next_node==-1:
                    play_tree=False
                env.board=np.array(MCTS.tree[next_node]['state'])
                env.turn=2
            #if no longer on tree        
            if not play_tree:   #if not on tree anymore fall back on q of MCTS
                actions=env.get_actions()
                action=MCTS.q.max_action(env.board,actions,learning=False)
                env.execute(action)
            win=env.check_end()
            if win !=-1:
                wins.append(win)
                break;

            ############PLAYER 2##############
            old_node=node_id
            if random:  #if no q_opponent loaded p2 plays random moves
                env.random_play()
            else:       #otherwise plays according to q_opponent
                actions=env.get_actions()
                action=q_opponent.max_action(env.board,actions,learning=False)
                env.execute(action)
            win=env.check_end()
            #env.print_board()
            if win!=-1: 
                wins.append(win)
                break;
            for child in MCTS.tree[next_node]['child']: #check if action of p2 is on tree
                if (env.board==MCTS.tree[child]['state']).all():
                    node_id=child
                    break;
            #if node does not exist tree can not longer be played
            if node_id==old_node:
                play_tree=False

    return wins, played_on_tree/nr_tests
                    
                    
            
            
                    
q_train=1000    #Nr of episodes the q strategy of MCTS was learnd -> used to load correct q file
#arrays to get parameters for plotting (wins to calc win ratio, std errors and p values) for different MCTS agents
wins_tot_arr=[]
std_arr=[]
p_arr=[]
for j in [9000,79000]:  #array of episodes trained for the MCTS -> used to load correct MCTS trees
    #load q of MCTS (should be same q as was trained with)  
    filename=   'D:\Dokumente\Studium\Master\IDA2\Project/p2r_sp1_ep'+str(q_train)+'_rw10_rl10_r0.pkl'    
    f=open(filename,'rb')
    q=pickle.load(f)
    f.close()   
    MCTS_agent=MCTS(q)
    
    
    #load MCTS tree
    filename='D:\Dokumente\Studium\Master\IDA2\Project/MCTS_q'+str(q_train)+'_sp1_ep'+str(j)+'_rw10_rl10_r0.pkl'
    f=open(filename,'rb')
    MCTS_agent.tree=pickle.load(f)
    f.close()   
    
    #load opponent q 
    '''f=open('D:\Dokumente\Studium\Master\IDA2/Project/p2r_sp1_ep1000_rw10_rl10_r0.pkl','rb')
    q_opponent=pickle.load(f)
    f.close()   '''
    
    #run test
    wins,played_on_tree=MCTS_test2(MCTS_agent,1000)
    
    split_wins=np.ndarray([10]) #split the test in 10 sets of 100 for t-test
    for i in range(10):
        values , p1_wins=np.unique(wins[100*(i):100*(i+1)-1],return_counts=True) #get the wins of p1 in each split
        split_wins[i]=p1_wins[values==1]    
    p1_wins=sum(split_wins)     #total wins of p1 in the 1000 tests
    p_sr=stats.ttest_1samp(split_wins/100,0.50)[1]  #get p value
    p2r_sr_counts=np.array(p1_wins/1000)    #get win ratio of MCTS
    std=np.std(split_wins/100)  #get std error
    
    values ,wins=np.unique(wins,return_counts=True)
    wins_tot=wins[values==1]
    wins_tot_arr.append(wins_tot/1000)
    std_arr.append(std)
    p_arr.append(p_sr)
    print('MCTS won: ',wins_tot/1000)
    print(played_on_tree,' games could be completed on the tree')
    print('STD: ',std)
    print(p_sr)
print(wins_tot_arr)
print(std_arr)
print(p_arr)