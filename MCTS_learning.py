# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 17:12:08 2022

@author: tobia
"""

import sys
sys.path.append(".")
from MCTS import MCTS
import numpy as np
import time
import pickle
from connect4 import connect4
from QValues import QValues


def MCTS_learning(MCTS,nr_episodes,alpha=0.1,gamma=0.9,epsilon=0.05):
    wins=[]
    for i in range(nr_episodes):
        #set start node (p1 or p2)
        #start_node=i%2+1    #agent and opponent take turns in who starts
        start_node=1
        if start_node==2:   #if start node p2 execute p2s turn
            start_node=np.random.choice(MCTS.tree[start_node]['child'])
        #tree search
        start_node=MCTS.tree_search(start_node)
        if MCTS.tree[start_node]['Value']==-10 or MCTS.tree[start_node]['Value']==10:
            MCTS.backpropagation(start_node)
            continue
        #expand
        MCTS.expand(start_node)
        start_node=MCTS.get_next_node(start_node)   #choose node from expansion
        #simulate
        final_node=MCTS.simulate(start_node)
        #backpropagation
        MCTS.backpropagation(final_node)
        #update wins
        if MCTS.tree[final_node]['Value']==10:
            wins.append(1)
        elif MCTS.tree[final_node]['Value']<=-10:
            wins.append(2)
        else:
            wins.append(0)
        print('EPISODE FINISHED', i)
    print(wins)
    return wins

#load q
f=open('D:\Dokumente\Studium\Master\IDA2\Project/p2r_sp1_ep100_rw10_rl10_r0.pkl','rb')
q=pickle.load(f)
f.close()                   
#init MCTS     
MCTS=MCTS(q)
N_EPISODES=79900
#learn
wins=MCTS_learning(MCTS,N_EPISODES)
wins=np.array(wins)-1
print(sum(wins)/N_EPISODES)

#save MCTS tree (only tree since MCTS contains q and q can be very big) -> must be loaded seperatly again when MCTS is loaded
f=open('D:\Dokumente\Studium\Master\IDA2\Project/MCTS_q100_sp1_ep79900_rw10_rl10_r0.pkl','wb')
pickle.dump(MCTS.tree,f)
f.close()

sys.exit()  #remove to get prints

#some prints for evaluation
for key in MCTS.tree.keys():
    print(MCTS.tree[key]['state'])
    print(MCTS.tree[key]['Value'])
    print(MCTS.tree[key]['parent'])
    print(MCTS.tree[key]['child'])
    
for key in MCTS.tree.keys():
    print(MCTS.tree[key])