# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 15:48:46 2022

@author: tobia
"""
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
import math


class MCTS:
    def __init__(self,q,epsilon=0.05,gamma=0.9,alpha=0.1,UCB_factor=1):
        #q =default q learned strategy
        #epsilon for epsilon greedy of q
        #backpropagation: (1-alpha)*V_i+alpha*(r_i+gamma*V_{i+1})
        self.final_action = None
        self.root_node_p1 = 1
        self.root_node_p2= 2
        state2=connect4()
        state2.reset(2)
        #two seperate start nodes for tree for start of p1 (algorithm) or start of p2 (opponent) -> agent is always marked as 1
        self.tree = {self.root_node_p1:{'state':connect4().board, 'turn':1, 'child':[], 'parent': None,'Value':0,'Visits':0,'expanded':False}
                     ,self.root_node_p2:{'state':state2.board, 'turn':2, 'child':[], 'parent': None,'Value':0,'Visits':0,'expanded':False}}

        self.q=q
        self.max_id=3   #tree starts with nodes 1 and 2 -> id given to added new node to tree
        self.epsilon=epsilon
        self.gamma=gamma    #gamme for value update (scaling of rewards through episode length)
        self.alpha=alpha    #alpha for value update (old vs new value)
        self.UCB_factor=UCB_factor  #factor to control exploration part of UCB
        #add nodes for player 2 start (all possible first moves of p2)
        for i in range(state2.COLUMN_COUNT):
            state=connect4()
            state.reset(2)
            state.execute([0,i])
            self.add_node(2, state.board, 1, 1)
        
    def backpropagation(self,node_id):
        parent_id=self.tree[node_id]['parent']
        while parent_id is not None:  #go back the whole path
            V_old=self.tree[parent_id]['Value']
            self.tree[parent_id]['Value']=(1-self.alpha)*V_old+self.alpha*self.gamma*self.tree[node_id]['Value']    #value is set to reward for last node
            self.tree[parent_id]['Visits']+=1
            node_id=parent_id
            parent_id=self.tree[node_id]['parent']
    
      
    def get_ucb(self,node_id):
        #returns upeer confidence bound of node
        parent_id=self.tree[node_id]['parent']
        if self.tree[node_id]['Visits']==0:
            return float('inf')
        exploration=math.sqrt(2*math.log(self.tree[parent_id]['Visits'])/self.tree[node_id]['Visits'])
        ucb=self.tree[node_id]['Value']+self.UCB_factor*exploration
        return ucb

        
    def remove_existing_actions(self,node_id,actions):
        #used in expansion so only nodes that not already exist are created new
        removed=0
        new_actions=np.array(actions)
        for i in range(len(actions)):   #remove all actions that lead to a board thats already in the tree
            test_state=connect4()
            test_state.board=np.array(self.tree[node_id]['state'])
            test_state.execute(actions[i],1)

            for child in self.tree[node_id]['child']:
                if (test_state.board==self.tree[child]['state']).all():
                    print('if activated')
                    new_actions=np.delete(new_actions,i-removed,axis=0)
                    removed+=1
                    break;
        return new_actions
    
    def add_node(self,parent_id,state,turn,value):
        #adds node
        self.tree[self.max_id]={'state':state,'turn':turn,'child':[],'parent':parent_id,'Value':value,'expanded':False,'Visits':0}
        self.tree[parent_id]['child'].append(self.max_id)
        self.max_id+=1
        
    def expand(self,start_id):
        #expands the tree with parent=start_id
        node_id=start_id
        player=1  #since expansion is only done when its the agents turn player is set 1      
        state=connect4()    
        state.board=np.array(self.tree[node_id]['state'])   #create new board that is same as old board
        actions=state.get_actions()   
        old_state=connect4()
        old_state.board=np.array(self.tree[node_id]['state'])
        #check if valid actions exist         
        if np.all(actions==-1):
            if not state.winning_move(1) and not state.winning_move(2):
                self.tree[node_id]['Value']=0   #if board not won the board is full (reward=0)
                return
        #remove actions for which there is already a node
        actions=self.remove_existing_actions(node_id,actions)
        for i in range(len(actions)):
            #create new state and get actions
            state=connect4()    
            state.board=np.array(self.tree[node_id]['state'])
            _,reward=state.execute(actions[i],turn=player)
            #if winning move update tree and end loop to start backpropagation
            if state.winning_move(player):
                self.add_node(node_id,state.board,2,reward)               
                break;
            #if no winning move update tree and let player 2 play
            else:   
                self.add_node(node_id,state.board,2,self.q.get_value(old_state.board,np.array(actions[i])))
        self.tree[start_id]['expanded']=True  
        return 
                
    def simulate(self,start_id):
        #simulation after expansion -> stat_id is leaf node
        node_id=start_id
        while True: 
            ########### PLAYER 2 ################
            #since simulation is used after expansion and expansion is only done for p1 turns it is always p2 turn at start of simulation
            state=connect4()    #create new state 
            state.board=np.array(self.tree[node_id]['state'])
            old_state=connect4()
            old_state.board=np.array(self.tree[node_id]['state'])
            state.turn=2
            actions=state.get_actions() #get actions
            if np.all(actions==-1): #if no action exists simulation is stopped
                break;
            action=connect4.random_action(actions)      #choose random action
            _,reward=state.execute(action,2)    #execute on state
            #if winning move update tree and break
            if state.winning_move(2):
                self.add_node(node_id,state.board,1,-10)    #value set to -10 (reward for loss)
                node_id=self.max_id-1
                break;
            else:   #if no winning move update tree and continue
                self.add_node(node_id,state.board,1,self.q.get_value(old_state.board,np.array(action))) #initialize with value of old board and executed action
                node_id=self.max_id-1

                
            ####################PLAYER 1##############
            #get epsilon_greedy action of q learned strategy and execute
            actions=state.get_actions()
            if np.all(actions==-1):
                break;
            action=self.q.epsilon_greedy(state.board,actions,self.epsilon)
            old_state=connect4()
            old_state.board=np.array(state.board)
            state.reset()
            state.board=np.array(self.tree[node_id]['state'])
            _,reward=state.execute(action,turn=1)
            #if winning move update tree and end loop to start backpropagation
            if state.winning_move(1):              
                self.add_node(node_id,state.board,2,reward) #add new node with reward as value
                node_id=self.max_id-1
                break;
            #if no winning move update tree and let player 2 play
            else:                               
                self.add_node(node_id,state.board,2,self.q.get_value(old_state.board,np.array(action))) #initialize with value of old board and executed action
                node_id=self.max_id-1
        #backpropagation of values
        return node_id
        
    def get_next_node(self,node_id):
        #gets best node according to ucb (used during learning)
        max_ucb=float('-inf')
        best_node=0
        for i in self.tree[node_id]['child']:
            ucb=self.get_ucb(i)
            if ucb>max_ucb:
                max_ucb=ucb
                best_node=i
        return best_node

    def get_best_node(self,node_id):
        #gets best node according to highest value (used during testing)
        max_value=float('-inf')
        best_node=0
        if self.tree[node_id]['child']==[]:
            return -1
        for i in self.tree[node_id]['child']:
            if self.tree[i]['Value']>max_value:
                max_value=self.tree[i]['Value']
                best_node=i
        return best_node

    def tree_search(self,start_id):
        #plays along the tree starting from start id and returns first unexpanded node of p1
        node_id=start_id
        while True:
            exists=False    #used to check if the p2 node already exists
            self.tree[node_id]['Visits']+=1 #update visits of node
            if self.tree[node_id]['turn']==1:   #if player 1 turn
                if self.tree[node_id]['expanded']:  #if node expanded choose next node according to ucb
                    next_node=self.get_next_node(node_id)
                    node_id=next_node
                else:   #if not expanded return node_id which will then be expanded
                    return node_id
            else:
                ########### PLAYER 2 ################
                state=connect4()
                state.board=np.array(self.tree[node_id]['state'])
                old_state=connect4()
                old_state.board=np.array(self.tree[node_id]['state'])
                state.turn=2
                actions=state.get_actions()
                if np.all(actions==-1):
                    return node_id
                action=connect4.random_action(actions)
                _,reward=state.execute(action,2)
                
                #check if node already exists -> if exists just update node_id and visists
                for child in self.tree[node_id]['child']:
                    if (state.board==self.tree[child]['state']).all():
                        node_id=child
                        self.tree[child]['Visits']+=1
                        exists=True
                        break;
                
                #if winning move add new node to tree and break
                if state.winning_move(2):
                    if not exists:
                        self.add_node(node_id,state.board,1,-10)
                        node_id=self.max_id-1
                    return node_id
                elif not exists:   #if no winning move add new node to tree and continue
                    self.add_node(node_id,state.board,1,self.q.get_value(old_state.board,np.array(action)))
                    node_id=self.max_id-1

            
            
                
                

    
        
    
            
        


