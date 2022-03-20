# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 11:57:30 2022

@author: tobia
"""
import numpy as np
import sys
sys.path.append(".")
from connect4 import connect4

class QValues(object):
    
    def __init__(self):
        '''Initialize with empty lookup table.'''
        self.values = {}    #dictionary with states as keys
        self.wins=np.array([[0,0]])
        
    def get_value(self, state, action):
        '''Return stored q value for (state, action) pair or a random number if unknown.'''
        #make state and action hashable  
        state_hash=tuple(state.flatten())
        action_hash=tuple(action.flatten())
        
        if not state_hash in self.values:
            self.values[state_hash] = {}
        if not action_hash in self.values[state_hash]:
            self.values[state_hash][action_hash] = abs(np.random.randn()) + 1
        return self.values[state_hash][action_hash]
    
    def set_value(self, state, action, value):
        '''Stored q value for (state, action) pair.'''
        #make state and action hashable  
        state_hash=tuple(state.flatten())
        action_hash=tuple(action)
        
        if not state_hash in self.values:   #if state not yet in table create entry of state
            self.values[state_hash] = {}
        if not action_hash in self.values[state_hash]:  #if action not already in table for this state create enty for action
            self.values[state_hash][action_hash] = 0
        
        self.values[state_hash][action_hash] = value    #set the value
    
    def max_action(self, state, actions, learning=True):
        '''Return the action with highest q value for given state and action list.'''
        if not learning:
            state_hash=tuple(state.flatten())
            if not state_hash in self.values:   #if state not in table perform a random action
                return connect4.random_action(actions) #if actions else None
        
        max_value = -np.inf
        max_action = actions[0] #set default action to first action in case actions not in table
        for action in actions:  #check all actions to find best
            if not learning:    #if action not in table skip
                action_hash=tuple(action)
                if not action_hash in self.values[state_hash]:
                    continue

            value = self.get_value(state, action)   
            if value > max_value:   #if value bigger update value and action
                max_value = value
                max_action = action
            elif value == max_value and learning:
                choice = np.random.choice([0,1])
                if choice==1:
                    max_action=action
        return max_action
    
    def epsilon_greedy(self, state, actions, epsilon):
        '''Returns max_action or random action with probability of epsilon.'''
        if np.random.rand() < epsilon:        
            return connect4.random_action(actions)
        return self.max_action(state, actions)
    
    def __str__(self):
        #nr_states = len(self.values.keys())
        output='\n'
        for key in self.values.keys():
            key_arr=np.array(key)
            key_arr=np.reshape(key_arr,[6,7])
            output=output+str(np.flip(key_arr,0))+'\n'
            output=output+str(self.values[key])+'\n\n\n'
        return output
        #return 'Number of states: {}'.format(nr_states)  