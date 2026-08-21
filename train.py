from collections import namedtuple, deque

import pickle
from typing import List
import numpy as np

import events as e
from .callbacks import state_to_features, ACTIONS

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 100,
        e.KILLED_SELF: -500,
        e.WAITED: -1, 
        e.INVALID_ACTION: -5 
    }
    
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def get_action_for_state(self, state):
    
    if state not in self.q_table:
        self.q_table[state] = {a: 0.0 for a in ACTIONS}
        
    # Explore
    if np.random.rand() < self.epsilon:
        return np.random.choice(ACTIONS)
    
    # Exploit
    return max(self.q_table[state], key=self.q_table[state].get)

def setup_training(self):

    self.alpha = 0.1      # Learning rate
    self.gamma = 0.9      # Discount factor
    self.epsilon = 0.1    # Exploration rate      
    self.epsilon_min = 0.0      
    self.epsilon_decay = 0.998
    
    self.next_action = None
    
    self.stats_rewards = []
    self.stats_steps = []
    
   
# Master trackers for all rounds
    self.stats_rewards = []
    self.stats_steps = []
    self.stats_events = [] 

    self.current_reward = 0
    self.current_steps = 0
    
    self.current_events = {
        e.COIN_COLLECTED: 0,
        e.INVALID_ACTION: 0,
        e.WAITED: 0
    }


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    
    S = state_to_features(old_game_state)
    A = self_action
    R = reward_from_events(self, events)
    
    # REWARD SHAPING 
    if S is not None:
        compass_direction = S[0] # This pulls 'UP', 'DOWN', 'LEFT', or 'RIGHT' from your features
        
        if compass_direction != 'WAIT':
            if A == compass_direction:
                R += 2  # followed the BFS compass.
            elif A != 'BOMB': 
                R -= 1  # moved the wrong way.
    
    self.current_reward += R
    self.current_steps += 1
    

    for event in events:
        if event in self.current_events:
            self.current_events[event] += 1

    S_prime = state_to_features(new_game_state)
    A_prime = get_action_for_state(self, S_prime)
    self.next_action = A_prime 

    if S not in self.q_table: self.q_table[S] = {a: 0.0 for a in ACTIONS}
    if S_prime not in self.q_table: self.q_table[S_prime] = {a: 0.0 for a in ACTIONS}
        
    old_q_value = self.q_table[S][A]
    next_q_value = self.q_table[S_prime][A_prime]
    
    new_q_value = old_q_value + self.alpha * (R + self.gamma * next_q_value - old_q_value)
    self.q_table[S][A] = new_q_value


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # FINAL SARSA MATH 
    S = state_to_features(last_game_state)
    A = last_action
    R = reward_from_events(self, events)
    
    if S not in self.q_table:
        self.q_table[S] = {a: 0.0 for a in ACTIONS}

    old_q_value = self.q_table[S][A]
    # next_q_value is 0 because the game is over, there is no future state
    new_q_value = old_q_value + self.alpha * (R + (self.gamma * 0.0) - old_q_value)
    self.q_table[S][A] = new_q_value

    self.current_reward += R
    self.current_steps += 1


    # reduce epsilon
    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay

    for event in events:
        if event in self.current_events:
            self.current_events[event] += 1
    
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.q_table, file)

    self.stats_rewards.append(self.current_reward)
    self.stats_steps.append(self.current_steps)
    self.stats_events.append(self.current_events.copy())
    
    # Reset for next round 
    self.current_reward = 0
    self.current_steps = 0
    self.current_events = {e.COIN_COLLECTED: 0, e.INVALID_ACTION: 0, e.WAITED: 0}


    with open("training_stats.pkl", "wb") as file:
        pickle.dump({
            'rewards': self.stats_rewards, 
            'steps': self.stats_steps,
            'events': self.stats_events
        }, file)