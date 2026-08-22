import os
import pickle
import random
import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    """
    # We check if we are in training mode OR if the save file is missing
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up Q-table from scratch.")
        self.q_table = {} 
    else:
        self.logger.info("Loading Q-table from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.q_table = pickle.load(file)



def get_path_direction(game_state):
    # Breadth-First Search (BFS) pathfinding to coins

    agent_info = game_state['self']
    agent_x = agent_info[3][0]
    agent_y = agent_info[3][1]
    
    coins = game_state['coins']
    arena = game_state['field']
    
    if len(coins) == 0:
        return 'WAIT'

    queue = []
    

    start_node = (agent_x, agent_y, [])
    queue.append(start_node)
    
    visited = []
    visited.append((agent_x, agent_y))
    

    while len(queue) > 0:
        current_node = queue.pop(0)
        
        current_x = current_node[0]
        current_y = current_node[1]
        current_path = current_node[2]

        for coin in coins:
            if current_x == coin[0] and current_y == coin[1]:
                if len(current_path) == 0:
                    return 'WAIT'
                return current_path[0]

        moves = [
            (0, -1, 'UP'),
            (0, 1, 'DOWN'),
            (-1, 0, 'LEFT'),
            (1, 0, 'RIGHT')
        ]
        
        for move in moves:
            next_x = current_x + move[0]
            next_y = current_y + move[1]
            direction_name = move[2]

            if arena[next_x][next_y] == 0:
                

                if (next_x, next_y) not in visited:
                    visited.append((next_x, next_y))

                    new_path = current_path.copy()
                    new_path.append(direction_name)
                    

                    queue.append((next_x, next_y, new_path))
                    
    # If all coins are blocked by walls and we can't reach them
    return 'WAIT'
     

def state_to_features(game_state):
    if game_state is None: 
        return None
    
    agent_x, agent_y = game_state['self'][3]
    arena = game_state['field']
    
    coin_dir = get_path_direction(game_state)
    
    wall_up = arena[agent_x, agent_y - 1] != 0
    wall_down = arena[agent_x, agent_y + 1] != 0
    wall_left = arena[agent_x - 1, agent_y] != 0
    wall_right = arena[agent_x + 1, agent_y] != 0 

    return (coin_dir, wall_up, wall_down, wall_left, wall_right)
    
def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """ 
    # IF training: 
    if self.train and hasattr(self, 'next_action') and self.next_action is not None:
        action = self.next_action
        self.next_action = None # Clear it
        return action
        
    # IF not training: 
    state = state_to_features(game_state)
    
    if state not in self.q_table:
        return np.random.choice(ACTIONS)
    
    actions = list(self.q_table[state].keys())
    q_values = list(self.q_table[state].values())
    return actions[np.argmax(q_values)]
    

