import numpy as np
import random

class WarehouseEnv:
    def __init__(self):
        self.grid_size = 10
        self.states = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        self.actions = ['up', 'down', 'left', 'right']
        self.state = (0, 0)  
        self.dropoff_position = (7, 9)  
        self.barriers = {(0, 3), (0, 7), (1, 1), (1, 5), (2, 9), (3, 2), (3, 3), (4, 2),
                         (4, 6), (5, 0), (5, 9), (6, 6), (7, 3), (7, 4), (8, 0),
                         (8, 7), (9, 9)}

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 'up' and y > 0:
            next_state = (x, y - 1)
        elif action == 'down' and y < self.grid_size - 1:
            next_state = (x, y + 1)
        elif action == 'left' and x > 0:
            next_state = (x - 1, y)
        elif action == 'right' and x < self.grid_size - 1:
            next_state = (x + 1, y)
        else:
            next_state = (x, y)  
        
        reward = -1
        if next_state in self.barriers:
            next_state = (x, y)  
            reward = -10
        if next_state == self.dropoff_position:
            reward = 100  

        self.state = next_state
        return next_state, reward, next_state == self.dropoff_position

    def is_terminal(self, state):
        return state == self.dropoff_position

def choose_action(state, q_values, epsilon=0.1):
    if random.random() < epsilon:
      return random.choice(['up', 'down', 'left', 'right'])
    else:
        return max(q_values[state], key=q_values[state].get)

def sarsa(env, episodes, alpha=0.1, gamma=0.9):
    q_values = {state: {action: 0 for action in env.actions} for state in env.states}
    steps_per_episode = []  
    
    for _ in range(episodes):
        state = env.reset()
        action = choose_action(state, q_values)
        steps = 0  
        
        while not env.is_terminal(state):
            next_state, reward, done = env.step(action)
            next_action = choose_action(next_state, q_values)
            
            
            q_values[state][action] += alpha * (reward + gamma * q_values[next_state][next_action] - q_values[state][action])
            
            state, action = next_state, next_action
            steps += 1  
            if done:
                break

        steps_per_episode.append(steps)  
    return q_values, steps_per_episode


env = WarehouseEnv()
q_values, steps_per_episode = sarsa(env, 1000)
print("Training complete.")
print("Steps taken per episode:", steps_per_episode)

