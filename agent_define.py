import numpy as np

class GridWorldEnvironment:
    def __init__(self, width, height, obstacles, goal):
        self.width = width
        self.height = height
        self.obstacles = obstacles
        self.goal = goal
        self.agent_position = (0, 0)

    def reset(self):
        self.agent_position = (0, 0)
        return self.agent_position

    def step(self, action):
        if action == 0 and self.agent_position[1] < self.height - 1:
            self.agent_position = (self.agent_position[0], self.agent_position[1] + 1)
        elif action == 1 and self.agent_position[1] > 0:
            self.agent_position = (self.agent_position[0], self.agent_position[1] - 1)
        elif action == 2 and self.agent_position[0] > 0:
            self.agent_position = (self.agent_position[0] - 1, self.agent_position[1])
        elif action == 3 and self.agent_position[0] < self.width - 1:
            self.agent_position = (self.agent_position[0] + 1, self.agent_position[1])

        reward = -1  
        done = False

        if self.agent_position in self.obstacles:
            reward = -10  
            done = True
        elif self.agent_position == self.goal:
            reward = 10  
            done = True

        return np.array(self.agent_position), reward, done, None

width = 5
height = 5
obstacles = [(1, 1), (2, 2), (3, 3)]
goal = (4, 4)

env = GridWorldEnvironment(width, height, obstacles, goal)

class RandomAgent:
    def __init__(self, action_size):
        self.action_size = action_size

    def act(self, state):
        return np.random.randint(self.action_size)


agent = RandomAgent(action_size=4) 
