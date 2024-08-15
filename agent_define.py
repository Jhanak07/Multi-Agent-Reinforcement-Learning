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
        # Define actions: 0=up, 1=down, 2=left, 3=right
        if action == 0 and self.agent_position[1] < self.height - 1:
            self.agent_position = (self.agent_position[0], self.agent_position[1] + 1)
        elif action == 1 and self.agent_position[1] > 0:
            self.agent_position = (self.agent_position[0], self.agent_position[1] - 1)
        elif action == 2 and self.agent_position[0] > 0:
            self.agent_position = (self.agent_position[0] - 1, self.agent_position[1])
        elif action == 3 and self.agent_position[0] < self.width - 1:
            self.agent_position = (self.agent_position[0] + 1, self.agent_position[1])

        reward = -1  # Default reward for moving
        done = False

        if self.agent_position in self.obstacles:
            reward = -10  # Penalty for hitting an obstacle
            done = True
        elif self.agent_position == self.goal:
            reward = 10  # Reward for reaching the goal
            done = True

        return np.array(self.agent_position), reward, done, None

# Define environment parameters
width = 5
height = 5
obstacles = [(1, 1), (2, 2), (3, 3)]
goal = (4, 4)

# Create environment
env = GridWorldEnvironment(width, height, obstacles, goal)

# Define agent
class RandomAgent:
    def __init__(self, action_size):
        self.action_size = action_size

    def act(self, state):
        return np.random.randint(self.action_size)

# Create agent
agent = RandomAgent(action_size=4)  # Assuming 4 actions: up, down, left, right
