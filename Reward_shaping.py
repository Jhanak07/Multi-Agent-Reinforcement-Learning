import numpy as np

class AmazonWarehouseEnv:
    def __init__(self):
        self.grid_size = 10
        self.robot_position = (0, 0)  
        self.dropoff_position = (7, 9)  
        self.intermediate_goals = [(2, 2), (4, 5), (6, 7)]  
        self.barriers = {(0, 3), (0, 7), (1, 1), (1, 5), (2, 9), (3, 2), (3, 3), (4, 2),
                         (4, 6), (5, 0), (5, 9), (6, 6), (7, 3), (7, 4), (8, 0),
                         (8, 7), (9, 9)}
        self.visited = set()  
        self.steps = 0 

    def reset(self):
        self.robot_position = (0, 0)
        self.visited = set([self.robot_position])
        self.intermediate_goals = [(2, 2), (4, 5), (6, 7)]
        self.steps = 0  
        return self.robot_position

    def step(self, action):
        self.steps += 1  
        x, y = self.robot_position
        dx, dy = [(0, 1), (0, -1), (1, 0), (-1, 0)][action]  
        new_position = (x + dx, y + dy)

        if new_position in self.barriers or not (0 <= new_position[0] < self.grid_size and 0 <= new_position[1] < self.grid_size):
            reward = -10  
            return self.robot_position, reward, False

        self.robot_position = new_position
        self.visited.add(new_position)  

       
        if new_position in self.intermediate_goals:
            reward = 50  
            self.intermediate_goals.remove(new_position)  
        else:
            reward = -1 

        
        if new_position == self.dropoff_position:
            reward += 100 
            return new_position, reward, True
        
        return new_position, reward, False


env = AmazonWarehouseEnv()
