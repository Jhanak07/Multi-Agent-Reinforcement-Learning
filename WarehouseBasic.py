import numpy as np 
import random

class AmazonWarehouseEnv:
    def __init__(self):
        self.grid_size = 10
        self.robot_position = (0, 0)  
        self.pickup_position = (random.randint(0, 9), random.randint(0, 9))
        self.dropoff_position = (random.randint(0, 9), random.randint(0, 9))
        self.object_held = False
        self.barriers = self.generate_barriers()
        self.actions = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
        self.state = (self.robot_position, self.object_held)
        
    def generate_barriers(self):
        barriers = set()
        for _ in range(20):  
            barriers.add((random.randint(0, 9), random.randint(0, 9)))
       
        barriers.discard(self.pickup_position)
        barriers.discard(self.dropoff_position)
        return barriers

    def step(self, action):
       
        delta = self.actions[action]
        new_position = (self.robot_position[0] + delta[0], self.robot_position[1] + delta[1])

     
        if (0 <= new_position[0] < self.grid_size and
            0 <= new_position[1] < self.grid_size and
            new_position not in self.barriers):
            self.robot_position = new_position

        reward = -1  
        done = False
        
        
        if not self.object_held and self.robot_position == self.pickup_position:
            self.object_held = True
            reward = 10 
        elif self.object_held and self.robot_position == self.dropoff_position:
            self.object_held = False
            reward = 20  
            done = True  

        return self.get_state(), reward, done

    def get_state(self):
        return self.robot_position, self.object_held

    def reset(self):
        self.robot_position = (0, 0)
        self.object_held = False
        self.barriers = self.generate_barriers()
        return self.get_state()

env = AmazonWarehouseEnv()
print(env.reset())
