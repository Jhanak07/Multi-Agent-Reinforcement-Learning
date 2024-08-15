import random
class AmazonWarehouseEnv:
    def __init__(self):
        self.grid_size = 10
        self.robot_position = (0, 0)  
        self.pickup_position = (0, 0)  
        self.dropoff_position = (7, 9)  
        self.object_held = False
        
        self.barriers = {(0, 3), (0, 7), (1, 1), (1, 5), (2, 9), (3, 2), (3, 3), (4, 2),
                         (4, 5), (4, 6), (5, 0), (5, 9), (6, 6), (7, 3), (7, 4), (8, 0),
                         (8, 7), (9, 9)}
        self.actions = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
        self.steps = 0 
        self.total_reward = 0  

    def step(self, action):
        self.steps += 1  
        reward = -1  
       

     
        delta = self.actions[action]
        new_position = (self.robot_position[0] + delta[0], self.robot_position[1] + delta[1])

    
        if (0 <= new_position[0] < self.grid_size and
                0 <= new_position[1] < self.grid_size):
            if new_position in self.barriers:
                reward = -10  
            else:
                self.robot_position = new_position 

    
        if not self.object_held and self.robot_position == self.pickup_position:
            self.object_held = True
            reward = 10  
        elif self.object_held and self.robot_position == self.dropoff_position:
            self.object_held = False
            reward = 20  
            

        self.total_reward += reward

        print(f"Step: {self.steps}, Position: {self.robot_position}, Total Reward: {self.total_reward}")

        return self.get_state(), reward, self.object_held == False and self.robot_position == self.dropoff_position

    def get_state(self):
        return (self.robot_position, self.object_held)

    def reset(self):
        self.robot_position = (0, 0)
        self.object_held = False
        self.steps = 0  
        self.total_reward = 0 
        return self.get_state()

env = AmazonWarehouseEnv()
env.reset()
done = False
while not done:
    action = random.choice(list(env.actions.keys()))  
    state, reward, done = env.step(action)
