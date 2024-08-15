import numpy as np
import random

class AmazonWarehouseEnv:
    def __init__(self):
        self.grid_size = 10
        self.reset()

    def reset(self):
        self.robot_position = (0, 0)  # Start at the top-left corner
        self.dropoff_position = (7, 9)  # Target position
        self.steps = 0  # Step counter
        self.barriers = {(0, 3), (0, 7), (1, 1), (1, 5), (2, 9), (3, 2), (3, 3), (4, 2),
                         (4, 5), (4, 6), (5, 0), (5, 9), (6, 6), (7, 3), (7, 4), (8, 0),
                         (8, 7), (9, 9)}
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Actions: Up, Down, Left, Right

        return self.robot_position

    def step(self, action):
        x, y = self.robot_position
        if action == 0 and y > 0: y -= 1  # Move up
        if action == 1 and y < self.grid_size - 1: y += 1  # Move down
        if action == 2 and x > 0: x -= 1  # Move left
        if action == 3 and x < self.grid_size - 1: x += 1  # Move right
        self.robot_position = (x, y)
        self.steps += 1

        # Check for barriers
        if self.robot_position in self.barriers:
            reward = -10
        elif self.robot_position == self.dropoff_position:
            reward = 100  # Large reward for reaching the goal
            return self.robot_position, reward, True
        else:
            reward = -1  # Small penalty for each move

        # Adding a condition to stop after many steps
        done = self.steps >= 100
        return self.robot_position, reward, done

def q_learning(env, episodes, learning_rate, discount_factor, epsilon):
    q_table = np.zeros((env.grid_size, env.grid_size, 4))  # Initialize Q-table

    for episode in range(episodes):
        state = env.reset()  # Reset environment at the start of each episode
        total_reward = 0
        done = False
        steps = 0  # Reset steps counter at the start of each episode

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, 3)  # Explore with a random action
            else:
                action = np.argmax(q_table[state[0], state[1]])  # Exploit by choosing the best known action

            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1  # Increment step count

            # Update Q-table using the Q-learning formula
            old_value = q_table[state[0], state[1], action]
            future_optimal_value = np.max(q_table[next_state[0], next_state[1]])
            new_value = old_value + learning_rate * (reward + discount_factor * future_optimal_value - old_value)
            q_table[state[0], state[1], action] = new_value

            state = next_state  # Move to the next state

        # Print total reward and steps at the end of each episode
        print(f"Episode {episode + 1}: Steps = {steps}, Total Reward = {total_reward}")

# Parameters
episodes = 1000
learning_rate = 0.1
discount_factor = 0.99
epsilon = 0.1

# Create and train the environment
env = AmazonWarehouseEnv()
q_learning(env, episodes, learning_rate, discount_factor, epsilon)
