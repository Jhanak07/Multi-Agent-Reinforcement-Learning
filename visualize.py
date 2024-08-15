import matplotlib.pyplot as plt

def visualize_training(env, agent, episodes=1000, batch_size=32):
    rewards = []

    plt.figure(figsize=(10, 5))

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        rewards.append(total_reward)

        agent.replay(batch_size)

        if episode % 10 == 0:
            plt.plot(rewards, color='blue')
            plt.title('Training Performance')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.grid(True)
            plt.show()

    plt.plot(rewards, color='blue')
    plt.title('Training Performance')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.show()

# Visualize training process
visualize_training(env, agent)
