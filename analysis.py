import numpy as np
import matplotlib.pyplot as plt

# Your data
steps = np.array([718, 133, 263, 108, 688, 451, 1382, 357, 586, 237, 1487, 155, 442, 435, 1455, 509, 133, 2773, 220, 1054, 128, 485, 732, 1885, 519, 219, 138, 1269, 96, 278])
total_rewards = np.array([-1991, -317, -735, -265, -1763, -995, -3654, -928, -1337, -610, -3795, -285, -1148, -916, -3394, -1206, -344, -6737, -566, -2651, -321, -1272, -1960, -4895, -1180, -696, -259, -3316, -262, -705])

# Calculate basic statistics
mean_steps = np.mean(steps)
median_steps = np.median(steps)
std_steps = np.std(steps)

mean_rewards = np.mean(total_rewards)
median_rewards = np.median(total_rewards)
std_rewards = np.std(total_rewards)

print(f"Mean Steps: {mean_steps}, Median Steps: {median_steps}, Std Dev Steps: {std_steps}")
print(f"Mean Rewards: {mean_rewards}, Median Rewards: {median_rewards}, Std Dev Rewards: {std_rewards}")

# Plotting histograms
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.hist(steps, bins=10, color='blue', edgecolor='black')
plt.title('Distribution of Steps')
plt.xlabel('Steps')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(total_rewards, bins=10, color='red', edgecolor='black')
plt.title('Distribution of Total Rewards')
plt.xlabel('Total Rewards')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
