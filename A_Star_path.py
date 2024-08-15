import matplotlib.pyplot as plt
import numpy as np

grid_size = 10
barriers = {(0, 3), (0, 7), (1, 1), (1, 5), (2, 9), (3, 2), (3, 3), (4, 2),
            (4, 6), (5, 0), (5, 9), (6, 6), (7, 3), (7, 4), (8, 0),
            (8, 7), (9, 9)}
path = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (3, 8), 
        (4, 8), (5, 8), (6, 8), (6, 9), (7, 9)]

grid = np.zeros((grid_size, grid_size))

for barrier in barriers:
    grid[barrier] = -1 

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(grid, cmap='gray')

start = path[0]
goal = path[-1]
ax.plot(start[1], start[0], 'go', markersize=10, label="Start")  
ax.plot(goal[1], goal[0], 'ro', markersize=10, label="Goal")  
for p in path:
    ax.plot(p[1], p[0], 'bo', markersize=5)  

ax.set_xticks(np.arange(-.5, 10, 1))
ax.set_yticks(np.arange(-.5, 10, 1))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(True)

ax.legend()

plt.title('A* Path Visualization in Amazon Warehouse Grid')
plt.gca().invert_yaxis()  
plt.show()
