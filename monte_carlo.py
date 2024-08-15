import random

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_actions = self.get_actions()
        self.steps = 0 if not parent else parent.steps + 1 

    def get_actions(self):
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        random.shuffle(actions)
        return actions

    def expand(self, barriers):
        while self.untried_actions:
            action = self.untried_actions.pop()
            next_position = (self.position[0] + action[0], self.position[1] + action[1])
            
            if 0 <= next_position[0] < 10 and 0 <= next_position[1] < 10 and next_position not in barriers:
                child_node = Node(next_position, self)
                self.children.append(child_node)
                return child_node
        return None

def simulate(env, node):
    current_node = node
    steps = 0
    while not current_node.is_terminal(env.dropoff_position) and steps < 100:  
        possible_moves = env.get_possible_moves(current_node.position)
        if not possible_moves:
            break  
        next_move = random.choice(possible_moves)
        current_node = Node(next_move, current_node)
        steps += 1
    return 1 if current_node.position == env.dropoff_position else 0

class WarehouseEnv:
    def __init__(self):
        self.dropoff_position = (7, 9)
        self.barriers = {(0, 3), (0, 7), (1, 1), (1, 5), (2, 9), (3, 2), (3, 3), (4, 2),
                         (4, 5), (4, 6), (5, 0), (5, 9), (6, 6), (7, 3), (7, 4), (8, 0),
                         (8, 7), (9, 9)}
    
    def get_possible_moves(self, position):
        x, y = position
        moves = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 10 and 0 <= ny < 10 and (nx, ny) not in self.barriers:
                moves.append((nx, ny))
        return moves


env = WarehouseEnv()
root = Node((0, 0))
best_child = mcts_search(env, root, 1000)
path, steps = output_path(best_child)
print("Path from start to goal:", path)
print("Steps taken:", steps)
