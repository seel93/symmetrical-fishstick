import heapq
from collections import deque
import itertools
from bfs import bfs
from dfs import dfs


class CubeTower:
    def __init__(self, configuration, parent=None, depth=0, g=0, h=0):
        self.order = ['red', 'blue', 'green', 'yellow']  # Define the color order
        self.configuration = configuration  # The current configuration of the cube tower
        self.parent = parent  # The parent configuration from which this configuration was derived
        self.depth = depth  # Depth for IDDFS and DLS
        self.g = g  # Cost from start node to current node (used in A*)
        self.h = h  # Heuristic cost from current node to goal (used in A*)
        self.f = g + h  # Total cost (used in A*)

    def is_goal_state(self):
        # Check if all cubes are aligned correctly
        return all(color == self.configuration[0] for color in self.configuration)

    def generate_moves(self):
        # Generate all possible moves from the current configuration
        moves = []
        for i in range(len(self.configuration)):
            for _ in range(3):  # Each cube can be rotated three times
                new_config = self.configuration[:]
                new_config[i] = self.rotate_color(new_config[i])
                new_g = self.g + 1
                new_h = self.calculate_heuristic(new_config)
                moves.append(CubeTower(new_config, self, self.depth + 1, new_g, new_h))
        return moves

    def rotate_color(self, color):
        # Rotate the color to the next one in the order
        index = self.order.index(color)
        return self.order[(index + 1) % 4]

    @staticmethod
    def calculate_heuristic(configuration):
        # Example heuristic function: Count of cubes not matching the first cube's color
        return sum(1 for color in configuration if color != configuration[0])

    def a_star_search(self):
        open_list = []
        counter = itertools.count()  # Create a unique sequence counter
        heapq.heappush(open_list, (self.f, next(counter), self))  # Use counter for tie-breaking
        visited = set()

        while open_list:
            current_f, _, current = heapq.heappop(open_list)
            if str(current.configuration) in visited:
                continue
            if current.is_goal_state():
                return current.trace_path()

            visited.add(str(current.configuration))
            for move in current.generate_moves():
                if str(move.configuration) not in visited:
                    heapq.heappush(open_list, (move.f, next(counter), move))
        return None

    def iddfs(self):
        # Iterative Deepening Depth-First Search implementation
        depth = 0
        while True:
            result = self.dls(depth)
            if result != 'cutoff':
                return result
            depth += 1

    def dls(self, limit):
        # Depth-Limited Search to support IDDFS
        if self.depth > limit:
            return 'cutoff'
        if self.is_goal_state():
            return self.trace_path()
        if self.depth == limit:
            return 'cutoff'
        cutoff_occurred = False
        for move in self.generate_moves():
            result = move.dls(limit)
            if result == 'cutoff':
                cutoff_occurred = True
            elif result is not None:
                return result
        return 'cutoff' if cutoff_occurred else None

    def trace_path(self):
        # Utility method to trace back the path from the current configuration to the root
        path = []
        current = self
        while current:
            path.append(current.configuration)
            current = current.parent
        return path[::-1]


# Example usage could be similar for each algorithm, just calling the desired method on the CubeTower instance.
# Example of usage:
initial_configuration = ['red', 'blue', 'green', 'yellow']  # Example configuration
cube_tower = CubeTower(initial_configuration)
solution_path = dfs(cube_tower)
if solution_path:
    print("Solution found:")
    for step in solution_path:
        print(step)
else:
    print("No solution found.")