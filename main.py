import heapq
import itertools
from collections import deque
from bfs import bfs
from dfs import dfs
from iddfs import iddfs
from a_star import a_star_search
from cubetower import CubeTower

# Example usage could be similar for each algorithm, just calling the desired method on the CubeTower instance.
# Example of usage:
initial_configuration = ['red', 'blue', 'green', 'yellow']  # Example configuration
cube_tower = CubeTower(initial_configuration)
solution_path = a_star_search(cube_tower)
if solution_path:
    print("Solution found:")
    for step in solution_path:
        print(step)
else:
    print("No solution found.")