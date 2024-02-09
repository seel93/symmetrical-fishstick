import heapq
import itertools
from collections import deque
from bfs import bfs
from dfs import dfs
from iddfs import iddfs
from a_star import a_star_search
from cubetower import CubeTower, visualize_towers
import time


def time_and_visualize(search_method, tower, method_name):
    start_time = time.time()
    solution = search_method(tower)
    duration = time.time() - start_time
    print(f"{method_name} solution found with {len(solution)} rotations and time {duration} seconds.")

    # Visualize solution steps
    if len(solution) > 10:
        visualize_towers(solution[0])
        visualize_towers(solution[-1])
    else:
        for step in solution:
            visualize_towers(step)


def main():
    initial_configuration = ["red", "blue", "red", "green"]
    tower = CubeTower(initial_configuration)
    # # Run and visualize DFS Search
    time_and_visualize(dfs, tower, "DFS Search")
    #
    # # Run and visualize BFS Search
    time_and_visualize(bfs, tower, "BFS Search")
    #
    # Run and visualize A* Search
    time_and_visualize(a_star_search, tower, "A* Search")

    time_and_visualize(iddfs, tower, "iddfs")


if __name__ == "__main__":
    main()
