from oblig1.bfs import bfs
from oblig1.dfs import dfs
from oblig1.iddfs import iddfs
from oblig1.a_star import a_star_search
from oblig1.cubetower import CubeTower, visualize_towers
import time
import matplotlib.pyplot as plt
import numpy as np  # For generating a range of colors
import psutil
import random


def generate_random_colors():
    colors = ['red', 'blue', 'green', 'yellow']
    random_colors = random.choices(colors, k=4)
    return random_colors


def plot_results(results, graph_label, result_column):
    """
    Plots the execution times of search algorithms with distinct colors for each algorithm.
    """
    plt.figure(figsize=(10, 6))
    algorithms = list(results.keys())
    times = list(results.values())
    colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))  # Using tab10 colormap
    plt.bar(algorithms, times, color=colors)
    plt.xlabel('Search Algorithm')
    plt.ylabel(result_column)
    plt.title(graph_label)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def time_and_visualize_algorithm(search_method, tower, visualize_solution):
    """
    Times a search method and optionally visualizes the solution.
    """
    memory_init = psutil.virtual_memory().used
    start_time = time.time()
    solution = search_method(tower)
    duration = (time.time() - start_time) * 1000  # Convert to milliseconds
    duration = round(duration, 3)
    memory_end = psutil.virtual_memory().used - memory_init

    if visualize_solution:
        visualize_towers(solution[0])
        visualize_towers(solution[-1])

    return duration, len(solution) if solution else 0, memory_end


def main():
    tower = CubeTower(generate_random_colors())
    alg_performance = {}
    alg_rotations = {}
    alg_memory = {}

    print(f"Initial tower configuration: {tower.configuration}")

    # Define algorithms and whether their solutions should be visualized
    algorithms = [
        ("BFS Search", bfs, False),
        ("DFS Search", dfs, False),
        ("A* Search", a_star_search, False),
        ("IDDFS", iddfs, False),
    ]

    # Time and optionally visualize each algorithm
    for name, algorithm, visualize in algorithms:
        duration, steps, memory = time_and_visualize_algorithm(algorithm, tower,  visualize)
        print(f"{name} solution found with {steps} rotations in {duration} milliseconds and memory: {memory}")

        alg_performance[name] = duration
        alg_rotations[name] = steps
        alg_memory[name] = memory

    # Plot performance with distinct colors
    plot_results(alg_performance,  "Search performance", "Execution Time (ms)")
    plot_results(alg_rotations,  "Cube rotations", "Rotations")
    plot_results(alg_memory, "Memory usage", "Memory Usage (bytes)")


if __name__ == "__main__":
    main()
