import matplotlib.pyplot as plt
import numpy as np


class CubeTower:
    def __init__(self, configuration, parent=None, depth=0, g=0, h=0):
        self.order = ['red', 'blue', 'green', 'yellow']  # Define the color order
        self.configuration = configuration  # The current configuration of the cube tower
        self.parent = parent  # The parent configuration from which this configuration was derived
        self.depth = depth  # Depth for IDDFS and DLS
        self.g = g  # Cost from start node to current node (used in A*)
        self.h = h  # Heuristic cost from current node to goal (used in A*)
        self.f = g + h  # Total cost (used in A*)

    def color_to_num(self, color):
        """Map color to a numerical value based on its order."""
        return self.order.index(color)

    def is_goal_state(self):
        # Check if all cubes are aligned correctly
        return all(color == self.configuration[0] for color in self.configuration)

    def generate_moves(self):
        moves = []
        # Including the first rotation option for completeness
        for i in range(len(self.configuration)):
            new_config = self.configuration[:]
            for j in range(i, len(self.configuration)):
                new_config[j] = self.rotate_color(new_config[j])
            moves.append(
                CubeTower(new_config, self, self.depth + 1, self.g + 1, self.calculate_heuristic(new_config))
            )

        if self.parent:
            # Option 2: Rotate a block of cubes, keeping the cube above it steady
            for i in range(len(self.configuration) - 1):  # No need to rotate the last cube with this option
                for k in range(i + 2,
                               len(self.configuration) + 1):  # Ensure there's at least one cube above the block to keep steady
                    new_config = self.configuration[:]
                    for j in range(i, k - 1):  # Rotate all cubes in the block from i to k-2 (since k is exclusive)
                        new_config[j] = self.rotate_color(new_config[j])
                    moves.append(
                        CubeTower(
                            new_config, self, self.depth + 1, self.g + 1, self.calculate_heuristic(new_config)
                        )
                    )

        return moves

    def rotate_color(self, color):
        # Rotate the color to the next one in the order

        index = self.order.index(color)
        new_color = self.order[(index + 1) % 4]
        return new_color

    @staticmethod
    def calculate_heuristic(configuration):
        # Example heuristic function: Count of cubes not matching the first cube's color
        return sum(1 for color in configuration if color != configuration[0])

    def calculate_euclidean_distance(self, other):
        """Calculate the Euclidean distance based on color differences between two configurations."""
        sum_of_squares = 0
        for c1, c2 in zip(self.configuration, other):
            diff = self.color_to_num(c1) - self.color_to_num(c2)
            sum_of_squares += diff ** 2
        return sum_of_squares ** 0.5

    def calculate_manhattan_distance(self, other):
        """Calculate the Manhattan distance based on color differences between two configurations."""
        total_distance = 0
        for c1, c2 in zip(self.configuration, other):
            diff = abs(self.color_to_num(c1) - self.color_to_num(c2))
            total_distance += diff
        return total_distance

    def trace_path(self):
        # Utility method to trace back the path from the current configuration to the root
        path = []
        current = self
        while current:
            path.append(current.configuration)
            current = current.parent
        return path[::-1]


def visualize_towers(color_list):
    """
    Visualizes a list of tower colors as a horizontal bar chart.
    :param color_list: A list of color strings.
    """
    # Setup for visualization
    fig, ax = plt.subplots()
    y_pos = np.arange(len(color_list))
    performance = np.ones(len(color_list))
    # Create horizontal bars with colors based on the input list
    ax.barh(y_pos, performance, color=color_list, edgecolor='black')
    # Labeling the y-axis with the position of each cube in the tower
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'Cube {i + 1}' for i in range(len(color_list))])
    # Invert the y-axis to show the bottom cube at the bottom
    ax.invert_yaxis()
    # Set labels and title
    ax.set_xlabel('Cube Position')
    ax.set_title('Cube Tower Configuration')
    # Display the plot
    plt.show()
