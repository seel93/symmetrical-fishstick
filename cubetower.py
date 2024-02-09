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

    def trace_path(self):
        # Utility method to trace back the path from the current configuration to the root
        path = []
        current = self
        while current:
            path.append(current.configuration)
            current = current.parent
        return path[::-1]
