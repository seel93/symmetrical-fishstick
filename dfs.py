
def dfs(cube, visited=None):
    if visited is None:
        visited = set()

    if str(cube.configuration) in visited:
        return None
    if cube.is_goal_state():
        return cube.trace_path()

    visited.add(str(cube.configuration))
    for move in cube.generate_moves('dfs'):
        result = dfs(move, visited)
        if result:
            return result
    return None
