

def dls(cube, limit):
    if cube.depth > limit:
        return 'cutoff'
    if cube.is_goal_state():
        return cube.trace_path()
    if cube.depth == limit:
        return 'cutoff'
    cutoff_occurred = False
    for move in cube.generate_moves('IDDFS'):
        result = dls(move, limit)
        if result == 'cutoff':
            cutoff_occurred = True
        elif result is not None:
            return result
    return 'cutoff' if cutoff_occurred else None


def iddfs(cube, max_depth=100):
    depth = 0
    while depth < max_depth:
        result = dls(cube, depth)
        if result != 'cutoff':
            return result
        depth += 1
