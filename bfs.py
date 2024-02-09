from collections import deque


def bfs(start_cube):
    visited = set()
    queue = deque([start_cube])

    while queue:
        current = queue.popleft()
        if str(current.configuration) in visited:
            continue
        if current.is_goal_state():
            return current.trace_path()

        visited.add(str(current.configuration))
        for move in current.generate_moves():
            if str(move.configuration) not in visited:
                queue.append(move)
    return None
