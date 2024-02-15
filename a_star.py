import heapq
import itertools


def a_star_search(start_cube):
    open_list = []
    counter = itertools.count()
    heapq.heappush(open_list, (start_cube.f, next(counter), start_cube))
    visited = set()

    while open_list:
        current_f, _, current = heapq.heappop(open_list)
        if str(current.configuration) in visited:
            continue
        if current.is_goal_state():
            return current.trace_path()

        visited.add(str(current.configuration))
        for move in current.generate_moves('A*'):
            if str(move.configuration) not in visited:
                heapq.heappush(open_list, (move.f, next(counter), move))
    return None
