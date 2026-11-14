from collections import deque

def sliding_minimum(values, window):
    if window < 1:
        raise ValueError("window must be positive")
    indices = deque()
    result = []
    for index, value in enumerate(values):
        while indices and indices[0] <= index - window:
            indices.popleft()
        while indices and values[indices[-1]] >= value:
            indices.pop()
        indices.append(index)
        if index + 1 >= window:
            result.append(values[indices[0]])
    return result
