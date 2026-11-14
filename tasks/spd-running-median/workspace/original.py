def running_median(values):
    result = []
    for end in range(1, len(values) + 1):
        ordered = sorted(values[:end])
        middle = end // 2
        result.append(float(ordered[middle]) if end % 2 else
                      (ordered[middle - 1] + ordered[middle]) / 2.0)
    return result
