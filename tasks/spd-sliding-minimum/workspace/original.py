def sliding_minimum(values, window):
    if window < 1:
        raise ValueError("window must be positive")
    return [min(values[i:i + window]) for i in range(len(values) - window + 1)]
