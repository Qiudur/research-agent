def compute_signal(history):
    raw = {}
    for s, p in history.items():
        if len(p) >= 6:
            raw[s] = p[-1] / p[-5] - 1.0
        else:
            raw[s] = 0.0
    vals = list(raw.values())
    if len(vals) == 0:
        return {}
    avg = sum(vals) / len(vals)
    return {k: v - avg for k, v in raw.items()}
