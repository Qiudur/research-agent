import heapq

def running_median(values):
    low, high, result = [], [], []
    for value in values:
        if not low or value <= -low[0]:
            heapq.heappush(low, -value)
        else:
            heapq.heappush(high, value)
        if len(low) > len(high) + 1:
            heapq.heappush(high, -heapq.heappop(low))
        elif len(high) > len(low):
            heapq.heappush(low, -heapq.heappop(high))
        result.append((-low[0] + high[0]) / 2.0 if len(low) == len(high)
                      else float(-low[0]))
    return result
