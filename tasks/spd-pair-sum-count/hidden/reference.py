def pair_sum_count(values, target):
    seen = {}
    count = 0
    for value in values:
        count += seen.get(target - value, 0)
        seen[value] = seen.get(value, 0) + 1
    return count
