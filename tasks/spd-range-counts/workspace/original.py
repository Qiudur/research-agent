def range_counts(values, queries):
    return [sum(1 for value in values if left <= value <= right)
            for left, right in queries]
