from bisect import bisect_left, bisect_right

def range_counts(values, queries):
    ordered = sorted(values)
    return [bisect_right(ordered, right) - bisect_left(ordered, left)
            if left <= right else 0 for left, right in queries]
