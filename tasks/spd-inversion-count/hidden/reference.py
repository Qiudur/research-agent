def inversion_count(values):
    ranks = {value: index + 1 for index, value in enumerate(sorted(set(values)))}
    tree = [0] * (len(ranks) + 1)
    def prefix(index):
        total = 0
        while index:
            total += tree[index]
            index -= index & -index
        return total
    count = 0
    for seen, value in enumerate(values):
        index = ranks[value]
        count += seen - prefix(index)
        while index < len(tree):
            tree[index] += 1
            index += index & -index
    return count
