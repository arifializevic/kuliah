from itertools import permutations


def find_permutations(arr):
    return list(permutations(arr))


# Contoh
arr = [1, 2, 3]
print(find_permutations(arr))
# Output: [(1, 2, 3), (1, 3, 2), (2, 1, 3),
# (2, 3, 1), (3, 1, 2), (3, 2, 1)]
