def find_subsets(arr):
    subsets = []
    n = len(arr)

    for i in range(2**n):
        subset = []
        for j in range(n):
            if i & (1 << j):
                subset.append(arr[j])
        subsets.append(subset)

    return subsets


# Contoh
arr = [1, 2, 3]
# Output: [[], [1], [2], [1, 2], [3], [1, 3], [2, 3], [1, 2, 3]]
print(find_subsets(arr))
