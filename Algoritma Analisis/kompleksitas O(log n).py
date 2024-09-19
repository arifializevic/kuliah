def cari_biner(array, target):
    """Pencarian biner dalam array yang terurut"""
    low = 0
    high = len(array) - 1
    while low <= high:
        mid = (low + high) // 2
        if array[mid] == target:
            return mid
        elif array[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1


# Contoh penggunaan:
my_array = [1, 2, 3, 4, 5]
result = cari_biner(my_array, 4)
if result:
    print("Element ada pada index ke", str(result))
    # Output: Element ada pada index ke 3
else:
    print("Element tidak ada dalam array")
