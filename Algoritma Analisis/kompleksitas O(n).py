def cari_linear(array, target):
  """Pencarian linear dalam array"""
  for i in range(len(array)):
    if array[i] == target:
      return i
  return -1

# Contoh penggunaan:
my_array = [1, 2, 3, 4, 5]
result = cari_linear(my_array, 4)
if result:
  print("Element ada pada index ke", str(result)) 
  # Output: Element ada pada index ke-3
else:
  print("Element tidak ada dalam array")