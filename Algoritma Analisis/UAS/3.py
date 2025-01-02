from collections import Counter
import heapq


class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def huffman_tree(frequencies):
    heap = [Node(char, freq) for char, freq in frequencies.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]


def huffman_codes(node, prefix="", code={}):
    if node:
        if node.char is not None:
            code[node.char] = prefix
        huffman_codes(node.left, prefix + "0", code)
        huffman_codes(node.right, prefix + "1", code)
    return code


# Input data
text = "PERANCANGAN ANALISIS ALGORITMA UNTUK MAHASISWA UNPAM"
frequencies = Counter(text.replace(" ", ""))
tree = huffman_tree(frequencies)
codes = huffman_codes(tree)
total_bits = sum(frequencies[char] * len(codes[char]) for char in frequencies)

print("Huffman Codes:", codes)
print("Total bits required:", total_bits)
