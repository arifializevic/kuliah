class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None


# Binary Tree Traversal Functions
def in_order_traversal(root):
    return in_order_traversal(root.left) + [root.data] + in_order_traversal(root.right) if root else []


def post_order_traversal(root):
    return post_order_traversal(root.left) + post_order_traversal(root.right) + [root.data] if root else []


def level_order_traversal(root):
    if not root:
        return []
    queue = [root]
    result = []
    while queue:
        node = queue.pop(0)
        result.append(node.data)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return result


# Contoh Pohon
root = Node('A')
root.left = Node('B')
root.right = Node('C')
root.left.left = Node('D')
root.left.right = Node('E')
root.right.left = Node('F')
root.right.right = Node('G')

print("In-order:", in_order_traversal(root))
print("Post-order:", post_order_traversal(root))
print("Level-order:", level_order_traversal(root))
