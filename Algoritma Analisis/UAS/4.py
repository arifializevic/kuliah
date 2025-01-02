import heapq

# Define the puzzle dimensions
N = 3

# Define the goal state
GOAL_STATE = [
    [1, 2, 3],
    [8, 0, 4],
    [7, 6, 5]
]


# Initial puzzle state
INITIAL_STATE = [
    [2, 8, 3],
    [1, 6, 4],
    [7, 0, 5]
]

# Define possible moves for the blank (0)
MOVES = {
    'UP': (-1, 0),
    'DOWN': (1, 0),
    'LEFT': (0, -1),
    'RIGHT': (0, 1)
}

# Helper function to calculate Manhattan distance


def calculate_heuristic(state):
    distance = 0
    for i in range(N):
        for j in range(N):
            if state[i][j] != 0:
                value = state[i][j]
                goal_x, goal_y = divmod(value - 1, N)
                distance += abs(goal_x - i) + abs(goal_y - j)
    return distance

# Find the position of the blank (0) in the state


def find_blank(state):
    for i in range(N):
        for j in range(N):
            if state[i][j] == 0:
                return i, j

# Generate a new state by moving the blank


def generate_new_state(state, move, blank_pos):
    x, y = blank_pos
    dx, dy = MOVES[move]
    new_x, new_y = x + dx, y + dy

    # Ensure the move is within bounds
    if 0 <= new_x < N and 0 <= new_y < N:
        # Create a copy of the state
        new_state = [row[:] for row in state]
        # Swap the blank with the target position
        new_state[x][y], new_state[new_x][new_y] = new_state[new_x][new_y], new_state[x][y]
        return new_state
    return None

# Best-First Search algorithm


def best_first_search(initial_state):
    # Priority queue to store the states based on heuristic value
    open_list = []
    heapq.heappush(open_list, (calculate_heuristic(
        initial_state), initial_state, []))

    # Set to store visited states
    visited = set()

    while open_list:
        # Pop the state with the lowest heuristic value
        _, current_state, path = heapq.heappop(open_list)

        # Convert the state to a tuple for hashing
        state_tuple = tuple(tuple(row) for row in current_state)

        # If the state is the goal, return the path
        if current_state == GOAL_STATE:
            return path

        # Skip if the state has already been visited
        if state_tuple in visited:
            continue

        visited.add(state_tuple)

        # Find the position of the blank
        blank_pos = find_blank(current_state)

        # Generate new states for all possible moves
        for move in MOVES:
            new_state = generate_new_state(current_state, move, blank_pos)
            if new_state:
                heapq.heappush(open_list, (calculate_heuristic(
                    new_state), new_state, path + [move]))

    return None


# Solve the puzzle
solution = best_first_search(INITIAL_STATE)

if solution:
    print("Solution found:")
    print(" -> ".join(solution))
else:
    print("No solution found.")
