import pygame
import time
import random
import heapq
import math
import copy
import numpy as np
import sys
import csv

pygame.init()
alpha = 0.009
idle_time = 0
time_step=0
moves = 0
# Number of previous visited cells that we store
track_len = 15
# Variables to store cell and beep value
track_path = []
track_beep = []
aliens = []
desired_nodes = []
open_pairs = []
pair_probs = []
old_prob_dist = []
k = 1
fps = 60
grid_node_width = 25
grid_node_height = 25
size = 30
crew_alien_grid_size = 3
alien_threshold = 0
grid_distances = {}
simulate = True
prob_mat = [[1 for i in range(size)] for j in range(size)]
WIDTH = grid_node_width * size
HEIGHT = grid_node_height * size
screen = pygame.display.set_mode([WIDTH, HEIGHT])
pygame.display.set_caption('Bot 5 Algo')
img = pygame.image.load('bot.jpg')
img = pygame.transform.scale(img, (grid_node_height - 5, grid_node_width - 5))
img_1 = pygame.image.load('Deoxys.png')
img_1 = pygame.transform.scale(img_1, (grid_node_height - 5, grid_node_width - 5))
img_2 = pygame.image.load('crew.png')
img_2 = pygame.transform.scale(img_2, (grid_node_height - 5, grid_node_width - 5))


# creating a closed grid of given dimensions
grid = np.array([[1 for i in range(size)] for j in range(size)])


# class based representation of a single cell in the graph
class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __repr__(self):
        return f"{self.position} - g: {self.g} h: {self.h} f: {self.f}"

    # defining less than for purposes of heap queue
    def __lt__(self, other):
        return self.f < other.f

    # defining greater than for purposes of heap queue
    def __gt__(self, other):
        return self.f > other.f


# function to create squares while displaying the grid
def createSquare(x, y, color):
    pygame.draw.rect(screen, color, [x, y, grid_node_width, grid_node_height])


# Function to open the grid and create a new configuration
def open_grid(start_x, start_y):
    grid[start_x][start_y] = 0
    while True:
        temp = []
        for i in range(size):
            for j in range(size):
                if grid[i][j] == 1:
                    neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                    count = 0
                    for nx, ny in neighbors:
                        if (nx > -1 and nx < size) and (ny > -1 and ny < size):
                            if grid[nx][ny] == 0:
                                count = count + 1
                if count == 1:
                    temp.append((i, j))
        if len(temp) == 0:
            break
        # randomly picking a cell to be opened from the list of eligible cells
        new_x, new_y = random.choice(temp)
        grid[new_x][new_y] = 0


# Function to create and return the traversed path
def return_path(current_node):
    path = []
    current = current_node
    while current is not None:
        path.append(current.position)
        current = current.parent
    return path[::-1]  # Return reversed path


# The A* path finding algorithm function
def astar(grid, start, end, aliens):
    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Heapify the open_list and Add the start node
    heapq.heapify(open_list)
    heapq.heappush(open_list, start_node)

    # what squares do we search
    adjacent_squares = ((0, -1), (0, 1), (-1, 0), (1, 0),)

    # Loop until you find the end
    while len(open_list) > 0:
        # Get the current node
        current_node = heapq.heappop(open_list)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            return return_path(current_node)

        # Generate children
        children = []

        for new_position in adjacent_squares:  # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(grid) - 1) or node_position[0] < 0 or node_position[1] > (
                    len(grid[len(grid) - 1]) - 1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if grid[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:
            # Child is on the closed list
            if len([closed_child for closed_child in closed_list if closed_child == child]) > 0:
                continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = abs(child.position[0] - end_node.position[0]) + abs(child.position[1] - end_node.position[1])
            child.f = child.g + child.h

            # Child is already in the open list
            if len([open_node for open_node in open_list if
                    child.position == open_node.position and child.g > open_node.g]) > 0:
                continue

            # Add the child to the open list
            heapq.heappush(open_list, child)

    return None


# Function to display the current grid configurationx
def visualizeGrid(bot_pos, aliens):
    y = 0  # we start at the top of the screen
    for row in grid:
        x = 0  # for every row we start at the left of the screen again
        for item in row:
            if item == 0:
                createSquare(x, y, (255, 255, 255))
            else:
                createSquare(x, y, (0, 0, 0))

            x += grid_node_width  # for ever item/number in that row we move one "step" to the right
        y += grid_node_height  # for every new row we move one "step" downwards
    screen.blit(img, (bot_pos[1] * grid_node_width, bot_pos[0] * grid_node_height))
    for i in range(len(desired_nodes)):
        screen.blit(img_2, (desired_nodes[i][1] * grid_node_width, desired_nodes[i][0] * grid_node_height))
    for x, y in aliens:
        screen.blit(img_1, (y * grid_node_width, x * grid_node_height))
    pygame.display.update()


# Function to determine where the aliens would be in the next time step
def new_alien_pos(aliens):
    # choices that the alien can make
    move_set = ((0, -1), (0, 1), (-1, 0), (1, 0), (0, 0))
    new_pos = []
    i = 0
    while len(new_pos) < k:
        current_node = aliens[i]
        new_position = random.choice(move_set)
        # Get node position
        node_position = (current_node[0] + new_position[0], current_node[1] + new_position[1])

        # Make sure within range
        if node_position[0] > (len(grid) - 1) or node_position[0] < 0 or node_position[1] > (
                len(grid[len(grid) - 1]) - 1) or node_position[1] < 0:
            continue

        # Make sure walkable terrain
        if grid[node_position[0]][node_position[1]] != 0:
            continue

        # An alien should not collide with another alien
        if node_position in new_pos:
            continue

        new_pos.append(node_position)
        i += 1
    return new_pos


def rand_cords():
    return (random.randint(0, size - 1), random.randint(0, size - 1))


def beep_prob(alpha, d1, d2 = 0):
    p1 = math.exp(-alpha * (d1 - 1))
    p2 = 0
    if d2 != 0:
        p2 = math.exp(-alpha * (d2 - 1))
    prob = p1 + p2 - p1*p2
    return random.choices([1, 0], weights=[prob, 1 - prob])


def is_valid(node):
    if node[0] < 0 or node[0] >= size:
        return False
    if node[1] < 0 or node[1] >= size:
        return False
    if grid[node[0]][node[1]] == 1:
        return False
    return True

# Function used to calulcated distance of all other open cells in the grid from a given cell
def calc_distances(start_cell):
    queue = []
    visited = []
    queue.append((start_cell, 0))
    grid_distances[start_cell] = {}
    while (len(queue) > 0):
        node, dist = queue.pop(0)
        grid_distances[start_cell][node] = dist
        visited.append(node)

        if is_valid((node[0] - 1, node[1])) and (node[0] - 1, node[1]) not in visited:
            queue.append(((node[0] - 1, node[1]), dist + 1))
        if is_valid((node[0] + 1, node[1])) and (node[0] + 1, node[1]) not in visited:
            queue.append(((node[0] + 1, node[1]), dist + 1))
        if is_valid((node[0], node[1] - 1)) and (node[0], node[1] - 1) not in visited:
            queue.append(((node[0], node[1] - 1), dist + 1))
        if is_valid((node[0], node[1] + 1)) and (node[0], node[1] + 1) not in visited:
            queue.append(((node[0], node[1] + 1), dist + 1))

            
# Function used for probability updation when a beep is heard
# This function is used when only one crew member is present on the grid
def update_probs_1(start_cell, prob_mat):
    total = 0
    for i in range(len(prob_mat)):
        for j in range(len(prob_mat[0])):
            if prob_mat[i][j] > -1:
                prob_mat[i][j] = prob_mat[i][j]*(math.exp(-alpha * grid_distances[start_cell][(i, j)]))
                total += (math.exp(-alpha * grid_distances[start_cell][(i, j)]))

    for i in range(size):
        for j in range(size):
            if prob_mat[i][j] > -1:
                prob_mat[i][j] = prob_mat[i][j] / total
    return prob_mat

# Function used for probability updation when a beep is heard
# This function is used when two crew members are present in the grid
def update_probs(start_cell, pair_probs):
    total = 0
    for i in open_pairs:
        if pair_probs[i[0][0]][i[0][1]][i[1][0]][i[1][1]] > 0:
            d1 = grid_distances[start_cell][i[0]]
            d2 = grid_distances[start_cell][i[1]]
            p1 = math.exp(-alpha * d1)
            p2 = math.exp(-alpha * d2)
            prob = p1 + p2 - (p1 * p2)
            pair_probs[i[0][0]][i[0][1]][i[1][0]][i[1][1]] = pair_probs[i[0][0]][i[0][1]][i[1][0]][i[1][1]] * prob
            total += prob
    pair_probs = pair_probs / total
    return pair_probs

# Function used for probability updation when no beep is heard
# This function is used when one crew member is present in the grid
def update_no_probs_1(start_cell, prob_mat):
    total = 0
    for i in range(size):
        for j in range(size):
            if prob_mat[i][j] > -1:
                prob_mat[i][j] = (1 - math.exp(-alpha * grid_distances[start_cell][(i, j)]))
    for i in range(size):
        for j in range(size):
            if prob_mat[i][j] > -1:
                prob_mat[i][j] = prob_mat[i][j] / total   
    return prob_mat

# Function used for probability updation when no beep is heard
# This function is used when two crew members are present in the grid
def update_no_probs(start_cell, pair_probs):
    total = 0
    for i in open_pairs:
        if pair_probs[i[0][0]][i[0][1]][i[1][0]][i[1][1]] > 0:
            d1 = grid_distances[start_cell][i[0]]
            d2 = grid_distances[start_cell][i[1]]
            p1 = math.exp(-alpha * d1)
            p2 = math.exp(-alpha * d2)
            prob = p1 + p2 - (p1 * p2)
            not_prob = 1 - prob
            pair_probs[i[0][0]][i[0][1]][i[1][0]][i[1][1]] = pair_probs[i[0][0]][i[0][1]][i[1][0]][i[1][1]] * not_prob
            total += not_prob 
    pair_probs = pair_probs / total
    return pair_probs

# Function to update the alien probability distribution once the alien moves
def update_crew_alien_grid_AlienMoved(grid, crew_alien_grid, x, y, prev, crew_found_inside_detection_grid, blocked_nodes,
                                 movement):
    copy_crew_alien_grid = copy.deepcopy(crew_alien_grid)
    crew_alien_grid = [[0 for i in range(size)] for j in range(size)]

    blocked_nodes_inside_crew_alien = []
    opened_nodes_inside_crew_alien = []
    summation = 0

    if prev is None:
        if crew_found_inside_detection_grid == True:
            oper = "stepOneInside"
        if crew_found_inside_detection_grid == False:
            oper = "stepOneOutside"

    elif crew_found_inside_detection_grid == True:
        if prev == True:
            oper = "insideUpdate"
        if prev == False:
            oper = "insideBorder"

    elif crew_found_inside_detection_grid == False:
        if prev == True:
            oper = "outsideBorder"
        if prev == False:
            oper = "outsideUpdate"

    x_start = x - (crew_alien_grid_size // 2)
    x_end = x - (crew_alien_grid_size // 2) + crew_alien_grid_size - 1
    y_start = y - (crew_alien_grid_size // 2)
    y_end = y - (crew_alien_grid_size // 2) + crew_alien_grid_size - 1

    for i in range(x_start, x_end + 1):
        for j in range(y_start, y_end + 1):
            if i < 0 or j < 0 or i >= size or j >= size:
                continue

            if (i, j) in blocked_nodes:
                blocked_nodes_inside_crew_alien.append((i, j))
            else:
                opened_nodes_inside_crew_alien.append((i, j))

            if oper == "stepOneInside":

                if (i, j) not in blocked_nodes and (i, j) is not (x, y):
                    crew_alien_grid[i][j] += 1
                    summation += crew_alien_grid[i][j]

            if oper == "insideBorder":
                if (i, j) in blocked_nodes:
                    continue

                if i == x_start and (i - 1, j) not in blocked_nodes and i > 0:
                    summation += copy_crew_alien_grid[i - 1][j]
                    crew_alien_grid[i][j] += copy_crew_alien_grid[i - 1][j]

                if j == y_start and (i, j - 1) not in blocked_nodes and j > 0:
                    summation += copy_crew_alien_grid[i][j - 1]
                    crew_alien_grid[i][j] += copy_crew_alien_grid[i][j - 1]

                if i == x_end and (i + 1, j) not in blocked_nodes and i + 1 <= size - 1:
                    summation += copy_crew_alien_grid[i + 1][j]
                    crew_alien_grid[i][j] += copy_crew_alien_grid[i + 1][j]

                if j == y_end and (i, j + 1) not in blocked_nodes and j + 1 <= size - 1:
                    summation += copy_crew_alien_grid[i][j + 1]
                    crew_alien_grid[i][j] += copy_crew_alien_grid[i][j + 1]

                if movement == (1, 0) and i == x_end and j in range(y_start, y_end + 1):
                    possible_moves = getPossibleMoves(i, j, blocked_nodes)
                    moveInsideBoundary = []
                    summation += copy_crew_alien_grid[i][j]
                    for possible_move in possible_moves:
                        if possible_move[0] < x_start or possible_move[1] < y_start or possible_move[0] > x_end or \
                                possible_move[1] > y_end:
                            continue

                        moveInsideBoundary.append((possible_move))

                    for mov in moveInsideBoundary:
                        crew_alien_grid[mov[0]][mov[1]] += (copy_crew_alien_grid[i][j] / len(moveInsideBoundary))

                if movement == (-1, 0) and i == x_start and j in range(y_start, y_end + 1):
                    possible_moves = getPossibleMoves(i, j, blocked_nodes)
                    moveInsideBoundary = []
                    summation += copy_crew_alien_grid[i][j]
                    for possible_move in possible_moves:
                        if possible_move[0] < x_start or possible_move[1] < y_start or possible_move[0] > x_end or \
                                possible_move[1] > y_end:
                            continue

                        moveInsideBoundary.append((possible_move))

                    for mov in moveInsideBoundary:
                        crew_alien_grid[mov[0]][mov[1]] += (copy_crew_alien_grid[i][j] / len(moveInsideBoundary))

                if movement == (0, 1) and j == y_end and i in range(x_start, x_end + 1):
                    possible_moves = getPossibleMoves(i, j, blocked_nodes)
                    moveInsideBoundary = []
                    summation += copy_crew_alien_grid[i][j]
                    for possible_move in possible_moves:
                        if possible_move[0] < x_start or possible_move[1] < y_start or possible_move[0] > x_end or \
                                possible_move[1] > y_end:
                            continue

                        moveInsideBoundary.append((possible_move))

                    for mov in moveInsideBoundary:
                        crew_alien_grid[mov[0]][mov[1]] += (copy_crew_alien_grid[i][j] / len(moveInsideBoundary))

                if movement == (0, -1) and j == y_start and i in range(x_start, x_end + 1):
                    possible_moves = getPossibleMoves(i, j, blocked_nodes)
                    moveInsideBoundary = []
                    summation += copy_crew_alien_grid[i][j]
                    for possible_move in possible_moves:
                        if possible_move[0] < x_start or possible_move[1] < y_start or possible_move[0] > x_end or \
                                possible_move[1] > y_end:
                            continue

                        moveInsideBoundary.append((possible_move))

                    for mov in moveInsideBoundary:
                        crew_alien_grid[mov[0]][mov[1]] += (copy_crew_alien_grid[i][j] / len(moveInsideBoundary))



            elif oper == "outsideBorder":
                if (i, j) in blocked_nodes:
                    continue

                elif i == x_start and j == y_start:
                    possible_moves = []
                    if (i - 1, j) not in blocked_nodes and i - 1 >= 0:
                        possible_moves.append((i - 1, j))

                    if (i, j - 1) not in blocked_nodes and j - 1 >= 0:
                        possible_moves.append((i, j - 1))

                    if len(possible_moves) == 0:
                        continue
                    summation += copy_crew_alien_grid[i][j]
                    for possible_move in possible_moves:
                        crew_alien_grid[possible_move[0]][possible_move[1]] += copy_crew_alien_grid[i][j] / len(
                            possible_moves)

                elif i == x_start and j == y_end:
                    possible_moves = []
                    if (i - 1, j) not in blocked_nodes and i - 1 >= 0:
                        possible_moves.append((i - 1, j))

                    if (i, j + 1) not in blocked_nodes and j + 1 <= size - 1:
                        possible_moves.append((i, j + 1))

                    if len(possible_moves) == 0:
                        continue
                    summation += copy_crew_alien_grid[i][j]
                    for possible_move in possible_moves:
                        crew_alien_grid[possible_move[0]][possible_move[1]] += copy_crew_alien_grid[i][j] / len(
                            possible_moves)

                elif i == x_end and j == y_start:
                    possible_moves = []
                    if (i + 1, j) not in blocked_nodes and i + 1 <= size - 1:
                        possible_moves.append((i + 1, j))

                    if (i, j - 1) not in blocked_nodes and j - 1 >= 0:
                        possible_moves.append((i, j - 1))

                    if len(possible_moves) == 0:
                        continue
                    summation += copy_crew_alien_grid[i][j]
                    for possible_move in possible_moves:
                        crew_alien_grid[possible_move[0]][possible_move[1]] += copy_crew_alien_grid[i][j] / len(
                            possible_moves)

                elif i == x_start and j == y_start:
                    possible_moves = []
                    if (i + 1, j) not in blocked_nodes and i + 1 <= size - 1:
                        possible_moves.append((i + 1, j))

                    if (i, j + 1) not in blocked_nodes and j + 1 <= size - 1:
                        possible_moves.append((i, j + 1))

                    if len(possible_moves) == 0:
                        continue
                    summation += copy_crew_alien_grid[i][j]
                    for possible_move in possible_moves:
                        crew_alien_grid[possible_move[0]][possible_move[1]] += copy_crew_alien_grid[i][j] / len(
                            possible_moves)


                elif i == x_start and (i - 1, j) not in blocked_nodes and i > 0:
                    crew_alien_grid[i - 1][j] += copy_crew_alien_grid[i][j]
                    summation += copy_crew_alien_grid[i][j]

                elif j == y_start and (i, j - 1) not in blocked_nodes and j > 0:
                    crew_alien_grid[i][j - 1] += copy_crew_alien_grid[i][j]
                    summation += copy_crew_alien_grid[i][j]

                elif i == x_end and (i + 1, j) not in blocked_nodes and i != size - 1:
                    crew_alien_grid[i + 1][j] += copy_crew_alien_grid[i][j]
                    summation += copy_crew_alien_grid[i][j]

                elif j == y_end and (i, j + 1) not in blocked_nodes and j != size - 1:
                    crew_alien_grid[i][j + 1] += copy_crew_alien_grid[i][j]
                    summation += copy_crew_alien_grid[i][j]


            elif oper == "insideUpdate":
                # how many possible values
                if (i, j) not in blocked_nodes and (x, y) is not (i, j):
                    move_set = ((0, -1), (0, 1), (-1, 0), (1, 0), (0, 0))
                    possible_movements = []
                    summation += copy_crew_alien_grid[i][j]
                    for move in move_set:
                        adj_i = i + move[0]
                        adj_j = j + move[1]
                        if adj_i < 0 or adj_j < 0 or adj_i >= size or adj_j >= size or (adj_i, adj_j) is (x, y):
                            continue

                        if (adj_i, adj_j) not in blocked_nodes and not (
                                adj_i < x_start or adj_j < y_start or adj_i > x_end or adj_j > y_end):
                            possible_movements.append((adj_i, adj_j))

                    for possible_movement in possible_movements:
                        crew_alien_grid[possible_movement[0]][possible_movement[1]] += copy_crew_alien_grid[i][j] / len(
                            possible_movements)

    if oper == "outsideBorder":

        if movement == (1, 0):
            if x_start - 1 >= 0:
                for j in range(y_start, y_end + 1):
                    if j >= 0 and j <= size - 1:
                        possible_moves = getPossibleMoves(x_start - 1, j,
                                                          blocked_nodes + opened_nodes_inside_crew_alien)
                        summation += copy_crew_alien_grid[x_start - 1][j]
                        for poss_move in possible_moves:
                            crew_alien_grid[poss_move[0]][poss_move[1]] += (
                                        copy_crew_alien_grid[x_start - 1][j] / len(possible_moves))

        if movement == (-1, 0):
            if x_end + 1 <= size - 1:
                for j in range(y_start, y_end + 1):
                    if j >= 0 and j <= size - 1:
                        possible_moves = getPossibleMoves(x_end + 1, j, blocked_nodes + opened_nodes_inside_crew_alien)
                        summation += copy_crew_alien_grid[x_end + 1][j]
                        for poss_move in possible_moves:
                            crew_alien_grid[poss_move[0]][poss_move[1]] += (
                                        copy_crew_alien_grid[x_end + 1][j] / len(possible_moves))

        if movement == (0, 1):
            if y_start - 1 >= 0:
                for i in range(x_start, x_end + 1):
                    if i >= 0 and i <= size - 1:
                        possible_moves = getPossibleMoves(i, y_start - 1,
                                                          blocked_nodes + opened_nodes_inside_crew_alien)
                        summation += copy_crew_alien_grid[i][y_start - 1]
                        for poss_move in possible_moves:
                            crew_alien_grid[poss_move[0]][poss_move[1]] += (
                                        copy_crew_alien_grid[i][y_start - 1] / len(possible_moves))

        if movement == (0, -1):
            if y_end + 1 <= size - 1:
                for i in range(x_start, x_end + 1):
                    if i >= 0 and i <= size - 1:
                        possible_moves = getPossibleMoves(i, y_end + 1, blocked_nodes + opened_nodes_inside_crew_alien)
                        summation += copy_crew_alien_grid[i][y_end + 1]
                        for poss_move in possible_moves:
                            crew_alien_grid[poss_move[0]][poss_move[1]] += (
                                        copy_crew_alien_grid[i][y_end + 1] / len(possible_moves))

    if oper == "outsideUpdate":

        for i in range(size):
            for j in range(size):
                if (i, j) not in blocked_nodes:
                    move_set = ((0, -1), (0, 1), (-1, 0), (1, 0), (0, 0))
                    possible_movements = []
                    summation += copy_crew_alien_grid[i][j]
                    for move in move_set:
                        adj_i = i + move[0]
                        adj_j = j + move[1]
                        if adj_i < 0 or adj_j < 0 or adj_i >= size or adj_j >= size:
                            continue

                        if (adj_i, adj_j) not in blocked_nodes and (adj_i, adj_j) not in opened_nodes_inside_crew_alien:
                            possible_movements.append((adj_i, adj_j))

                    for possible_movement in possible_movements:
                        crew_alien_grid[possible_movement[0]][possible_movement[1]] += copy_crew_alien_grid[i][j] / len(
                            possible_movements)

    if oper == "insideUpdate":
        if movement == (1, 0):
            if x_start - 1 >= 0:
                for j in range(y_start, y_end + 1):
                    if j >= 0 and j <= size - 1:
                        summation += copy_crew_alien_grid[x_start - 1][j]
                        crew_alien_grid[x_start][j] += copy_crew_alien_grid[x_start - 1][j]

        if movement == (-1, 0):
            if x_end + 1 <= size - 1:
                for j in range(y_start, y_end + 1):
                    if j >= 0 and j <= size - 1:
                        crew_alien_grid[x_end][j] += copy_crew_alien_grid[x_end + 1][j]
                        summation += copy_crew_alien_grid[x_end + 1][j]

        if movement == (0, 1):
            if y_start - 1 >= 0:
                for i in range(x_start, x_end + 1):
                    if i >= 0 and i <= size - 1:
                        crew_alien_grid[i][y_start] += copy_crew_alien_grid[i][y_start - 1]
                        summation += copy_crew_alien_grid[i][y_start - 1]

        if movement == (0, -1):
            if y_end + 1 <= size - 1:
                for i in range(x_start, x_end + 1):
                    if i >= 0 and i <= size - 1:
                        crew_alien_grid[i][y_end] += copy_crew_alien_grid[i][y_end + 1]
                        summation += copy_crew_alien_grid[i][y_end + 1]

    if oper == "stepOneOutside":
        for i in range(size):
            for j in range(size):
                if (i, j) not in blocked_nodes and (i, j) not in opened_nodes_inside_crew_alien:
                    crew_alien_grid[i][j] = 1
                    summation += crew_alien_grid[i][j]
                    ##Probalizing the values
    for i in range(size):
        for j in range(size):
            crew_alien_grid[i][j] = crew_alien_grid[i][j] / summation
    return crew_alien_grid

# Function the alien probability distribution once the bot moves
def update_crew_alien_grid_BotMoved(grid, crew_alien_grid, x, y, prev, crew_found_inside_detection_grid, blocked_nodes,
                                 movement):
    copy_crew_alien_grid = copy.deepcopy(crew_alien_grid)
    crew_alien_grid = [[0 for i in range(size)] for j in range(size)]

    blocked_nodes_inside_crew_alien = []
    opened_nodes_inside_crew_alien = []
    summation = 0

    if prev is None:
        if crew_found_inside_detection_grid == True:
            oper = "stepOneInside"
        if crew_found_inside_detection_grid == False:
            oper = "stepOneOutside"

    elif crew_found_inside_detection_grid == True:
        oper="inside"

    elif crew_found_inside_detection_grid == False:
        oper="outside"

    x_start = x - (crew_alien_grid_size // 2)
    x_end = x - (crew_alien_grid_size // 2) + crew_alien_grid_size - 1
    y_start = y - (crew_alien_grid_size // 2)
    y_end = y - (crew_alien_grid_size // 2) + crew_alien_grid_size - 1

    for i in range(x_start, x_end + 1):
        for j in range(y_start, y_end + 1):
            if i < 0 or j < 0 or i >= size or j >= size:
                continue

            if (i, j) in blocked_nodes:
                blocked_nodes_inside_crew_alien.append((i, j))
            else:
                opened_nodes_inside_crew_alien.append((i, j))

            if oper == "stepOneInside":

                if (i, j) not in blocked_nodes and (i, j) is not (x, y):
                    crew_alien_grid[i][j] += 1
                    summation += crew_alien_grid[i][j]

            elif oper == "inside"  :
                # how many possible values
                if (i, j) not in blocked_nodes :
                    crew_alien_grid[i][j]+=copy_crew_alien_grid[i][j]
                    
                    summation += copy_crew_alien_grid[i][j]
                    

    if oper == "outside":

        for i in range(size):
            for j in range(size):
                if (i, j) not in blocked_nodes and (i,j) not in opened_nodes_inside_crew_alien:
                    crew_alien_grid[i][j]+=copy_crew_alien_grid[i][j]
                    summation += copy_crew_alien_grid[i][j]
                    
    if oper == "stepOneOutside":
        for i in range(size):
            for j in range(size):
                if (i, j) not in blocked_nodes and (i, j) not in opened_nodes_inside_crew_alien:
                    crew_alien_grid[i][j] = 1
                    summation += crew_alien_grid[i][j]

                    ##Probalizing the values
    for i in range(size):
        for j in range(size):
            crew_alien_grid[i][j] = crew_alien_grid[i][j] / summation

    return crew_alien_grid

# Checking all the possible moves that the alien can make
def getPossibleMoves(p, q, blocked_nodes):
    move_set = ((0, -1), (0, 1), (-1, 0), (1, 0), (0, 0))
    possibles = []
    for move in move_set:
        p_adj = p + move[0]
        q_adj = q + move[1]
        if p_adj < 0 or q_adj < 0 or p_adj >= size or q_adj >= size:
            continue
        if (p_adj, q_adj) not in blocked_nodes:
            possibles.append((p_adj, q_adj))

    return possibles

# Function to check whether the alien is inside the detection square
def find_crew_found_inside_detection_grid(grid, aliens, x, y):
    x_start = x - (crew_alien_grid_size // 2)
    x_end = x - (crew_alien_grid_size // 2) + crew_alien_grid_size - 1
    y_start = y - (crew_alien_grid_size // 2)
    y_end = y - (crew_alien_grid_size // 2) + crew_alien_grid_size - 1

    for i in range(x_start, x_end + 1):
        for j in range(y_start, y_end + 1):
            if (i, j) in aliens:
                return True
    return False

# Function to get the neighbouring cell with the least probability
def getBestPossibleNeighbour(node, blocked_nodes, crew_alien_grid):
    move_set = ((0, -1), (0, 1), (-1, 0), (1, 0))
    possibles = None
    probMin = float("inf")
    for move in move_set:
        p_adj = node[0] + move[0]
        q_adj = node[1] + move[1]
        if p_adj < 0 or q_adj < 0 or p_adj >= size or q_adj >= size:
            continue
        if (p_adj, q_adj) not in blocked_nodes:
            if crew_alien_grid[p_adj][q_adj] < probMin and crew_alien_grid[p_adj][q_adj] <= alien_threshold :
                probMin = crew_alien_grid[p_adj][q_adj]
                possibles = (p_adj, q_adj)
    if possibles is None:
        possibles = node
    return possibles

# Function to find the probability of an alien entering the current bot position
def probOfComingToTheBotPosition(node, blocked_nodes, crew_alien_grid):
    move_set = ((0, -1), (0, 1), (-1, 0), (1, 0))
    summing = 0
    for move in move_set:
        p_adj = node[0] + move[0]
        q_adj = node[1] + move[1]
        if p_adj < 0 or q_adj < 0 or p_adj >= size or q_adj >= size:
            continue
        if (p_adj, q_adj) not in blocked_nodes:
            length = len(getPossibleMoves(p_adj, q_adj, blocked_nodes))
            if length > 1:
                summing += (crew_alien_grid[p_adj][q_adj] / length)
    return summing

# This function is used to covert the pair_wise probability of the whole grid into a single cell probability matrix
def compress_probs(pair_probs):
    for i in range(size):
        for j in range(size):
            temp_prob = 0
            for p in range(size):
                for q in range(size):
                    if pair_probs[i][j][p][q] > 0:
                        temp_prob += pair_probs[i][j][p][q]
            if temp_prob > 0:
                prob_mat[i][j] = temp_prob
            else:
                prob_mat[i][j] = -1
    return prob_mat

# This function is used to update the crew probability distribution once we enter a cell without a crew member present in it
def no_crew_update(factor):
    for i in range(size):
        for j in range(size):
            if prob_mat[i][j] > -1:
                prob_mat[i][j] = prob_mat[i][j] / factor

# This fuction is used to estimate what the grid distribution would have been without the first crew member
def refactor_probs(initial_node, track_path, track_beep, old_prob_dist):
    beep_nodes = []
    # Number of beeps encountered in track_path
    beep_count = 0
    for i in range(len(track_beep)):
        if track_beep[i] == 1:
            beep_nodes.append(i)
            beep_count += 1
    # Estimating beeps due to single crew member
    remove_beeps = math.ceil(beep_count * 0.)

    # Generating single crew member beep pattern
    for i in range(remove_beeps):
        remove = random.choice(beep_nodes)
        beep_nodes.remove(remove)
        track_beep[remove] = 0
    
    # Generating distribution estimate for one crew member
    for pos in range(len(track_beep)):
        if track_beep[pos] == 1:
            old_prob_dist = update_probs_1(track_path[pos], old_prob_dist)    
    return old_prob_dist

# performing all the initialization steps
open_node = rand_cords()
open_grid(open_node[0], open_node[1])

crew_alien_grid = [[0 for i in range(size)] for j in range(size)]
prev = None

# Performing all the initalization steps
blocked_nodes = []
open_cells = []
for i in range(size):
    for j in range(size):
        if grid[i][j] == 1:
            blocked_nodes.append((i, j))
        else:
            open_cells.append((i, j))

initial_node = rand_cords()

while initial_node in blocked_nodes:
    initial_node = rand_cords()

while len(aliens) < k:
    new_node = rand_cords()
    if new_node not in blocked_nodes and new_node != initial_node and new_node not in aliens:
        aliens.append(new_node)

while len(desired_nodes) < 2:
    desired_node = rand_cords()
    while desired_node in blocked_nodes or desired_node == initial_node or desired_node in aliens:
        desired_node = rand_cords()
    desired_nodes.append(desired_node)

# creating a list with all the open cells in our grid
for i in range(len(open_cells) - 1):
    for j in range(i + 1, len(open_cells)):
        open_pairs.append((open_cells[i], open_cells[j]))

# Creating a 4D array to keep track of all the crew probabilties pairwise
total = 0
for p in range(size):
    row1 = []
    for q in range(size):
        row2 = []
        for r in range(size):
            row3 =[]
            for s in range(size):
                if (p, q) != (r, s):
                    if (p, q) not in blocked_nodes and (r, s) not in blocked_nodes:
                        row3.append(1)
                        total +=1
                    else:
                        row3.append(-1)
                else:
                    row3.append(-1)
            row2.append(row3)
        row1.append(row2)
    pair_probs.append(row1)

pair_probs = np.array(pair_probs)
pair_probs = pair_probs / total


prev_node = initial_node
f=open('bot5Analysis.csv','a')
writer=csv.writer(f)
ans = 0
while simulate:
    flag = 0
    while len(desired_nodes) > 0:
        visualizeGrid(initial_node, aliens)
        if initial_node in desired_nodes:
            desired_nodes.remove(initial_node)
            # Both crew members found
            if len(desired_nodes) == 0:
                writer.writerow([alpha,crew_alien_grid_size,alien_threshold,time_step,idle_time,"alien Found",2 - len(desired_nodes)])
                f.close()
                exit(1)
            # Only one crew member found
            if len(desired_nodes) == 1:
                # simulating alternate distribution for one crew member
                prob_mat = refactor_probs(initial_node,track_path, track_beep, old_prob_dist)
        visualizeGrid(initial_node, aliens)
        time.sleep(0.2)

        # Checking whether the alien has entred the detection square
        crew_found_inside_detection_grid = find_crew_found_inside_detection_grid(grid, aliens, initial_node[0],initial_node[1])
        # Updating the alien distribution once the bot moves
        crew_alien_grid = update_crew_alien_grid_BotMoved(grid, crew_alien_grid, initial_node[0], initial_node[1], prev,
                                                       crew_found_inside_detection_grid, blocked_nodes,
                                                       (initial_node[0] - prev_node[0], initial_node[1] - prev_node[1]))
        prev = crew_found_inside_detection_grid
        prev_node=initial_node

        # Generating beep if two crew members are present
        if len(desired_nodes) > 1:
            d1 = len(astar(grid, initial_node, desired_nodes[0], aliens))
            d2 = len(astar(grid, initial_node, desired_nodes[1], aliens))
            ans = beep_prob(alpha, d1, d2)[0]
        # Generating beep if one crew member is present
        else:
            d = len(astar(grid, initial_node, desired_nodes[0], aliens))
            ans = beep_prob(alpha, d)[0]

        if initial_node not in grid_distances:
            calc_distances(initial_node)
        if ans == 1:
            if len(desired_nodes) == 2:
                pair_probs = update_probs(initial_node, pair_probs)
            elif len(desired_nodes) == 1:
                prob_mat = update_probs_1(initial_node, prob_mat)
        elif ans == 0:
            if len(desired_nodes) == 2:
                pair_probs = update_no_probs(initial_node, pair_probs)
            elif len(desired_nodes) == 1:
                prob_mat = update_no_probs_1(initial_node, prob_mat)
        aliens = new_alien_pos(aliens)
        if initial_node in aliens:
            # Alien enters bot occupied cell
            writer.writerow([alpha,crew_alien_grid_size,alien_threshold,time_step,idle_time,"alien Found",2 - len(desired_nodes)])
            exit(1)
        
        # Checking whether the alien has entered the detection square
        crew_found_inside_detection_grid = find_crew_found_inside_detection_grid(grid, aliens, initial_node[0],initial_node[1])
        # Updating the alien probability distribution once the alien moves
        crew_alien_grid = update_crew_alien_grid_AlienMoved(grid, crew_alien_grid, initial_node[0], initial_node[1], prev,
                                                       crew_found_inside_detection_grid, blocked_nodes,
                                                       (initial_node[0] - prev_node[0], initial_node[1] - prev_node[1]))
        #movement will always be (0,0) since prev node has been set to initial node
        prev = crew_found_inside_detection_grid
        prev_node=initial_node

        # Checking whether the current bot position is acceptable
        if probOfComingToTheBotPosition(initial_node, blocked_nodes, crew_alien_grid) > alien_threshold:
            initial_node = getBestPossibleNeighbour(initial_node, blocked_nodes, crew_alien_grid)
            prev_node = initial_node
            idle_time += 1
            flag = 1
        if flag == 1:
            break

        # Block to be executed if two crew members are present
        if len(desired_nodes) == 2:    
            # Checking which pair has the maximum probability
            max_prob = -1
            move_to = None
            for i in open_pairs:
                if i[0] != initial_node and i[1] != initial_node:
                    if pair_probs[i[0][0]][i[0][1]][i[1][0]][i[1][1]] > max_prob:
                        max_prob = pair_probs[i[0][0]][i[0][1]][i[1][0]][i[1][1]]
                        max_pairs = [i]
                    elif pair_probs[i[0][0]][i[0][1]][i[1][0]][i[1][1]] == max_prob:
                        max_pairs.append(i)
            move_to = random.choice(max_pairs)

            # Choosing which cell within the given pair to visit
            p1 = 0
            p2 = 0
            for i in range(size):
                for j in range(size):
                    if pair_probs[move_to[0][0]][move_to[0][1]][i][j] > 0:
                        p1 += pair_probs[move_to[0][0]][move_to[0][1]][i][j]
                    if pair_probs[i][j][move_to[0][0]][move_to[0][1]] > 0:
                        p1 += pair_probs[i][j][move_to[0][0]][move_to[0][1]]
                    if pair_probs[move_to[1][0]][move_to[1][1]][i][j] > 0:
                        p1 += pair_probs[move_to[1][0]][move_to[1][1]][i][j]
                    if pair_probs[i][j][move_to[1][0]][move_to[1][1]] > 0:
                        p1 += pair_probs[i][j][move_to[1][0]][move_to[1][1]]
            if p1 > p2:
                path = astar(grid, initial_node, move_to[0], aliens)
            elif p2 > p1:
                path = astar(grid, initial_node, move_to[1], aliens)
            else:
                node = random.choice([move_to[0], move_to[1]])
                path = astar(grid, initial_node, node, aliens)
            # moving the bot towards the cell with maximum probability
            initial_node = path[1]
            # storing the previouly visited cells and beeps
            track_path.append(initial_node)
            track_beep.append(ans)            

            # removing cells if they exceed the tracking value we have set
            if len(track_path) > track_len:
                track_path.pop(0)
                track_beep.pop(0)
                old_prob_dist = compress_probs(pair_probs)
            # Calculating the probability of the current cell
            cell_prob = 0
            for i in range(initial_node[0] + 1, size):
                for j in range(initial_node[1] + 1, size):
                    if pair_probs[initial_node[0]][initial_node[1]][i][j] > 0:
                        cell_prob += pair_probs[initial_node[0]][initial_node[1]][i][j]
            # probabitliy of crew not being present in the current cell
            factor = 1 - cell_prob

            # Checking if bot found a crew member
            if initial_node in desired_nodes:
                desired_nodes.remove(initial_node)
                if len(desired_nodes) == 0:
                    writer.writerow([alpha,crew_alien_grid_size,alien_threshold,time_step,idle_time,"alien Found",2 - len(desired_nodes)])
                    exit(1)
                if len(desired_nodes) == 1:
                    prob_mat = refactor_probs(initial_node, track_path, track_beep, old_prob_dist)
            
            # Setting the probaility all cells containing the current pair to 0 if crew member is not present
            for a in range(size):
                for b in range(size):
                    pair_probs[initial_node[0]][initial_node[1]][a][b] = 0
                    pair_probs[a][b][initial_node[0]][initial_node[1]] = 0

            for pair in open_pairs:
                # Removing a pair from dictionary if it has been visited
                if pair[0] == initial_node or pair[1] == initial_node:
                    open_pairs.remove(pair)
                # Distributing the probability of non crew pair throughout the grid
                else:
                    pair_probs[pair[0][0]][pair[0][1]][pair[1][0]][pair[1][1]] = pair_probs[pair[0][0]][pair[0][1]][pair[1][0]][pair[1][1]] / factor
                prob_mat = compress_probs(pair_probs)
        
        # Block to be executed if one crew member is present
        elif len(desired_nodes) == 1:
            # Checking the grid for cells with maximum probability
            max_prob = -1
            max_nodes = []
            for i in range(size):
                for j in range(size):
                    if prob_mat[i][j] > max_prob:
                        max_prob = prob_mat[i][j]
                        max_nodes = [(i, j)]
                    elif prob_mat[i][j] == max_prob: 
                        max_nodes.append((i, j))
            if initial_node in max_nodes:
                max_nodes.remove(initial_node)
            # Choosing which cell to move towards
            move_to = random.choice(max_nodes)
            path = astar(grid, initial_node, move_to, aliens)
            # Moving the crew towards the cell with maximum probability
            initial_node = path[1]
            if initial_node in aliens:
                # bot entered alien occupied cell
                writer.writerow([alpha,crew_alien_grid_size,alien_threshold,time_step,idle_time,"alien Found",2 - len(desired_nodes)])
                f.close()
                sys.exit(1)
            # probability of crew not being present in the current cell
            factor = (1 - prob_mat[initial_node[0]][initial_node[1]])
            #  Blocking the current cell probability because no crew member is present
            prob_mat[initial_node[0]][initial_node[1]] = -1
            # Distributing current cell crew probability to all other open cells if crew member is not present
            no_crew_update(factor)
pygame.quit()