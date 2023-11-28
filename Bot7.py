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
alpha = 0.5
aliens = []
desired_nodes = []
open_pairs = []
pair_probs = []
k = 2
fps = 60
grid_node_width = 25
grid_node_height = 25
size = 30
prob_mat = [[1 for i in range(size)] for j in range(size)]
simulate = True
total_open = 0
time_step = 0
crew_alien_grid_size = 5
alien_threshold = 0
grid_distances = {}
idle_time = 0
WIDTH = grid_node_width * size
HEIGHT = grid_node_height * size
screen = pygame.display.set_mode([WIDTH, HEIGHT])
pygame.display.set_caption('Bot 7 Algo')
img = pygame.image.load('bot.jpg')
img = pygame.transform.scale(img, (grid_node_height - 5, grid_node_width - 5))
img_1 = pygame.image.load('Deoxys.png')
img_1 = pygame.transform.scale(img_1, (grid_node_height - 5, grid_node_width - 5))
img_2 = pygame.image.load('crew.png')
img_2 = pygame.transform.scale(img_2, (grid_node_height - 5, grid_node_width - 5))


# creating a closed grid of given dimensions
grid = np.array([[1 for i in range(size)] for j in range(size)])
prob_mat = copy.deepcopy(grid)
prob_mat = [[1 for i in range(size)] for j in range(size)]


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
    global total_open
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
        # prob_mat[new_x][new_y] = 1
        total_open += 1


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

# Function to calculate distance to all other open nodes from a given cell
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
                total += prob_mat[i][j]

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
            total += pair_probs[i[0][0]][i[0][1]][i[1][0]][i[1][1]] 
    pair_probs = pair_probs / total
    return pair_probs

# Function used for probability updation when no beep is heard
# This function is used when one crew member is present in the grid
def update_no_probs_1(start_cell, prob_mat):
    total = 0
    for i in range(size):
        for j in range(size):
            if prob_mat[i][j] > -1:
                prob_mat[i][j] = prob_mat[i][j]*(1 - math.exp(-alpha * grid_distances[start_cell][(i, j)]))
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
            total += pair_probs[i[0][0]][i[0][1]][i[1][0]][i[1][1]] 
    pair_probs = pair_probs / total
    return pair_probs

# Function to get all the possible moves that an alien can make
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

# Function to check whether an alien has entered the detection box
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

# Function to get the neighbour cell with the least alien probability
def getBestPossibleNeighbour(node, blocked_nodes, alien_pairs):
    move_set = ((0, -1), (0, 1), (-1, 0), (1, 0))
    possibles = None
    probMin = float("inf")
    for move in move_set:
        p_adj = node[0] + move[0]
        q_adj = node[1] + move[1]
        if p_adj < 0 or q_adj < 0 or p_adj >= size or q_adj >= size:
            continue
        if (p_adj, q_adj) not in blocked_nodes:
            if get_sum_of_prob(p_adj,q_adj,alien_pairs) < probMin and get_sum_of_prob(p_adj,q_adj,alien_pairs) <= alien_threshold :
                probMin = get_sum_of_prob(p_adj,q_adj,alien_pairs)
                possibles = (p_adj, q_adj)
    if possibles is None:
        possibles = node
    return possibles

def get_sum_of_prob(x,y,alien_pairs):
    part0=alien_pairs[0]
    part1=alien_pairs[1]
    value=alien_pairs[2]
    sums=0
    for idx,ele in enumerate(part0):
        if ele == (x,y):
            
            sums+=value[idx]
    for idx,ele in enumerate(part1):
        if ele == (x,y):
            
            sums+=value[idx]
    return sums

# Function to check the probability of an alien entering the bot occupied cell
def probOfComingToTheBotPosition(node, blocked_nodes, alien_pairs):
    move_set = ((0, -1), (0, 1), (-1, 0), (1, 0))
    summing = 0
    for move in move_set:
        p_adj = node[0] + move[0]
        q_adj = node[1] + move[1]
        if p_adj < 0 or q_adj < 0 or p_adj >= size or q_adj >= size:
            continue
        if (p_adj, q_adj) not in blocked_nodes:
            
            summing+=(get_sum_of_prob(p_adj,q_adj,alien_pairs)/ len(getPossibleMoves(p_adj, q_adj, blocked_nodes)) )
            
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

# performing all the initialization steps
open_node = rand_cords()
open_grid(open_node[0], open_node[1])

# Function co 
def generate_alien_pairs(grid,blocked_nodes,open_pairs):
    
    total=0
    part1=[]
    part2=[]
    part3=[]
    alien_pairs=[]
    for i in range(len(open_pairs)):
        part1.append(open_pairs[i][0])
        part2.append(open_pairs[i][1])
        part3.append(0)
    
    return [part1,part2,part3] 

def generate_neighbours_for_each_cell(open_cells,blocked_nodes):
    dic={}
    for i, ele in enumerate(open_cells):
        move_set = ((0, -1), (0, 1), (-1, 0), (1, 0))
        possibles = []
        for move in move_set:
            p_adj = ele[0] + move[0]
            q_adj = ele[1] + move[1]
            if p_adj < 0 or q_adj < 0 or p_adj >= size or q_adj >= size:
                continue
            if (p_adj, q_adj) not in blocked_nodes:
                possibles.append((p_adj, q_adj))
                
        dic[ele]=possibles
    return dic

def update_alien_pairs(x,y,blocked_nodes,alien_pairs,prev,crew_found_inside_detection_grid,neigbhours_dic):
    copy_alien_pairs=copy.deepcopy(alien_pairs)
    part0=alien_pairs[0]
    part1=alien_pairs[1]
    part2=alien_pairs[2]
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
    
    
    if oper=="stepOneInside":
        for i in range(len(part2)):
            part2[i]=1/len(part2)
        return [part0,part1,part2]
    total=0
    if oper=="stepOneOutside":
        for i in range(len(part0)):
            if not (((part0[i][0]>=x_start and part0[i][0]<=x_end) and (part0[i][1]>=y_start and part0[i][1]<=y_end )) or ((part1[i][0]>=x_start and part1[i][0]<=x_end) and (part1[i][1]>=y_start and part1[i][1]<=y_end ))):
                part2[i]=1
                total+=1
        for i in range(len(part2)):
            part2[i]=part2[i]/total
            
        return [part0,part1,part2]
    
    if oper=="inside":
        newpart2=[0]*len(part2)
        for i in range(len(part0)):
            for j in range(i,len(part1)):
                temp=0
                if check_if_neighbour(part0[i], part0[j],neigbhours_dic) and check_if_neighbour( part1[i], part1[j],neigbhours_dic)  :
                    temp+=((1/len(neigbhours_dic[part0[j]])+1) *  (1/len(neigbhours_dic[part1[j]])+1) * part2[j] )
                    
                    
                
                if check_if_neighbour(part0[i], part1[j],neigbhours_dic) and check_if_neighbour(part1[i], part0[j],neigbhours_dic): 
                    temp+=((1/len(neigbhours_dic[part1[j]])+1) *  (1/len(neigbhours_dic[part0[j]])+1) * part2[j] )
                    
                newpart2[i]+=temp
        
        part2=[x / sum(newpart2) for x in newpart2]
        return [part0,part1,part2]
    
    if oper=="outside":
        opened_detection_cells=[]
        for i in range(x_start, x_end + 1):
            for j in range(y_start, y_end + 1):
                if i < 0 or j < 0 or i >= size or j >= size and (i,j) in blocked_nodes:
                    continue
                opened_detection_cells.append((i,j))
                
        
        
        newpart2=[0]*len(part2)
        for i in range(len(part0)):
            for j in range(i,len(part1)):
                if not (((part0[i][0]>=x_start and part0[i][0]<=x_end) and (part0[i][1]>=y_start and part0[i][1]<=y_end )) or ((part1[i][0]>=x_start and part1[i][0]<=x_end) and (part1[i][1]>=y_start and part1[i][1]<=y_end ))):
                    temp=0
                    if check_if_neighbour(part0[i], part0[j],neigbhours_dic) and check_if_neighbour( part1[i], part1[j],neigbhours_dic)  :
                        temp+=((1/len(getPossibleMoves( part0[j][0] ,part0[j][1] , blocked_nodes+opened_detection_cells))) *  (1/len(getPossibleMoves( part1[j][0] ,part1[j][1] , blocked_nodes+opened_detection_cells)) * part2[j] ))
                        
                        
                    
                    if check_if_neighbour(part0[i], part1[j],neigbhours_dic) and check_if_neighbour(part1[i], part0[j],neigbhours_dic):
                        temp+=((1/len(getPossibleMoves( part1[j][0] ,part1[j][1] , blocked_nodes+opened_detection_cells))) *  (1/len(getPossibleMoves( part0[j][0] ,part0[j][1] , blocked_nodes+opened_detection_cells)) * part2[j] ))
                        
                    newpart2[i]+=temp
        
        part2=[x / sum(newpart2) for x in newpart2]
        return [part0,part1,part2]

def check_if_neighbour(nodea,nodeb,neigbhours_dic):
    if nodeb in neigbhours_dic[nodea]:
        return True
    return False


prev = None

# generating a list of all the cells that are blocked
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

for i in range(len(open_cells) - 1):
    for j in range(i + 1, len(open_cells)):
        open_pairs.append((open_cells[i], open_cells[j]))
        
alien_pairs=generate_alien_pairs(grid, blocked_nodes,open_pairs)

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

neigbhours_dic=generate_neighbours_for_each_cell(open_cells,blocked_nodes)

prev_node = initial_node
f=open('bot7Analysis.csv','a')
writer=csv.writer(f)
ans = 0
while simulate:
    flag = 0
    while len(desired_nodes) > 0:
        flag=0
        time_step += 1
        # Checking whether a crew member is present in our current cell
        if initial_node in desired_nodes:
            desired_nodes.remove(initial_node)
            if len(desired_nodes) == 0:
                writer.writerow([alpha,crew_alien_grid_size,alien_threshold,time_step,idle_time,"crew Found",2 - len(desired_nodes)])
                f.close()
                exit(1)
            if len(desired_nodes) == 1:
                prob_mat = compress_probs(pair_probs)
        visualizeGrid(initial_node, aliens)
        time.sleep(0.2)

        #Detect if alien is inside the detection square
        crew_found_inside_detection_grid = find_crew_found_inside_detection_grid(grid, aliens, initial_node[0],initial_node[1])
        
        #Update alien probabilty pairs
        alien_pairs=update_alien_pairs(initial_node[0],initial_node[1], blocked_nodes, alien_pairs, prev, crew_found_inside_detection_grid,neigbhours_dic)
        prev = crew_found_inside_detection_grid
        prev_node=initial_node
        
        # Generating beep for when there are two crew members
        if len(desired_nodes) > 1:
            d1 = len(astar(grid, initial_node, desired_nodes[0], aliens))
            d2 = len(astar(grid, initial_node, desired_nodes[1], aliens))
            ans = beep_prob(alpha, d1, d2)[0]
        # Generating beep if there is one crew member
        else:
            d = len(astar(grid, initial_node, desired_nodes[0], aliens))
            ans = beep_prob(alpha, d)[0]

        if initial_node not in grid_distances:
            calc_distances(initial_node)
        
        # Updating the crew probability distribution
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

        # Moving the aliens within the grid
        aliens = new_alien_pos(aliens)
        if initial_node in aliens:
            # Alien enters bot occupied cell
            writer.writerow([alpha,crew_alien_grid_size,alien_threshold,time_step,idle_time,"alien Found",2 - len(desired_nodes)])
            f.close()
            exit(1)
        
        # check if alien is present inside the detection square
        crew_found_inside_detection_grid = find_crew_found_inside_detection_grid(grid, aliens, initial_node[0],initial_node[1])
        #Update alien probabilty pairs
        alien_pairs=update_alien_pairs(initial_node[0],initial_node[1], blocked_nodes, alien_pairs, prev, crew_found_inside_detection_grid,neigbhours_dic)
       
        prev = crew_found_inside_detection_grid
        prev_node=initial_node
        
        #checking for the probability of alien coming to bot position 
        if crew_found_inside_detection_grid and probOfComingToTheBotPosition(initial_node, blocked_nodes, alien_pairs) > alien_threshold:
            initial_node = getBestPossibleNeighbour(initial_node, blocked_nodes, alien_pairs)
            prev_node = initial_node
            idle_time+=1
            flag = 1
        # Block to be executed when there are two crew members present in the grid
        if len(desired_nodes) == 2:    
            # Checking for which cell has the maximum probability
            max_prob = -1
            move_to = None
            for i in open_pairs:
                if i[0] != initial_node and i[1] != initial_node:
                    if pair_probs[i[0][0]][i[0][1]][i[1][0]][i[1][1]] > max_prob:
                        max_prob = pair_probs[i[0][0]][i[0][1]][i[1][0]][i[1][1]]
                        max_pairs = [i]
                    elif pair_probs[i[0][0]][i[0][1]][i[1][0]][i[1][1]] == max_prob:
                        max_pairs.append(i)
            # Choosing a pair from all the pairs that have maximum probability
            move_to = random.choice(max_pairs)
            
            # Checking which cell within the pair has the highest probability
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
            # Selecting the cell with higher probability from the pair
            if p1 > p2:
                path = astar(grid, initial_node, move_to[0], aliens)
            elif p2 > p1:
                path = astar(grid, initial_node, move_to[1], aliens)
            else:
                node = random.choice([move_to[0], move_to[1]])
                path = astar(grid, initial_node, node, aliens)
            # moving the bot towards the max probability cell
            initial_node = path[1]
            # calculating the probability of crew member being present in the current cell
            cell_prob = 0
            for i in range(initial_node[0] + 1, size):
                for j in range(initial_node[1] + 1, size):
                    if pair_probs[initial_node[0]][initial_node[1]][i][j] > 0:
                        cell_prob += pair_probs[initial_node[0]][initial_node[1]][i][j]
            # Probability of crew member not being present in the current cell
            factor = 1 - cell_prob
            if initial_node in desired_nodes:
                desired_nodes.remove(initial_node)
                if len(desired_nodes) == 0:
                    writer.writerow([alpha,crew_alien_grid_size,alien_threshold,time_step,idle_time,"alien Found",2 - len(desired_nodes)])
                    f.close()
                    exit(1)
                if len(desired_nodes) > 0:
                    prob_mat = compress_probs(pair_probs)
            
            # Setting the probability of all the pairs containing the current cell to 0
            for a in range(size):
                for b in range(size):
                    pair_probs[initial_node[0]][initial_node[1]][a][b] = 0
                    pair_probs[a][b][initial_node[0]][initial_node[1]] = 0

            for pair in open_pairs:
                # Removing a pair from the list of open pairs if it has been visited
                if pair[0] == initial_node or pair[1] == initial_node:
                    open_pairs.remove(pair)
                # Distributing the probability of the pair without crew member across all other open pairs
                else:
                    pair_probs[pair[0][0]][pair[0][1]][pair[1][0]][pair[1][1]] = pair_probs[pair[0][0]][pair[0][1]][pair[1][0]][pair[1][1]] / factor
            calc_distances(initial_node)
            # Converting the pairwise probability into single cell probability
            prob_mat = compress_probs(pair_probs)
        # Block to be executed when there is one crew member in the grid
        elif len(desired_nodes) == 1:
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
            move_to = random.choice(max_nodes)
            path = astar(grid, initial_node, move_to, aliens)
            # moving the alien towards the max probability cell
            initial_node = path[1]
        
            if initial_node in aliens:
                # bot entered alien cells
                writer.writerow([alpha,crew_alien_grid_size,alien_threshold,time_step,idle_time,"alien Found",2 - len(desired_nodes)])
                f.close()
                sys.exit(1)
            if initial_node == desired_node:
                # bot entered alien cells
                writer.writerow([alpha,crew_alien_grid_size,alien_threshold,time_step,idle_time,"alien Found",2 - len(desired_nodes)])
                f.close()
                sys.exit(1)
            # Probability of crew member not being present in the current cell
            factor = (1 - prob_mat[initial_node[0]][initial_node[1]])
            # Blocking the probabilty of crew member being present in the current cell
            prob_mat[initial_node[0]][initial_node[1]] = -1
            # Distributing the probability of crew member across the whole grid
            no_crew_update(factor)
pygame.quit()