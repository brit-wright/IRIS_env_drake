#!/usr/bin/env python3.10
import numpy as np
import matplotlib.pyplot as plt

from math import sqrt, inf, pi, sin, cos, atan2, ceil
from scipy.spatial      import KDTree
from shapely.geometry   import Point, LineString, Polygon, MultiPolygon, MultiLineString
from shapely.prepared   import prep


import random
from pydrake.all import *
import bisect
import time
import torch
import csv


STEP_SIZE = 0.5
NMAX = 3000
SMAX = 3000

device='cpu'

###############################################################################################
# Create a seed
# seed = int(random.random()*10000)
# seed = 2040
# random.seed(seed)
# print(f"{seed=}")
###############################################################################################
# helper functions

# function to reorder the vertices of a 2D polygon in clockwise order
def reorder_verts_2D(verts):
    """
    Reorder the vertices of a 2D polygon in clockwise order.
    """
    # unpack the vertices
    x = verts[0, :]
    y = verts[1, :]

    # calculate the centroid
    x_c = np.mean(x)
    y_c = np.mean(y)

    # calculate the angles
    angles = np.arctan2(y - y_c, x - x_c)

    # sort the angles
    idx = np.argsort(angles)

    # reorder the vertices
    verts = verts[:, idx]

    # add the first vertex to the end to close the loop
    verts = np.hstack((verts, verts[:, 0].reshape(-1, 1)))

    return verts

# function to plot a 2D ellipse
def get_ellipse_pts_2D(B, c):
    """
    Get the points of a 2D ellipse via affine Ball interpretation
    """
    # create a circle
    t = np.linspace(0, 2*np.pi, 100)
    circle = np.array([np.cos(t), np.sin(t)])
    
    # print(circle.shape)

    # scale the circle
    ellipse = B @ circle
    
    # translate the ellipse
    ellipse = ellipse + c

    # print(ellipse.shape)

    return ellipse

###############################################################################################
# DOMAIN AND OBSTACLES

# define the domain of the problem, Xcal = {x | Ax <= b}
x1_min = 0
x1_max = 30
x2_min = 0
x2_max = 20

# build the free space as H polyhedron
domain_A = np.array([[1, 0],
                              [0, 1],
                              [-1, 0],
                              [0, -1]])
domain_b = np.array([[x1_max],
                              [x2_max],
                              [-x1_min],
                              [-x2_min]])
domain = HPolyhedron(domain_A, domain_b)


# V_polytope rectangles
# rect_pts1 = np.array([[2, 18],
#                      [12, 18],
#                      [12, 17],
#                      [2, 17]])

# rect_pts2 = np.array([[3, 17],
#                      [6, 17],
#                      [6, 14],
#                      [3, 14]])

# rect_pts3 = np.array([[4, 14],
#                       [5, 14],
#                       [5, 6],
#                       [4, 6]])

# rect_pts4 = np.array([[3, 6],
#                       [6, 6],
#                       [6, 3],
#                       [3, 3]])

# rect_pts5 = np.array([[3, 3],
#                       [21, 3],
#                       [21, 1],
#                       [3, 1]])

# rect_pts6 = np.array([[11, 11], 
#                     [11, 6],
#                     [12, 6],
#                     [12, 11]])

# rect_pts7 = np.array([[10, 14],
#                       [10, 11],
#                       [13, 11],
#                       [13, 14]])

# rect_pts8 = np.array([[11, 15],
#                       [18, 15],
#                       [18, 14],
#                       [11, 14]])

# rect_pts9 = np.array([[18, 16],
#                       [18, 13],
#                       [21, 13],
#                       [21, 16]])

# rect_pts10 = np.array([[21, 15],
#                        [21, 14],
#                        [27, 14],
#                        [27, 15]])

# rect_pts11 = np.array([[27, 16],
#                        [27, 13],
#                        [30, 13],
#                        [30, 16]])

# rect_pts12 = np.array([[17, 10],
#                        [17, 3],
#                        [19, 3],
#                        [19, 10]])

# rect_pts13 = np.array([[23, 11],
#                        [25, 11],
#                        [25, 6],
#                        [23, 6]])

# rect_pts14 = np.array([[22, 3],
#                        [22, 1],
#                        [28, 1],
#                        [28, 3]])

rect_pts1 = np.array([[1.7, 18.3],
                    [12.3, 18.3],
                    [12.3, 17],
                    [1.7, 17]])

rect_pts2 = np.array([[2.7, 17],
                    [6.3, 17],
                    [6.3, 14],
                    [2.7, 14]])

rect_pts3 = np.array([[3.7, 14],
                    [5.3, 14],
                    [5.3, 6],
                    [3.7, 6]])

rect_pts4 = np.array([[2.7, 6],
                    [6.3, 6],
                    [6.3, 3],
                    [2.7, 3]])

rect_pts5 = np.array([[2.7, 3],
                    [21.3, 3],
                    [21.3, 0.7],
                    [2.7, 0.7]])

rect_pts6 = np.array([[10.7, 11],
                    [10.7, 5.7],
                    [12.3, 5.7],
                    [12.3, 11]])

rect_pts7 = np.array([[9.7, 14],
                    [9.7, 11],
                    [13.3, 11],
                    [13.3, 14]])

rect_pts8 = np.array([[10.7, 15.3],
                    [18, 15.3],
                    [18, 14],
                    [10.7, 14]])

rect_pts9 = np.array([[18, 16.3],
                    [18, 12.7],
                    [21, 12.7],
                    [21, 16.3]])

rect_pts10 = np.array([[21, 15.3],
                     [21, 13.7], 
                     [27, 13.7],
                     [27, 15.3]])

rect_pts11 = np.array([[27, 16.3],
                     [27, 12.7],
                     [30, 12.7],
                     [30, 16.3]])

rect_pts12 = np.array([[16.7, 10.3],
                     [16.7, 3],
                     [19.3, 3],
                     [19.3, 10.3]])

rect_pts13 = np.array([[22.7, 11.3],
                     [25.3, 11.3],
                     [25.3, 5.7],
                     [22.7, 5.7]])

rect_pts14 = np.array([[21.7, 3.3],
                     [21.7, 0.7],
                     [28.3, 0.7],
                     [28.3, 3.3]])

rect_pts15 = np.array([[1.7, 17],
                     [1.7, 16.7],
                     [2.7, 16.7],
                     [2.7, 17]])

rect_pts16 = np.array([[6.3, 17],
                     [6.3, 16.7],
                     [12.3, 16.7],
                     [12.3, 17]])

rect_pts17 = np.array([[2.7, 14],
                     [2.7, 13.7],
                     [3.7, 13.7],
                     [3.7, 14]])

rect_pts18 = np.array([[5.3, 14],
                     [5.3, 13.7],
                     [6.3, 13.7],
                     [6.3, 14],
                     [5.3, 14]])

rect_pts19 = np.array([[3.7, 6],
                     [3.7, 6.3],
                     [2.7, 6.3],
                     [2.7, 3],
                     [2.7, 3]])

rect_pts20 = np.array([[5.3, 6],
                     [5.3, 6.3],
                     [6.3, 6.3],
                     [6.3, 3]])

rect_pts21 = np.array([[6.3, 3],
                     [6.3, 3.3],
                     [16.7, 3.3],
                     [16.7, 3]])

rect_pts22 = np.array([[19.3, 3],
                     [19.3, 3.3],
                     [21.3, 3.3],
                     [21.3, 3]])

rect_pts23 = np.array([[10.7, 11],
                     [10.7, 10.7],
                     [9.7, 10.7],
                     [9.7, 11]])

rect_pts24 = np.array([[12.3, 11],
                     [12.3, 10.7],
                     [13.3, 10.7],
                     [13.3, 11]])

rect_pts25 = np.array([[13.3, 14],
                     [13.3, 13.7],
                     [18, 13.7],
                     [18, 14]])

rect_pts26 = np.array([[18, 13.7],
                     [18, 12.7],
                     [17.7, 12.7],
                     [17.7, 13.7]])

rect_pts27 = np.array([[9.7, 14.3],
                     [9.7, 14],
                     [10.7, 14],
                     [10.7, 14.3]])

rect_pts28 = np.array([[17.7, 15.3],
                     [17.7, 16.3],
                     [18, 16.3],
                     [18, 15.3]])

rect_pts29 = np.array([[21, 16.3],
                     [21, 15.3],
                     [21.3, 15.3],
                     [21.3, 16.3]])

rect_pts30 = np.array([[21.3, 12.7],
                     [21.3, 13.7],
                     [21, 13.7],
                     [21, 12.7]])

rect_pts31 = np.array([[26.7, 16.3],
                     [26.7, 15.3],
                     [27, 15.3],
                     [27, 16.3]])

rect_pts32 = np.array([[26.7, 12.7],
                     [26.7, 13.7],
                     [27, 13.7],
                     [27, 12.7]])

obs_rect1 = VPolytope(rect_pts1.T)
obs_rect2 = VPolytope(rect_pts2.T)
obs_rect3 = VPolytope(rect_pts3.T)
obs_rect4 = VPolytope(rect_pts4.T)
obs_rect5 = VPolytope(rect_pts5.T)
obs_rect6 = VPolytope(rect_pts6.T)
obs_rect7 = VPolytope(rect_pts7.T)
obs_rect8 = VPolytope(rect_pts8.T)
obs_rect9 = VPolytope(rect_pts9.T)
obs_rect10 = VPolytope(rect_pts10.T)
obs_rect11 = VPolytope(rect_pts11.T)
obs_rect12 = VPolytope(rect_pts12.T)
obs_rect13 = VPolytope(rect_pts13.T)
obs_rect14 = VPolytope(rect_pts14.T)
obs_rect15 = VPolytope(rect_pts15.T)
obs_rect16 = VPolytope(rect_pts16.T)
obs_rect17 = VPolytope(rect_pts17.T)
obs_rect18 = VPolytope(rect_pts18.T)
obs_rect19 = VPolytope(rect_pts19.T)
obs_rect20 = VPolytope(rect_pts20.T)
obs_rect21 = VPolytope(rect_pts21.T)
obs_rect22 = VPolytope(rect_pts22.T)
obs_rect23 = VPolytope(rect_pts23.T)
obs_rect24 = VPolytope(rect_pts24.T)
obs_rect25 = VPolytope(rect_pts25.T)
obs_rect26 = VPolytope(rect_pts26.T)
obs_rect27 = VPolytope(rect_pts27.T)
obs_rect28 = VPolytope(rect_pts28.T)
obs_rect29 = VPolytope(rect_pts29.T)
obs_rect30 = VPolytope(rect_pts30.T)
obs_rect31 = VPolytope(rect_pts32.T)
obs_rect32 = VPolytope(rect_pts32.T)


obstacles = [obs_rect1, obs_rect2, obs_rect3, obs_rect4, obs_rect5, obs_rect6, obs_rect7, 
             obs_rect8, obs_rect9, obs_rect10, obs_rect11, obs_rect12, obs_rect13, obs_rect14,
             obs_rect15, obs_rect16, obs_rect17, obs_rect18, obs_rect19, obs_rect20, obs_rect21,
             obs_rect22, obs_rect23, obs_rect24, obs_rect25, obs_rect26, obs_rect27, obs_rect28,
             obs_rect29, obs_rect30, obs_rect31, obs_rect32]


###############################################################################################
# DEFINE THE VISUALIZATION CLASS FOR RRT

xlabels = np.arange(x1_min, x1_max, 5)
ylabels = np.arange(x2_min, x2_max, 5)

class Visualization:
    def __init__(self):

        placeholder = True # I had nothing to put here :/

    def drawNode(self, node, *args, **kwargs):
        plt.plot(node.x, node.y, *args, **kwargs)

    def drawEdge(self, head, tail, *args, **kwargs):
        plt.plot((head.coords[0], tail.coords[0]), (head.coords[1], tail.coords[1]), *args, **kwargs)

    def drawPath(self, path, *args, **kwargs):
        for i in range(len(path)-1):
            self.drawEdge(path[i], path[i+1], *args, **kwargs)

###############################################################################################
# RRT, AND COLLISION-CHECKING FUNCTIONS
def inFreespace(next_node, starts, goals):
    t_freespace_start = time.time()
    
    start_equality_mask = torch.any(torch.all(next_node[:,None,:] == starts[None,:,:], dim=2), dim=1)

    goal_equality_mask = torch.any(torch.all(next_node[:,None,:] == goals[None,:,:], dim=2), dim=1)

    in_bounds_mask = ((next_node[:,0] >= x1_min) & (next_node[:,0] <= x1_max)) & ((next_node[:,1] >= x2_min) & (next_node[:,1] <= x2_max))

    # need to change the size of next_node to make it compatible for batch-processing
    next_node = next_node.unsqueeze(0).expand(len(obstacles), len(next_node), 2)
    
    # The A_stack_T variable has already been pre-calculated so I can just to the batch matrix multiplication here
    prod = torch.bmm(next_node, A_stack_T)

    mask = (prod <= b_stack.unsqueeze(1))
    mask2 = mask.all(dim=2)
    mask3 = mask2.any(dim=0)

    final_mask = (in_bounds_mask & ~mask3) | start_equality_mask | goal_equality_mask

    t_freespace_end = time.time()
    t_freespace = t_freespace_end - t_freespace_start
    # print(f'def inFreespace: {t_freespace}')

    return final_mask, t_freespace


def connectsTo(tens_start, tens_goal, starts, goals):

    # OPTIMIZING THIS

    t_connectsto_start = time.time()
    counting_infreespace = 0
    timing_infreespace = 0

    num_divisions = 200
    divs = torch.linspace(0, 1, steps=num_divisions, device=device).view(-1, 1)

    # Here we take in the tensors defining the start and goal points for each RRT
    # We start by doing an initial check to see which pairs of start-goal entries have the same x-value (vertical line check)

    vertical_mask = tens_start[:, 0] == tens_goal[:, 0]
    non_vertical_mask = ~vertical_mask

    
    # FOR THE NON-VERTICAL LINES
    start_non_vert = tens_start[non_vertical_mask]
    goal_non_vert = tens_goal[non_vertical_mask]

    # get the x and y values for the non-vertical case
    x_vals_non_vert = (start_non_vert[:, 0]).unsqueeze(0) + (goal_non_vert[:, 0] - start_non_vert[:, 0]).unsqueeze(0) * divs

    # check if we can calculate the y_values using the same method
    y_vals_non_vert = (start_non_vert[:, 1]).unsqueeze(0) + (goal_non_vert[:, 1] - start_non_vert[:, 1]).unsqueeze(0) * divs

    
    # collect all the test points
    new_non_vert = torch.stack([x_vals_non_vert, y_vals_non_vert], dim=2).reshape(-1, 2)

    # call inFreespace
    result_non_vert, timed = inFreespace(new_non_vert, starts, goals)

    # re-shape the result
    result_non_vert = result_non_vert.view(num_divisions, -1)

    # define the mask
    connects_non_vert = result_non_vert.all(dim=0)

    # FOR THE VERTICAL LINES
    start_vert = tens_start[vertical_mask]
    goal_vert = tens_goal[vertical_mask]

    y_vals_vert = (start_vert[:, 1]).unsqueeze(0) + (goal_vert[:, 1] - start_vert[:, 1]).unsqueeze(0) * divs
    x_vals_vert = start_vert[:, 0].unsqueeze(0).expand(num_divisions, -1)
    
    new_vert = torch.stack([x_vals_vert, y_vals_vert], dim=2).reshape(-1, 2)

    result_vert, timed = inFreespace(new_vert, starts, goals)

    result_vert = result_vert.view(num_divisions, -1)

    connects_vert = result_vert.all(dim=0)


    # next we need to re-combine based on the indices that are and aren't vertical
    connects_to_result = torch.zeros(len(tens_start), dtype=torch.bool, device=device)

    connects_to_result[vertical_mask] = connects_vert
    connects_to_result[non_vertical_mask] = connects_non_vert

    t_connectsto_end = time.time()
    t_connectsto = t_connectsto_end - t_connectsto_start
    # print(f'def connectsTo: {t_connectsto}')

    return connects_to_result, t_connectsto

def check_obstacle_collision(coord, obstacles):
    
    for v_pol in obstacles:
        # do an early check for whether the point intersects
        if v_pol.PointInSet(coord) == True:
            return True
    return False

def pointsConnect(point1, point2):

    # OPTIMIZING THIS
    num_divisions = 200
    divs = np.linspace(0, 1, num_divisions)

    if point1[1] == point2[1]:
        x_vals = [point1[0]] * num_divisions
        y_vals = point1[1] + (point2[1] - point1[1]) * divs
    else:
        x_vals = point1[0] + (point2[0] - point1[0]) * divs
        y_vals = point1[1] + (point2[1] - point1[1]) * divs

    for i in range(num_divisions):
        ans = check_obstacle_collision([x_vals[i], y_vals[i]], obstacles)
        if ans == True:
            # does not connect
            return False

    return True

# RUN RRT IN GPU (CPU for now)
def do_rrt(start_r, goal_r):
    t_begin_rrt = time.time()
    stuck_counter = 0
    device = torch.device('cpu')
    starts = torch.tensor(start_r, dtype=torch.float, device=device)
    goals = torch.tensor(goal_r, dtype=torch.float, device=device)

    batch_size = len(start_r)
    
    count_infreespace = 0
    time_infreespace = 0
    count_connects_to = 0
    time_connects_to = 0

    # checking the RRT freespace definition
    ans = inFreespace(starts, starts, goals)
    print(f'Answer for whether the starts are in Freespace: {ans}')

    ans = inFreespace(goals, starts, goals)
    print(f'Answer for whether the goals are in Freespace: {ans}')

    # I think I found an issue. Whenever things get stuck, it's because it sees the
    # starts/goal values as not being in freespace

    # time.sleep(20)

    # batch_size = 1
    print(f'Batch size is: {batch_size}')
    node_counts = torch.ones(batch_size, dtype=torch.long, device=device)
    tree_parents = torch.full((batch_size, NMAX), -1, dtype=torch.long, device=device)

    tree_positions = torch.zeros((batch_size, NMAX, 2), device=device)
    tree_positions[:, 0, 0] = starts[:, 0]
    tree_positions[:, 0, 1] = starts[:, 1]

    iter = 0

    def addtotree(valid_batches, valid_nextnodes, nearest_indices):
        # start by assigning the parent of the new_node to be the nearnode
        tree_parents[valid_batches, node_counts[valid_batches]] = nearest_indices

        # add the new node to the tree
        tree_positions[valid_batches, node_counts[valid_batches], :] = valid_nextnodes

        # increment the node cout
        next_node_index = node_counts[valid_batches]
        node_counts[valid_batches] += 1
        step_counts[valid_batches] += 1
        
    def addtogoal(goal_batches, goal_nodes):

        index_of_next_node = node_counts[goal_batches] - 1

        tree_parents[goal_batches, node_counts[goal_batches]] = index_of_next_node
        
        tree_positions[goal_batches, node_counts[goal_batches],:] = goal_nodes

        node_counts[goal_batches] += 1

    # GO INTO THE LOOP LOGIC #######################
    step_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
    p = 0.3

    active_batches = torch.ones(batch_size, dtype=torch.bool, device=device)
    all_goal_batches = torch.tensor([], dtype=torch.long, device=device)

    seen_flag = False

    while True:
        iter += 1
        # Define the random.random() tensor
        random_tensor = torch.rand(batch_size, device=device)

        # Define the boolean meask which compares each value in the random tensor to the probability variable
        mask = (random_tensor <= p) & active_batches

        # We update the mask to consider which batches are active where when mask returns true it means 
        # that we want to sample the goalnode and we can the node is active. If we wanna sample the goal 
        # node but the active_batches for some batch is false, then the target node remains as the sample?
        

        # Define the samples tensor which does non-uniform stampling
        samples = torch.empty((batch_size, 2), device=device)
        samples[:, 0] = torch.rand(batch_size, device=device) * (x1_max - x1_min) + x1_min
        samples[:, 1] = torch.rand(batch_size, device=device) * (x2_max - x2_min) + x2_min
        
        # Logic to check whether mask is true or not

        # Select rows in the random_tenser where mask is true. For those rows, copy the row from goal
        # and assign it to targetnode
        targetnode = torch.clone(samples)


        targetnode[mask] = goals[mask]

        # Now that we have the targetnode for each batch, we can move on to calculating what the next node is

        # we start by doing an 'unsqueeze' process on the targetnode. I don't really know how it works but this is
        # necessary for targetnode and tree_positions to be compatible for the subtraction
        targetnode_compat = targetnode.unsqueeze(1)

        # we then calculate the differenct between the tree_positions and targetnode_compat
        diff = targetnode_compat - tree_positions

        # now we have a difference tensor with the same size as tree_positions

        # next we square each element in the difference tensor
        squared_diff = diff**2

        # and then we take the square-root of the squared_diff tensor to get the distances
        # for inactive batches, the distance is infinity
        distances = torch.sqrt(torch.sum(squared_diff, dim=2))

        # Create a mask based on nodecounts where any index in distances beyond nodecounts has its distance set to
        # zero

        # this defines the number of nodes as a column vector. Shape is (1, NMAX)
        node_indices = torch.arange(NMAX, device=device).unsqueeze(0)

        # we then define a mask to test whether node_indices its less that node_counts.unsqueeze(1). Shape is (B, NMAX)
        # node indices is a list from 0 to NMAX-1
        # node_counts defines how many nodes are currently in the tree
        # valid_mask ensures that empty unseen node spaces in node_indices (the buffer up to the max number of nodes) are not included
        # in the distance calculation. Instead the distances for those node indices are set to infinity
        
        masked_node_counts = node_counts.clone()
        masked_node_counts[~active_batches] = 0 # zero out the inactive nodes
        
        # valid mask is used to say which values should be considered finding the smallest distance by taking 
        # into account whether the node number has been seen (we don't calculate distance for nodes that don't
        # yet exist). Masked node counts has 0 where the batch is inactive an integer where the batch is active

        # So for an inactive node, node_indices < masked_node_counts will always be False for inactive batches 
        # and so the distance will be set to infinity for all nodes in that batch

        # It also retains its original function where in active batches, any node count past the actual number of 
        # nodes in the tree is set to False
        valid_mask = node_indices < masked_node_counts.unsqueeze(1)

        # apply the mask to distances to check indices
        distances[~valid_mask] = float('inf')

        # after calculating the distance we can then look for the index where the distance is the smallest
        # this is of size (batch_size, 1)
        nearest_indices = torch.argmin(distances, dim=1) # Shape is (batch_size,)
        
        batch_indices = torch.arange(batch_size, device=device) # Shape is (batch_size,)

        # next we find the node (in tree positions) that corresponds to that index
        nearnode = tree_positions[batch_indices, nearest_indices, :] # this should be of shape (batch_size, 1, 2 -->(x,y))

        # next we get the minimum distance
        min_dist = distances[batch_indices, nearest_indices]

        # next we calculate the new x and y coordinates of the next node
        nextnode = nearnode + (STEP_SIZE/min_dist).unsqueeze(1) * diff[batch_indices, nearest_indices,:]
        # print(f'Nextnode is: {nextnode}')
        freespace_mask, timed_freespace = inFreespace(nextnode, starts, goals)

        count_infreespace += 1
        time_infreespace += timed_freespace

        connects_mask, timed_connects_to = connectsTo(nearnode, nextnode, starts, goals)

        count_connects_to += 1
        time_connects_to += timed_connects_to
    

        # next we need to combine the masks to see, for each batch, whether the node found is valid
        next_valid_mask = freespace_mask & connects_mask & active_batches

        # get the batches where the nextnode is valid
        valid_batches = torch.where(next_valid_mask)[0]

        # get the valid nextnodes
        valid_nextnodes = nextnode[valid_batches]
        if valid_nextnodes.shape[0] == 0:
            # print('No valid nodes found - stuck')
            stuck_counter += 1

        # get only the valid nearest indices
        valid_nearest = nearest_indices[valid_batches]

        # call addtotree for only the valid batches and valid nodes
        addtotree(valid_batches, valid_nextnodes, valid_nearest)
        
        if(iter % 500 == 0):
            print(f'Now at {iter} iterations')
            print(f'Node counts: {node_counts}')
            print(f'Step counts: {step_counts}')
            print(f'Active batches: {active_batches}')
            print(f'Stuck Counter: {stuck_counter}')

            # print(f'Number of infreespace calls: {count_infreespace}')
            # print(f'Average speed of infreespace calls: {time_infreespace/count_infreespace}')

            # print(f'Number of connectsto calls: {count_connects_to}')
            # print(f'Average speed of connectsto calls: {time_connects_to/count_connects_to}')

        if valid_nextnodes.shape[0] == 0:
            continue
        
        ##### GOAL CHECKING BLOCK #######

        # first make a mask to check whether the distance between the valid_nextnode and the goalnode
        # is within step_size

        # valid_nextnodes would be a subset of nextnodes which has shape valid_batches, (x,y)

        # Currently, the goalnode is of size (batch_size, (x,y)). Need to define valid_goal
        possible_goal = goals[valid_batches]

        # Find the Euclidean distance between the goal and each node in nextnode

        goal_diff = possible_goal - valid_nextnodes
        goal_distances = torch.norm(goal_diff, dim=1)
    

        within_threshold_mask = goal_distances < STEP_SIZE

        # Next we must create another mesh that tests whether the valid nextnode connects to the 
        # goalnode
        goal_connects_mask, timed_connects_to = connectsTo(valid_nextnodes, possible_goal, starts, goals)

        count_connects_to += 1
        time_connects_to += timed_connects_to

        valid_goal_connects_mask = within_threshold_mask & goal_connects_mask

        # get the batch indices where the goal can be added to the tree
        goal_batches = valid_batches[valid_goal_connects_mask]
        
        goal_nodes = goals[goal_batches]
        

        if goal_batches.numel()>0:
            addtogoal(goal_batches, goal_nodes)

            active_batches[goal_batches] = False
            all_goal_batches = torch.cat([all_goal_batches, goal_batches])

        ###FIXME: MOVING THIS TEMPORARILY TO SEE IF THIS CHANGES ANYTHING
        # step_counts[active_batches] += 1

        # Test to see if the step counts for any batch has exceeded the maximum steps
        step_mask = step_counts >= SMAX

        # Test to see if the node counts for any batch has exceeded the maximum nodes
        node_mask = node_counts >= NMAX

        # Create the overall expired mask
        expired_mask = step_mask | node_mask

        # Apply the expired mask to active batches
        active_batches[expired_mask] = False

        # And then the last thing is that I should break/end the loop if all the batches
        # have been stopped
        if False in active_batches and seen_flag == False:
            t_end_rrt = time.time()
            seen_flag = True
            t_rrt = t_end_rrt - t_begin_rrt
            print(f'First path found in time: {t_rrt} seconds')
            print(f'First path found at iteration: {iter}')

        if not active_batches.any():
            if (step_counts >= SMAX).all() | (node_counts >= NMAX).all():
                t_end_rrt = time.time()
                t_rrt = t_end_rrt - t_begin_rrt
                print(f'Process Aborted at Node Count = {node_counts} and \nStep Count = {step_counts}. No path found')
                print(f'RRT ended at {t_rrt} seconds')
                return t_rrt, None
            t_end_rrt = time.time()
            t_rrt = t_end_rrt - t_begin_rrt
            print(f'RRT ended at {t_rrt} seconds')
            break
        
        if iter > 10000:
            t_end_rrt = time.time()
            t_rrt = t_end_rrt - t_begin_rrt
            print(f'RRT ended at {t_rrt} seconds')
            break

    print(node_counts, step_counts)
    print(f'Stuck counter: {stuck_counter}')
    node_counts_cpu = node_counts.cpu().numpy()
    tree_parents_cpu = tree_parents.cpu().numpy()
    tree_positions_cpu = tree_positions.cpu().numpy()
    goal_indices = node_counts_cpu - 1
    all_goal_batches_cpu = all_goal_batches.cpu().numpy()
    # Let's send all this stuff to a function outside of the rrt called buildPath
    all_paths = buildPath(goal_indices, tree_parents_cpu, tree_positions_cpu, batch_size, all_goal_batches_cpu, starts, goals)
    
    print(f'The start points were: {starts}')
    print(f'The goal points were: {goals}')
    successful_paths = [p for p in all_paths if p != None]
    print(f'Number of paths found: {len(successful_paths)}')
    # print(f'These are the paths: {all_paths}')
    return t_rrt, all_paths

def buildPath(goal_idx, node_parents, node_positions, batch_size, goal_batches, starts, goals):
    
    # print('Now building the path')
    path = []
    # loop through each batch
    for batch_num in range(batch_size):
        if batch_num not in goal_batches:
            print(f'No path found for batch {batch_num}')
            path.append(None)
            continue
        current_path = []
        idx = goal_idx[batch_num]
        # print(idx)
        # Skip if no path was found:
        if idx < 0 or node_parents[batch_num, idx] == -1 and idx != 0:
            path.append(None)
            continue

        while idx != -1:
            # the parent node hasn't been encountered
            # add the node to the path
            current_path.append(node_positions[batch_num, idx])
            idx = node_parents[batch_num, idx]

        current_path.reverse()
        tup_list = [tuple(x for x in point) for point in current_path]
        node_list = []
        # let's just find the 'start' and 'goal' nodes here because something keeps getting messed up
        for (x, y) in tup_list:
            if tup_list.index((x, y)) == 0: # this means we have a start node
                # this means there exists an actual corresponding node
                corres_start = starts[batch_num]
                corres_node = [n for n in nodes if (n.coords[0] == corres_start[0]) and (n.coords[1] == corres_start[1])]
                corres_node = corres_node[0]
                node_list.append(corres_node)
            elif tup_list.index((x, y)) == len(tup_list) - 1:
                corres_goal = goals[batch_num]
                corres_node = [n for n in nodes if (n.coords[0] == corres_goal[0]) and (n.coords[1] == corres_goal[1])]
                corres_node = corres_node[0]
                node_list.append(corres_node)
            else:
                node_list.append(Node(None, [x, y]))
        
        path.append(node_list)
        
        # print('Checking path validity ahhhhh')
        # there's some weird rounding thing happening here. idk what it is :/
        
        for ele in range(len(tup_list) - 1):
            
            (x1, y1) = tup_list[ele]
            (x2, y2) = tup_list[ele+1]

            dist = sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return path

###############################################################################################
# DISTANCE HELPER FUNCTION

def distance(point1, point2):

    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]

    distance = sqrt((x1 - x2)**2 + (y1 - y2)**2) 
    return distance

#####################################################################################################
# DEFINE THE NODE CLASS

class Node:

    def __init__(self, polytopes, coords):

        self.polytopes = polytopes
        self.coords = coords
        self.neighbours = []

        self.parent = None
        self.cost = inf

        self.seen = False
        self.done = False

    
    def edge_cost(self, other):
        dist = sqrt((self.coords[0] - other.coords[0])**2 + (self.coords[1] - other.coords[1])**2)
        return dist 
        
    
    # print the node for debugging
    def __str__(self):
        return self.coords

    # Define the "less-than" to enable sorting by cost.
    def __lt__(self, other):
        return self.cost < other.cost
    
    # Equality checks
    def __eq__(self, other):
        return isinstance(other, Node) and self.coords == other.coords

    def __hash__(self):
        return hash(self.coords)

###############################################################################################################
# DEFINE NODE OBJECTS, NEIGHBOURS, AND EDGE COSTS

# first define every coordinate as a node object

# create the mega-list of all polytopes and all coords
t0_build = time.time()

# want to re-define our understanding of the nodes class so that it only considers the vertices and the polytopes
# we DON'T care about the centers, so we only loop through the polytopes
nodes = []

###############################################################################################
# RANDOMLY PLACE START AND GOAL NODES

# NEED TO INTRODUCE CONSTRAINT THAT THE START AND GOAL AREN'T GENERATED IN FREESPACE BUT IN AN ACTUAL POLYTOPE
t0_points = time.time()

start = [[21, 6]]
goal = [[2, 16]]

star = [21, 6]
goa = [2, 16]

# define the start and goal as nodes
startnode = Node(None, star)
goalnode = Node(None, goa)
tf_points = time.time()
time_points = tf_points - t0_points
print(f'Time taken to generate start/goal points: {time_points}')

nodes.append(startnode)
nodes.append(goalnode)
###############################################################################################

# also want to have a version of the obstacles as polyhdrons to prep for the inFreespace call
# so that this doesn't have to be continuously re-calculated

A_list = []
b_list = []
for obstacle in obstacles:

    as_H = HPolyhedron(obstacle)

    A_el = torch.tensor(as_H.A(), dtype=torch.float, device=device) # original size is (4, 2)
    b_el = torch.tensor(as_H.b(), dtype=torch.float, device=device) # original size is (4)

    # let's gather the A and b parameters into a list
    A_list.append(A_el)
    b_list.append(b_el)

# and then let's put them into a matrix stack
A_stack = torch.stack(A_list)
b_stack = torch.stack(b_list)

A_stack_T = A_stack.transpose(1, 2)

batch_num = 3
start_rrt = start * batch_num
goal_rrt = goal * batch_num

t_rrt, path_fixes = do_rrt(start_rrt, goal_rrt)

print(f'RRT results: {path_fixes}')

# Find the shortest (best) path
min_cost = 10000
min_path = []
for path in path_fixes:
    
    if path != None:
        cost = 0
        for idx in range(len(path) - 2):
            cost += distance(path[idx].coords, path[idx+1].coords)
        if cost < min_cost:
            min_cost = cost
            min_path = path
print(f'minimum path cost: {min_cost}')
print(f'minimum path: {min_path}')

# Post-process the shortest (best) path
i = 0
while (i < len(min_path)-2):
    if pointsConnect(min_path[i].coords, min_path[i+2].coords) == True:
        min_path.pop(i+1)
    else:
        i = i+1

print(f'path is: {min_path}')
print(f'Total time taken for RRT: {t_rrt} seconds')

# write the path to a csv file
with open('IRIS_env_drake/maze0_results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for ele in min_path:
        writer.writerow(ele.coords)


###################################################################################################################

# PLOTTING
plt.figure()

# plot the domain
domain_V = VPolytope(domain)
domain_pts = domain_V.vertices()
domain_pts = reorder_verts_2D(domain_pts)
plt.fill(domain_pts[0, :], domain_pts[1, :], 'white')

# plot the obstacles (the walls)
obs_rect1_pts = obs_rect1.vertices()
obs_rect1_pts = reorder_verts_2D(obs_rect1_pts)
plt.fill(obs_rect1_pts[0, :], obs_rect1_pts[1, :], 'r')

obs_rect2_pts = obs_rect2.vertices()
obs_rect2_pts = reorder_verts_2D(obs_rect2_pts)
plt.fill(obs_rect2_pts[0, :], obs_rect2_pts[1, :], 'r')

obs_rect3_pts = obs_rect3.vertices()
obs_rect3_pts = reorder_verts_2D(obs_rect3_pts)
plt.fill(obs_rect3_pts[0, :], obs_rect3_pts[1, :], 'r')

obs_rect4_pts = obs_rect4.vertices()
obs_rect4_pts = reorder_verts_2D(obs_rect4_pts)
plt.fill(obs_rect4_pts[0, :], obs_rect4_pts[1, :], 'r')

obs_rect5_pts = obs_rect5.vertices()
obs_rect5_pts = reorder_verts_2D(obs_rect5_pts)
plt.fill(obs_rect5_pts[0, :], obs_rect5_pts[1, :], 'r')

obs_rect6_pts = obs_rect6.vertices()
obs_rect6_pts = reorder_verts_2D(obs_rect6_pts)
plt.fill(obs_rect6_pts[0, :], obs_rect6_pts[1, :], 'r')

obs_rect7_pts = obs_rect7.vertices()
obs_rect7_pts = reorder_verts_2D(obs_rect7_pts)
plt.fill(obs_rect7_pts[0, :], obs_rect7_pts[1, :], 'r')

obs_rect8_pts = obs_rect8.vertices()
obs_rect8_pts = reorder_verts_2D(obs_rect8_pts)
plt.fill(obs_rect8_pts[0, :], obs_rect8_pts[1, :], 'r')

obs_rect9_pts = obs_rect9.vertices()
obs_rect9_pts = reorder_verts_2D(obs_rect9_pts)
plt.fill(obs_rect9_pts[0, :], obs_rect9_pts[1, :], 'r')

obs_rect10_pts = obs_rect10.vertices()
obs_rect10_pts = reorder_verts_2D(obs_rect10_pts)
plt.fill(obs_rect10_pts[0, :], obs_rect10_pts[1, :], 'r')

obs_rect11_pts = obs_rect11.vertices()
obs_rect11_pts = reorder_verts_2D(obs_rect11_pts)
plt.fill(obs_rect11_pts[0, :], obs_rect11_pts[1, :], 'r')

obs_rect12_pts = obs_rect12.vertices()
obs_rect12_pts = reorder_verts_2D(obs_rect12_pts)
plt.fill(obs_rect12_pts[0, :], obs_rect12_pts[1, :], 'r')

obs_rect13_pts = obs_rect13.vertices()
obs_rect13_pts = reorder_verts_2D(obs_rect13_pts)
plt.fill(obs_rect13_pts[0, :], obs_rect13_pts[1, :], 'r')

obs_rect14_pts = obs_rect14.vertices()
obs_rect14_pts = reorder_verts_2D(obs_rect14_pts)
plt.fill(obs_rect14_pts[0, :], obs_rect14_pts[1, :], 'r')

obs_rect15_pts = obs_rect15.vertices()
obs_rect15_pts = reorder_verts_2D(obs_rect15_pts)
plt.fill(obs_rect15_pts[0, :], obs_rect15_pts[1, :], 'r')

obs_rect16_pts = obs_rect16.vertices()
obs_rect16_pts = reorder_verts_2D(obs_rect16_pts)
plt.fill(obs_rect16_pts[0, :], obs_rect16_pts[1, :], 'r')

obs_rect17_pts = obs_rect17.vertices()
obs_rect17_pts = reorder_verts_2D(obs_rect17_pts)
plt.fill(obs_rect17_pts[0, :], obs_rect17_pts[1, :], 'r')

obs_rect18_pts = obs_rect18.vertices()
obs_rect18_pts = reorder_verts_2D(obs_rect18_pts)
plt.fill(obs_rect18_pts[0, :], obs_rect18_pts[1, :], 'r')

obs_rect19_pts = obs_rect19.vertices()
obs_rect19_pts = reorder_verts_2D(obs_rect19_pts)
plt.fill(obs_rect19_pts[0, :], obs_rect19_pts[1, :], 'r')

obs_rect20_pts = obs_rect20.vertices()
obs_rect20_pts = reorder_verts_2D(obs_rect20_pts)
plt.fill(obs_rect20_pts[0, :], obs_rect20_pts[1, :], 'r')

obs_rect21_pts = obs_rect21.vertices()
obs_rect21_pts = reorder_verts_2D(obs_rect21_pts)
plt.fill(obs_rect21_pts[0, :], obs_rect21_pts[1, :], 'r')

obs_rect22_pts = obs_rect22.vertices()
obs_rect22_pts = reorder_verts_2D(obs_rect22_pts)
plt.fill(obs_rect22_pts[0, :], obs_rect22_pts[1, :], 'r')

obs_rect23_pts = obs_rect23.vertices()
obs_rect23_pts = reorder_verts_2D(obs_rect23_pts)
plt.fill(obs_rect23_pts[0, :], obs_rect23_pts[1, :], 'r')

obs_rect24_pts = obs_rect24.vertices()
obs_rect24_pts = reorder_verts_2D(obs_rect24_pts)
plt.fill(obs_rect24_pts[0, :], obs_rect24_pts[1, :], 'r')

obs_rect25_pts = obs_rect25.vertices()
obs_rect25_pts = reorder_verts_2D(obs_rect25_pts)
plt.fill(obs_rect25_pts[0, :], obs_rect25_pts[1, :], 'r')

obs_rect26_pts = obs_rect26.vertices()
obs_rect26_pts = reorder_verts_2D(obs_rect26_pts)
plt.fill(obs_rect26_pts[0, :], obs_rect26_pts[1, :], 'r')

obs_rect27_pts = obs_rect27.vertices()
obs_rect27_pts = reorder_verts_2D(obs_rect27_pts)
plt.fill(obs_rect27_pts[0, :], obs_rect27_pts[1, :], 'r')

obs_rect28_pts = obs_rect28.vertices()
obs_rect28_pts = reorder_verts_2D(obs_rect28_pts)
plt.fill(obs_rect28_pts[0, :], obs_rect28_pts[1, :], 'r')

obs_rect29_pts = obs_rect29.vertices()
obs_rect29_pts = reorder_verts_2D(obs_rect29_pts)
plt.fill(obs_rect29_pts[0, :], obs_rect29_pts[1, :], 'r')

obs_rect30_pts = obs_rect30.vertices()
obs_rect30_pts = reorder_verts_2D(obs_rect30_pts)
plt.fill(obs_rect30_pts[0, :], obs_rect30_pts[1, :], 'r')

obs_rect31_pts = obs_rect31.vertices()
obs_rect31_pts = reorder_verts_2D(obs_rect31_pts)
plt.fill(obs_rect31_pts[0, :], obs_rect31_pts[1, :], 'r')

obs_rect32_pts = obs_rect32.vertices()
obs_rect32_pts = reorder_verts_2D(obs_rect32_pts)
plt.fill(obs_rect32_pts[0, :], obs_rect32_pts[1, :], 'r')

# plot the start and goal nodes
plt.plot(start[0][0], start[0][1], 'mo')
plt.plot(goal[0][0], goal[0][1], 'mo')
path_fix_cols = ['black', 'blue', 'magenta', 'green', 'orange']
# plot the path(s)
col_idx = 0
for path in path_fixes:
    if path != None:
        path_len = len(path)
        idx = 0
        while idx != path_len - 1:
            x_vals = [path[idx].coords[0], path[idx+1].coords[0]]
            y_vals = [path[idx].coords[1], path[idx+1].coords[1]]
            plt.plot(x_vals, y_vals, path_fix_cols[col_idx % len(path_fix_cols)])
            idx += 1
    col_idx += 1

idx = 0
while idx != len(min_path) - 1:
    x_vals = [min_path[idx].coords[0], min_path[idx+1].coords[0]]
    y_vals = [min_path[idx].coords[1], min_path[idx+1].coords[1]]
    plt.plot(x_vals, y_vals, 'yellow')
    idx += 1


plt.axis('equal')
plt.show()
