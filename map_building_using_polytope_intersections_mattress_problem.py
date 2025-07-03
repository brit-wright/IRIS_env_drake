#!/usr/bin/env python3.10
import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import *#!/usr/bin/env python3.8
import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import *#!/usr/bin/env python3.8
import numpy as np
import matplotlib.pyplot as plt

from math import sqrt

import shapely

import random

from pydrake.all import *

###############################################################################################
# Create a seed
# seed = int(random.random()*10000)
seed = 554
random.seed(seed)
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
rect_pts1 = np.array([[0, 8],
                     [0, 10],
                     [15, 8],
                     [15, 10]])

rect_pts2 = np.array([[18, 8],
                     [18, 10],
                     [30, 8],
                     [30, 10]])

rect_pts3 = np.array([[13, 20],
                      [15, 20],
                      [15, 12],
                      [13, 12]])

rect_pts4 = np.array([[12, 12],
                      [13, 12],
                      [13, 14],
                      [12, 14]])

rect_pts5 = np.array([[18, 12],
                      [18, 14],
                      [21, 14],
                      [21, 12]])

rect_pts6 = np.array([[13, 0], 
                    [15, 0],
                    [10, 5],
                    [12, 5]])

obs_rect1 = VPolytope(rect_pts1.T)
obs_rect2 = VPolytope(rect_pts2.T)
obs_rect3 = VPolytope(rect_pts3.T)
obs_rect4 = VPolytope(rect_pts4.T)
obs_rect5 = VPolytope(rect_pts5.T)
obs_rect6 = VPolytope(rect_pts6.T)


###############################################################################################
# DISTANCE HELPER FUNCTION

def distance(point1, point2):

    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]

    distance = sqrt((x1 - x2)**2 + (y1 - y2)**2) 
    return distance

###############################################################################################
# IRIS ALGORITHM

# list of all the obstalces
obstacles = [obs_rect1, obs_rect2, obs_rect3, obs_rect4, obs_rect5, obs_rect6]

# choose a sample intial point to do optimization from

sample_pts = []

# let's do 3 sample points

num_samples = 100

for pt in range(num_samples):
    sample_pt = np.array([np.random.uniform(x1_min, x1_max), np.random.uniform(x2_min, x2_max)])

    while (obs_rect1.PointInSet(sample_pt) or obs_rect2.PointInSet(sample_pt) or obs_rect3.PointInSet(sample_pt) 
    or obs_rect4.PointInSet(sample_pt) or obs_rect5.PointInSet(sample_pt) or obs_rect6.PointInSet(sample_pt)):
        sample_pt = np.array([np.random.uniform(x1_min, x1_max), np.random.uniform(x2_min, x2_max)])
        
    sample_pts.append(sample_pt)

# iris options
options = IrisOptions()
options.termination_threshold = 1e-3
options.iteration_limit = 200
options.configuration_obstacles = obstacles

refined_samples_list = []
r_H_list = []
center_list = []
vertex_list = []

for alg_num in range(num_samples):
    # run the algorithm
    r_H = Iris(obstacles, # obstacles list
            sample_pts[alg_num], # sample point, (intial condition)
            domain,    # domain of the problem
            options)   # options

    cheb_center = r_H.ChebyshevCenter()
    cheb_c = cheb_center.tolist()
    [x, y] = [int(cheb_c[0]), int(cheb_c[1])]
    
    if [x, y] in center_list:
        continue

    vertex_list.append(VPolytope(r_H).vertices())
    # print(f'Vertices: {VPolytope(r_H).vertices()}')
    center_list.append([x, y])
    r_H_list.append(r_H)
    refined_samples_list.append(sample_pts[alg_num])

###############################################################################################
# PERFORMING INTERPOLATION BETWEEN TWO POINTS

def check_obstacle_collision(coord, obstacles):
    
    for v_pol in obstacles:
        # do an early check for whether the point intersects
        if v_pol.PointInSet(coord) == True:
            return True
    return False
            
def create_coords(x1, y1, x2, y2):
    coord_list = []
    # create coords using linear interpolation

    # check that the lines isn't parallel and apply the linear
    # interpolation formula

    xmin = min(x1, x2)
    xmax = max(x1, x2)

    if x1 == xmin:
        y_xmin = y1
        y_xmax = y2
    else:
        y_xmin = y2
        y_xmax = y1

    if x1 != x2:

        for x_point in np.arange(xmin, xmax, 0.25):
            y_point = y_xmin + (y_xmax - y_xmin)/(xmax - xmin) * (x_point - xmin)
            coord_list.append((float(x_point), float(y_point)))

    else:

        y_min = min(y1, y2)
        y_max = max(y1, y2)

        for y_point in np.arange(y_min, y_max, 0.25):
            coord_list.append((x1, float(y_point)))
    # print(coord_list)
    return coord_list

# print('Connections')
# print(connections)


# For connecting intersection polytopes using chebyshev centers

# returns true if the points can connect without intersection
def check_interpolation(pair, obstacles):

    x1, y1 = pair[0][0], pair[0][1]
    x2, y2 = pair[1][0], pair[1][1]

    coord_checks = create_coords(x1, y1, x2, y2)

    for coord in coord_checks:

        interpolation_intersects = check_obstacle_collision(coord, obstacles)
        
        if interpolation_intersects == True:
            return False

    return True

###############################################################################################
# POLYTOPE INTERSECTION CHECKER


# note that center_list, r_H_list and refined_samples list already correspond to each other 
# indexing such that for each polytope, we already have its chebyshev center. 

# thus, as we cycle through hedron_idx, we know the index correspondence between polytope and 
# chebyshev center

center_pairs = [] # polytope pairs stores the pairs of chebyshev centers that have overlapping polytopes
polytope_pairs = []
inters_centers = []
inters_list = []
inters_list_hedron = []

# check for intersections between the polyhedrons
for hedron_idx in range(len(r_H_list)):
    hedron_counter = 0
    for hedron in r_H_list:
        if hedron != r_H_list[hedron_idx]:
            # check for intersection
            inters = hedron.Intersection(r_H_list[hedron_idx])  # this returns a polyhedron

            inters_list.append(VPolytope(inters))
            inters_list_hedron.append(inters)
            # the elements of the inters list are of type VPolytope. However, the actual vertices will be an empty
            # list for some. This is handled in the plotting section. 
            center_pairs.append([center_list[hedron_idx], center_list[hedron_counter]])
            polytope_pairs.append([VPolytope(r_H_list[hedron_idx]), VPolytope(r_H_list[hedron_counter])])
        hedron_counter += 1


refined_center_pairs = []
refined_inters_list = []
refined_polytope_pairs = []
refined_inters_centers = []

# post-process the polytope_pairs and inters_list entries to only consider the valid intersections
refine_index = 0
for inter_el in inters_list:
    inter_vertex = inter_el.vertices()
    if inter_vertex.size > 0:

        refined_center_pairs.append(center_pairs[refine_index])
        refined_polytope_pairs.append(polytope_pairs[refine_index])
        refined_inters_list.append(inters_list[refine_index])

        # Calculate the Chebyshev center of that intersection
        inters_cheb_center = inters_list_hedron[refine_index].ChebyshevCenter()
        inters_cheb_c = inters_cheb_center.tolist()
        [x_inter, y_inter] = [int(inters_cheb_c[0]), int(inters_cheb_c[1])]

        refined_inters_centers.append([x_inter, y_inter])

    refine_index += 1

###############################################################################################
# BUILD THE GRAPH

# Let's go into building the graph using the form G = (V, E)

# V represents the vertices --> In this case, each vertex is a polytope
# E represents the edges --> In this case, each edge is the intersection between two polytopes

# I currently have 'refined_center_pairs' of the form  [[P1, P2], [P3, P4], [P5, P6]] 
# and the corresponding center pairs

# Now, let's construct the edges 

# Okay, let's construct the graph of the form G = [[Polytope_Pair, Intersecting_Polytope],  [Polytope_Pair, Intersecting_Polytope]]
# And let's have a similar graph for just the Chebyshev centers g = [[Polytope_Pair_Centers, Intersecting_Polytope_Center],  [Polytope_Pair_Centers, Intersecting_Polytope_Center]]

# Requirement 1: Length of the polytope pair list = length of intersection list = length of polytope pair centers list = length of intersection center list

print(f'Length of polytope pair list: {len(refined_polytope_pairs)}')
print(f'Length of center pairs list: {len(refined_center_pairs)}')
print(f'Length of intersection list: {len(refined_inters_list)}')
print(f'Length of intersection centers list: {len(refined_inters_centers)}')

# Since this requirement has been passed, we can go into drawing the tree

# This is done in the plotting section where basically, for every pair of center points, there's
# a corresponding intersection center. So we do a plt.plot to connect the points


# Next, we need to assign node-node edge costs
edge_cost_list = []

for pair in refined_center_pairs:
    edge_cost = distance(pair[0], pair[1])
    edge_cost_list.append(edge_cost)
   
print(f'Length of edge cost list: {len(edge_cost_list)}')

print(refined_center_pairs)
print(refined_inters_centers)


# What data type is the graph????


# next we can go on to defining the start and goal nodes and attaching them to the tree

###############################################################################################
# RANDOMLY PLACE START AND GOAL NODES

start = [np.random.uniform(x1_min, x1_max), np.random.uniform(x2_min, x2_max)]

while check_obstacle_collision(start, obstacles) == True: # the start node intersects and obstacles
    start = [np.random.uniform(x1_min, x1_max), np.random.uniform(x2_min, x2_max)]

goal = [np.random.uniform(x1_min, x1_max), np.random.uniform(x2_min, x2_max)]

while (goal == start) or (check_obstacle_collision(goal, obstacles) == True) or (distance(start, goal) < (x1_max - x1_min)/2):
    goal = [np.random.uniform(x1_min, x1_max), np.random.uniform(x2_min, x2_max)]

###############################################################################################
# CONNECT THE START AND GOAL NODE TO THE TREE AND CALCULATE THE EDGE COSTS
 
# Connect the start node
dist = 100 # just a dummy numb er
for coord in center_list:
    test_dist = distance(start, coord)
    if test_dist < dist and check_interpolation([start, coord], obstacles):
        start_neighbour = coord
        dist = test_dist
start_cost = dist

dist = 100
for coord in center_list:
    test_dist = distance(goal, coord)
    if test_dist < dist and check_interpolation([goal, coord], obstacles):
        goal_neighbour = coord
        dist = test_dist
goal_cost = dist

print(start_neighbour)
print(goal_neighbour)

###############################################################################################
# BUILD THE PATH

path = [start, start_neighbour]

curr = start_neighbour
stuck = False
goal_found = False
num_iter = 0
while curr != goal_neighbour and stuck == False and num_iter < 40:

    # check whether curr is in the intersection list or the center pairs list

    
    ind_list = []
    candidates = []
    cost_list = []
    for connection in refined_center_pairs:
        if (curr in connection) and (connection[1 - connection.index(curr)] not in path):
            ind_list.append(refined_center_pairs.index(connection))
            candidates.append(connection[1 - connection.index(curr)])
            cost_list.append(edge_cost_list[refined_center_pairs.index(connection)])
    
    if ind_list == []:
        if curr in refined_inters_centers:
            # get the index of refined_inters_centers
            neighbours_index = refined_inters_centers.index(curr)

            # get the actual neighbour nodes
            neighbour1 = refined_center_pairs[neighbours_index][0]
            neighbour2 = refined_center_pairs[neighbours_index][1]

            # check which is the better neighbour based on cost
            if distance(curr, neighbour1) <= distance(curr, neighbour2) and neighbour1 not in path:
                # append this neighbour to the path and set it to current
                path.append(neighbour1)
                curr = neighbour1
            elif neighbour2 not in path:
                # use the other neighbour
                path.append(neighbour2)
                curr = neighbour2
        else:
            stuck = True
            break
    else:
        path.append(candidates[np.argmin(cost_list)])
        curr = candidates[np.argmin(cost_list)]

    num_iter += 1

    if curr == goal_neighbour:
        path.append(goal)

if goal in path:
    print(f'Goal Found. Path is: {path}')
    goal_found = True

else:
    print(f'Goal not found. Incomplete path: {path}')

###############################################################################################
# PLOTTING
plt.figure()

# plot the domain
domain_V = VPolytope(domain)
domain_pts = domain_V.vertices()
domain_pts = reorder_verts_2D(domain_pts)
plt.fill(domain_pts[0, :], domain_pts[1, :], 'gray')


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

# # plot the IRIS answers 
# for answer in range(len(r_H_list)):
#     r_V = VPolytope(r_H_list[answer])
#     r_pts = r_V.vertices()
#     r_pts = reorder_verts_2D(r_pts)
#     plt.fill(r_pts[0, :], r_pts[1, :], 'g')


# plot the Chebyshev centers
# print('Centers')
for center in center_list:
    plt.plot(center[0], center[1], 'bo')

# for point in refined_inters_centers:
#     plt.plot(point[0], point[1], 'wo')

# # plot the connections
# for connection in connections:
#     x_values = [connection[0][0], connection[1][0]]
#     y_values = [connection[0][1], connection[1][1]]

#     plt.plot(x_values, y_values, 'b')

# plot the start and goal nodes
plt.plot(start[0], start[1], 'mo')
plt.plot(goal[0], goal[1], 'mo')


# # plot the start and goal to center connections
# plt.plot([start[0], best_start_center[0]], [start[1], best_start_center[1]], 'm')
# plt.plot([goal[0], best_goal_center[0]], [goal[1], best_goal_center[1]], 'm')


# plot using the vertices 
# vertex_list has shape (num_v_polytopes,  num_dims (= 2 which are x, y), num_vertices)

# first we need to loop through each polytope:
# print(len(vertex_list))
colour_list = ['orange', 'turquoise', 'indianred', 'darkseagreen', 'palevioletred', 'goldenrod', 'forestgreen', 'mediumpurple', 'peru', 'rosybrown', 'orange', 'turquoise', 'indianred', 'darkseagreen']
idx = 0
for group_verts in vertex_list:
    group_verts = reorder_verts_2D(group_verts)
    # print('The points')
    # print(group_verts[0, :], group_verts[1, :])
    plt.plot(group_verts[0, :], group_verts[1, :], 'grey')
    plt.fill(group_verts[0, :], group_verts[1, :], colour_list[idx])
    idx += 1

for inter in inters_list:
    # group_inters = reorder_verts_2D(inter.vertices())

    inter_vertex = inter.vertices()
    # print(f'Inter Vertex: {inter_vertex}')

    if inter_vertex.size > 0:
        group_inters = reorder_verts_2D(inter_vertex)
        plt.fill(group_inters[0,:], group_inters[1,:], 'lime')

    # plt.fill(group_inters[0, :], group_inters[1, :], 'blue')

# plot the center-pairs
# for pair in refined_center_pairs:
#     x_vals = [pair[0][0], pair[1][0]]
#     y_vals = [pair[0][1], pair[1][1]]

#     # if check_interpolation(pair, obstacles) == True:
#     #     plt.plot(x_vals, y_vals, 'black')

#     plt.plot(x_vals, y_vals, 'black')

pair_idx = 0
for pair in refined_center_pairs:
    x_vals1 = [pair[0][0], refined_inters_centers[pair_idx][0]]
    y_vals1 = [pair[0][1], refined_inters_centers[pair_idx][1]]
    
    x_vals2 = [pair[1][0], refined_inters_centers[pair_idx][0]]
    y_vals2 = [pair[1][1], refined_inters_centers[pair_idx][1]]

    plt.plot(x_vals1, y_vals1, 'black')
    plt.plot(x_vals2, y_vals2, 'black')

    pair_idx += 1

# plot the path if the goal is found
length = len(path)
path_idx = 0

while path_idx != length-1:

    inter_exists = False

    # also want to plot the intermediate centers
    
    if ([path[path_idx], path[path_idx+1]] in refined_center_pairs):
        inter_exists = True
        idx = refined_center_pairs.index([path[path_idx], path[path_idx+1]])
    elif ([path[path_idx+1], path[path_idx]] in refined_center_pairs):
        inter_exists = True
        idx = refined_center_pairs.index([path[path_idx+1], path[path_idx]])

    if inter_exists == False:
        x_path = [path[path_idx][0], path[path_idx+1][0]]
        y_path = [path[path_idx][1], path[path_idx+1][1]]

        plt.plot(x_path, y_path, 'white')

    else:
        x_path1 = [path[path_idx][0], refined_inters_centers[idx][0]]
        x_path2 = [path[path_idx + 1][0], refined_inters_centers[idx][0]]

        y_path1 = [path[path_idx][1], refined_inters_centers[idx][1]]
        y_path2 = [path[path_idx + 1][1], refined_inters_centers[idx][1]]

        plt.plot(x_path1, y_path1, 'white')
        plt.plot(x_path2, y_path2, 'white')

    path_idx += 1

plt.axis('equal')
plt.show()


