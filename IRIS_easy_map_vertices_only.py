#!/usr/bin/env python3.10
import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import *#!/usr/bin/env python3.8
import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import *#!/usr/bin/env python3.8
import numpy as np
import matplotlib.pyplot as plt

from math import sqrt, inf

import shapely

import bisect

import random

import time

from pydrake.all import *

###############################################################################################
# Create a seed
# seed = int(random.random()*10000)
# seed = 554
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
rect1_pts = np.array([[5, 15],
                      [10, 15],
                      [5, 5],
                      [10, 5]])

rect2_pts = np.array([[15, 10],
                      [20, 10],
                      [15, 5],
                      [20, 5]])

rect3_pts = np.array([[16, 15],
                      [25, 15],
                      [16, 11],
                      [25, 11]])

obs_rect1 = VPolytope(rect1_pts.T)

obs_rect2 = VPolytope(rect2_pts.T)

obs_rect3 = VPolytope(rect3_pts.T)

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
obstacles = [obs_rect1, obs_rect2, obs_rect3]

# choose a sample intial point to do optimization from

sample_pts = []

# let's do 3 sample points

num_samples = 50

for pt in range(num_samples):
    sample_pt = np.array([np.random.uniform(x1_min, x1_max), np.random.uniform(x2_min, x2_max)])

    while (obs_rect1.PointInSet(sample_pt)):
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
vertex_dict = {}
vertex_list = []
all_pols = []
t0 = time.time()
for alg_num in range(num_samples):
    # run the algorithm

    # turning this into a try except, because sometimes Iris can't solve the optimization
    # problem

    try:

        r_H = Iris(obstacles, # obstacles list
                sample_pts[alg_num], # sample point, (intial condition)
                domain,    # domain of the problem
                options)   # options
        
    except RuntimeError as e:
        print("IRIS failed at this seed. Skipping. Error:", e)

    cheb_center = r_H.ChebyshevCenter()
    cheb_c = cheb_center.tolist()
    [x, y] = [round(cheb_c[0], 6), round(cheb_c[1], 6)]
    
    if [x, y] in center_list:
        continue
    all_pols.append(VPolytope(r_H))
    vertex_list.append(VPolytope(r_H).vertices()) # this stores the vertices associated with each IRIS solution
    # print(f'Vertices: {VPolytope(r_H).vertices()}')
    center_list.append([x, y])
    r_H_list.append(r_H)
    refined_samples_list.append(sample_pts[alg_num])

    vertex_dict[(x, y)] = VPolytope(r_H).vertices()

# print("Checkpoint 1")
# print(vertex_dict)
# print(f'Vertex List is: {vertex_list}')
tf = time.time()
t_IRIS = tf - t0
print(f'Time taken to solve IRIS: {t_IRIS}')
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

        for x_point in np.arange(xmin, xmax, 0.01):
            y_point = y_xmin + (y_xmax - y_xmin)/(xmax - xmin) * (x_point - xmin)
            coord_list.append((float(x_point), float(y_point)))

    else:

        y_min = min(y1, y2)
        y_max = max(y1, y2)

        for y_point in np.arange(y_min, y_max, 0.01):
            coord_list.append((x1, float(y_point)))
    # print(coord_list)
    return coord_list

# For connecting intersection polytopes using chebyshev centers

# returns true if the points can connect without intersection
def check_interpolation(pair, obstacles):

    x1, y1 = pair[0][0], pair[0][1]
    x2, y2 = pair[1][0], pair[1][1]

    coord_checks = create_coords(x1, y1, x2, y2)

    obstacle_vertex_list = []
    for opp in obstacles:
        obstacle_vertex_list.append(opp.vertices())

    for coord in coord_checks:

        coord = [[coord[0]], [coord[1]]]  

        is_a_vertex = False


        # check whether the current coordinate is a vertex
        for obs_vertex in obstacle_vertex_list:

            for ob_index in range(len(obs_vertex[0])):

                if coord == [obs_vertex[0][ob_index], obs_vertex[1][ob_index]]:

                    is_a_vertex = True

        if is_a_vertex == False:

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
t0_intersect = time.time()
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
    inter_vertex = inter_el.vertices() # stores the vertices for each 'possible' intersection. includes empties as well
    if inter_vertex.size > 0:

        refined_center_pairs.append(center_pairs[refine_index])
        refined_polytope_pairs.append(polytope_pairs[refine_index])
        refined_inters_list.append(inters_list[refine_index])

        # Calculate the Chebyshev center of that intersection
        inters_cheb_center = inters_list_hedron[refine_index].ChebyshevCenter()
        inters_cheb_c = inters_cheb_center.tolist()
        [x_inter, y_inter] = [round(inters_cheb_c[0], 6), round(inters_cheb_c[1], 6)]

        refined_inters_centers.append([x_inter, y_inter])

        if (x_inter, y_inter) not in vertex_dict:
            vertex_dict[(x_inter, y_inter)] = inter_vertex
        else:
            # print(f'True. Center is {(x_inter, y_inter)}')
            # print(f'and vertices to be added are {inter_vertex}')
            # print(f'Before: {vertex_dict}')
            vertex_dict[(x_inter, y_inter)] = np.append(vertex_dict[(x_inter, y_inter)], inter_vertex, axis = 1)
            # print(f'After: {vertex_dict}')

    refine_index += 1
tf_intersect = time.time()
time_intersect = tf_intersect - t0_intersect
print(f'Time taken to check intersections: {time_intersect}')
# print("Checkpoint 2")
# print(vertex_dict)

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

# Since this requirement has been passed, we can go into drawing the tree

# This is done in the plotting section where basically, for every pair of center points, there's
# a corresponding intersection center. So we do a plt.plot to connect the points

# So basically need to update my implementation. Each polytope and polytope intersection is supposed to be its own node lol.
# Each node will also have its own list of neighbours. Parents are initialized to nothing. 


# need to make a mega list in order to convert all the current intersections into nodes as well
# and to label the neighbours

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

###############################################################################################################
# DEFINE NODE OBJECTS, NEIGHBOURS, AND EDGE COSTS

# first define every coordinate as a node object

# create the mega-list of all polytopes and all coords
t0_build = time.time()
mega_topes = []
mega_coords = []

for element in refined_center_pairs:
    for coord in element:
        mega_coords.append(coord)
        mega_topes.append([refined_polytope_pairs[refined_center_pairs.index(element)][element.index(coord)]])

mega_coords = mega_coords + refined_inters_centers
mega_topes = mega_topes + refined_polytope_pairs

#################################################################################################################
# USING A DICTIONARY - THIS IS SO GNARLY

# Loop through all the elements of vertex_dict and look for the vertices

nodes = []

already_processed = []

for entry in vertex_dict: # entry is the key

    neighbour_mixer = []

    items = vertex_dict[entry] # gets the vertices associated with that particular key
    
    # next loop through the items and assign each item as a node

    for item_idx in range(len(items[0])):

        node_coord = [items[0][item_idx], items[1][item_idx]]

        if node_coord not in already_processed:

            node = Node(None, node_coord)

            nodes.append(node)

            already_processed.append(node_coord)

        neighbour_mixer.append(node)

    # assign the neighbours

    for neigh in neighbour_mixer:
        for ne_index in range(len(neighbour_mixer)):
            if neigh != neighbour_mixer[ne_index]:
                neigh.neighbours.append(neighbour_mixer[ne_index])


tf_build = time.time()
time_build = tf_build - t0_build
print(f'Time taken to build the graph: {time_build}')

###############################################################################################
# RANDOMLY PLACE START AND GOAL NODES
t0_points = time.time()

start = [np.random.uniform(x1_min, x1_max), np.random.uniform(x2_min, x2_max)]

# start = [12.474357775465961, 2.939952153423444]

while check_obstacle_collision(start, obstacles) == True: # the start node intersects and obstacles
    start = [round(np.random.uniform(x1_min, x1_max), 6), round(np.random.uniform(x2_min, x2_max), 6)]

goal = [np.random.uniform(x1_min, x1_max), np.random.uniform(x2_min, x2_max)]

while (goal == start) or (check_obstacle_collision(goal, obstacles) == True) or (distance(start, goal) < (x1_max - x1_min)/2):
    goal = [round(np.random.uniform(x1_min, x1_max), 6), round(np.random.uniform(x2_min, x2_max), 6)]

# define the start and goal as nodes
startnode = Node(None, start)
goalnode = Node(None, goal)
tf_points = time.time()
time_points = tf_points - t0_points
print(f'Time taken to generate start/goal points: {time_points}')

nodes.append(startnode)
nodes.append(goalnode)
###############################################################################################
# DEFINE THE NEIGHBOURS OF THE START AND GOAL NODE
t0_points_neighbours = time.time()

start_neigh_list = []
goal_neigh_list = []

# refined_inter_list is the list of intersection polytopes
for pol in refined_inters_list:
    if pol.PointInSet(start):

        # going to re-define startnode.polytopes
        start_neigh_list.append(pol)

        # need to find the chebyshev center associated with that polytope
        ind = refined_inters_list.index(pol)
        
        center_assoc = refined_inters_centers[ind]

        # then look in vertex dict for the vertices associated with that center

        vert_list = vertex_dict[(center_assoc[0], center_assoc[1])]
        # and then for each vertex in vert_list, make the neighbour assignment
        for vert in range(len(vert_list[0])):

            neigh = [n for n in nodes if n.coords == [vert_list[0][vert], vert_list[1][vert]]][0]

            if neigh != []:

                if neigh not in startnode.neighbours:

                    neigh.neighbours.append(startnode)
                    startnode.neighbours.append(neigh)


    if pol.PointInSet(goal):

        # going to re-define goalnode.polytopes
        goal_neigh_list.append(pol)

        # need to find the chebyshev center associated with that polytope
        ind = refined_inters_list.index(pol)
        
        center_assoc = refined_inters_centers[ind]

        # then look in vertex dict for the vertices associated with that center

        vert_list = vertex_dict[(center_assoc[0], center_assoc[1])]
        # and then for each vertex in vert_list, make the neighbour assignment
        for vert in range(len(vert_list[0])):

            neigh = [n for n in nodes if n.coords == [vert_list[0][vert], vert_list[1][vert]]][0]

            if neigh != []:

                if neigh not in goalnode.neighbours:

                    neigh.neighbours.append(goalnode)
                    goalnode.neighbours.append(neigh)

# find the neighbours of the startnode based on location in IRIS polytopes
for pol in all_pols:
    if pol.PointInSet(start):
        start_neigh_list.append(pol)

        # all_pols and vertex_list have the same vertices. Therefore, just need to assign based on indices
        ind = all_pols.index(pol)

        vert_list = vertex_list[ind]

        # assign neighbours
        for vert in range(len(vert_list[0])):
            neigh = [n for n in nodes if n.coords == [vert_list[0][vert], vert_list[1][vert]]][0]

            if neigh != []:

                if neigh not in startnode.neighbours:

                    neigh.neighbours.append(startnode)
                    startnode.neighbours.append(neigh)

    if pol.PointInSet(goal):
        goal_neigh_list.append(pol)

        # all_pols and vertex_list have the same vertices. Therefore, just need to assign based on indices
        ind = all_pols.index(pol)

        indices = [ind for (ind, all_pol) in enumerate(all_pols) if all_pol == pol]

        # print(f'Using method 2, ind id {indices} ')

        vert_list = vertex_list[ind]

        # assign neighbours
        for vert in range(len(vert_list[0])):
            neigh = [n for n in nodes if n.coords == [vert_list[0][vert], vert_list[1][vert]]][0]

            if neigh != []:

                if neigh not in goalnode.neighbours:

                    neigh.neighbours.append(goalnode)
                    goalnode.neighbours.append(neigh)

startnode.polytopes = start_neigh_list
goalnode.polytopes = goal_neigh_list

if (check_interpolation([start, goal], obstacles) == True):
    goalnode.neighbours.append(startnode)
    startnode.neighbours.append(goalnode)
tf_points_neighbours = time.time()
time_points_neighbours = tf_points_neighbours - t0_points_neighbours
print(f'Time taken to generate start/goal neighbours: {time_points_neighbours}')

# ###############################################################################################
# RUN DIJKSTRA'S ON THE TREE

def run_planner(start, goal):
    t0 = time.time()
    start.seen = True
    start.cost = 0
    start.parent = None
    onDeck = [start]
    path = []

    print("Starting Dijkstra's")

    while True:

        # check that the deck is not empty
        if not (len(onDeck) > 0):
            print('Path not found')
            return None
        
        # Pop the next node (state) from the deck
        node = onDeck.pop(0)

        node.done = True # this means the node has been processed

        if node.coords == goal.coords:
            path.append(goal)
            node_parent = goal.parent
            total_cost = node.cost
            tf = time.time()
            # Build the path and calculates the edge costs
            while node_parent != None:

                # adding the node to the path and calculating edge costs
                path.append(node_parent)
                node_parent = node_parent.parent

            path.reverse()

            t_planner = tf - t0

            print(f'Goal found. Goal cost is: {total_cost} in Time: {t_planner}')

            return path, t_planner
        
        elif node == start:

            for element in node.neighbours:
                element.seen = True
                element.parent = start
                element.cost = element.edge_cost(node)
                bisect.insort(onDeck, element)

        else: # node is neither a start nor goalnode
            for element in node.neighbours:
                
                # check whether the element has been seen (whether a parent has been assigned)
                if element.seen == False:
                    element.seen = True
                    element.parent = node
                    element.cost = element.edge_cost(node) + node.cost
                    bisect.insort(onDeck, element)

                # if the element has already been seen (a parent was assigned)
                elif (element.seen == True) and (element.done == False):
                    new_cost = element.edge_cost(node) + node.cost
                    if new_cost < element.cost:
                        onDeck.remove(element)
                        element.cost = new_cost
                        element.parent = node
                        bisect.insort(onDeck, element)
                


path, t_plan = run_planner(startnode, goalnode)
print('Did it work?')

# ###############################################################################################
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

# for pt in mega_coords:
#     plt.plot(pt[0], pt[1], 'bo')


# plot the polytopes
colour_list = ['orange', 'turquoise', 'indianred', 'darkseagreen', 'palevioletred', 
               'goldenrod', 'forestgreen', 'mediumpurple', 'peru', 'rosybrown', 'orange', 
               'turquoise', 'indianred', 'darkseagreen', 'orange', 'turquoise', 'indianred', 
               'darkseagreen', 'palevioletred', 'goldenrod', 'forestgreen', 'mediumpurple', 
               'peru', 'rosybrown', 'orange', 'turquoise', 'indianred', 'darkseagreen', 
               'orange', 'turquoise', 'indianred', 
               'darkseagreen', 'palevioletred', 'goldenrod', 'forestgreen', 'mediumpurple', 
               'peru', 'rosybrown', 'orange', 'turquoise', 'indianred', 'darkseagreen', 
               'orange', 'turquoise', 'indianred', 
               'darkseagreen', 'palevioletred', 'goldenrod', 'forestgreen', 'mediumpurple', 
               'peru', 'rosybrown', 'orange', 'turquoise', 'indianred', 'darkseagreen']
idx = 0
print(len(vertex_list))
for group_verts in vertex_list:
    group_verts = reorder_verts_2D(group_verts)
    # print('The points')
    # print(group_verts[0, :], group_verts[1, :])
    plt.plot(group_verts[0, :], group_verts[1, :], colour_list[int(abs((len(colour_list) - 1) - idx % (len(vertex_list) - 1)))], linewidth=2)
    plt.fill(group_verts[0, :], group_verts[1, :], colour_list[int(abs((len(colour_list) - 1) - idx % (len(vertex_list) - 1)))])
    idx += 1

# plot the polytope intersections
for inter in inters_list:

    inter_vertex = inter.vertices()
 
    if inter_vertex.size > 0:
        group_inters = reorder_verts_2D(inter_vertex)
        plt.fill(group_inters[0,:], group_inters[1,:], 'lime')
        # plt.plot(group_inters[0,:], group_inters[1,:], 'orange', linewidth = 2)

# # plot all the neighbour connections - # THIS KEEPS CRASHINGGGG NJSKFNVJSKFNV 
# for node in nodes:
#     ncoords = node.coords

#     for neigh in node.neighbours:
#         neighcoords = neigh.coords
        
#         x_vals = [ncoords[0], neighcoords[0]]
#         y_vals = [ncoords[1], neighcoords[1]]

#         plt.plot(x_vals, y_vals, 'slategrey')

if path:
    path_len = len(path)
    idx = 0
    while idx != path_len - 1:
        x_vals = [path[idx].coords[0], path[idx+1].coords[0]]
        y_vals = [path[idx].coords[1], path[idx+1].coords[1]]
        idx += 1
        plt.plot(x_vals, y_vals, 'black')

    # print the path
    print('Path')
    for el in path:
        print(el.coords)

else:
    print('Path not found')

# plot all the nodes
for no in nodes:
    plt.plot(no.coords[0], no.coords[1], 'bo')

# plot the start and goal nodes
plt.plot(start[0], start[1], 'mo')
plt.plot(goal[0], goal[1], 'mo')

print("Time Report")
print(f'Time for Iris: {t_IRIS}')
print(f'Time for planner: {t_plan}')
print(f'Time taken to check intersections: {time_intersect}')
print(f'Time taken to build the graph: {time_build}')
print(f'Time taken to generate start/goal points: {time_points}')
print(f'Time taken to generate start/goal neighbours: {time_points_neighbours}')
print(f'Total Time: {t_IRIS + t_plan + time_intersect + time_build + time_points + time_points_neighbours}')

plt.axis('equal')
plt.show()
