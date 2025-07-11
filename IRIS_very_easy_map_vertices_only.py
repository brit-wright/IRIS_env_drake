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
rect1_pts = np.array([[5, 5],
                      [25, 5],
                      [5, 15],
                      [25, 15]])

# rect2_pts = np.array([[15, 10],
#                       [20, 10],
#                       [15, 5],
#                       [20, 5]])

# rect3_pts = np.array([[16, 15],
#                       [25, 15],
#                       [16, 11],
#                       [25, 11]])

obs_rect1 = VPolytope(rect1_pts.T)

# obs_rect2 = VPolytope(rect2_pts.T)

# obs_rect3 = VPolytope(rect3_pts.T)

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
# obstacles = [obs_rect1, obs_rect2, obs_rect3]
obstacles = [obs_rect1]

# choose a sample intial point to do optimization from

sample_pts = []

# let's do 3 sample points

num_samples = 20

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

polytope_vertex_dict = {}  # THIS DICTIONARY HAS THE POLYTOPE AS THE KEY AND THE VERTICES AS THE VALUES

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
    
    vertex_list.append(VPolytope(r_H).vertices()) # this stores the vertices associated with each IRIS solution
    # print(f'Vertices: {VPolytope(r_H).vertices()}')

    # Assign all the vertices and polytope as key-value pairs
    curr_polytope = VPolytope(r_H)
    curr_vertex_list = curr_polytope.vertices()
    all_pols.append(curr_polytope)

    for curr_idx in range(len(curr_vertex_list[0])):
        
        current_vertex = (round(curr_vertex_list[0][curr_idx], 6), round(curr_vertex_list[1][curr_idx], 6))

        if current_vertex not in polytope_vertex_dict:

            polytope_vertex_dict[(current_vertex)] = [curr_polytope]

        elif current_vertex in polytope_vertex_dict:

            polytope_vertex_dict[(current_vertex)].append(curr_polytope)
    

    center_list.append([x, y])
    r_H_list.append(r_H)
    refined_samples_list.append(sample_pts[alg_num])

    vertex_dict[(x, y)] = VPolytope(r_H).vertices()


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

# HERE, we need to only keep the polytope-vertex pairs, where there actually is an intersection

# we currently have all_pols --> how to get parents and children from that?
print(f'Length of r_H_list is {len(r_H_list)}')
print(f'Length of all polytopes list is {len(all_pols)}')

polytope_parent_child_dict = {} # the child is the key and the parents are the values
# check for intersections between the polyhedrons
for hedron_idx in range(len(r_H_list)):
    hedron_counter = 0
    for hedron in r_H_list:
        if hedron != r_H_list[hedron_idx]:
            # check for intersection
            inters = hedron.Intersection(r_H_list[hedron_idx])  # this returns a polyhedron
            inters_vpol = VPolytope(inters)

            inters_list.append(inters_vpol)

            polytope_parent_child_dict[inters_vpol] = [all_pols[r_H_list.index(hedron)], all_pols[hedron_idx]]
            
        hedron_counter += 1

# print(f'The parent child dictionary is {polytope_parent_child_dict}')

print('Checking length compatibility')

print(f'Length of dictionary is {len(polytope_parent_child_dict)}')
print(f'Length of inters list: {len(inters_list)}')


refined_polytope_parent_child_dict = {}

# post-process the polytope_pairs and inters_list entries to only consider the valid intersections
refine_index = 0
for inter_el in inters_list:
    inter_vertex = inter_el.vertices() # stores the vertices for each 'possible' intersection. includes empties as well
    if inter_vertex.size > 0:
        
        # parent-child assignment in new dictionary
        refined_polytope_parent_child_dict[inter_el] = polytope_parent_child_dict[inter_el]

        # assign the vertices to polytopes
        for ver_idx in range(len(inter_vertex[0])):

            curr_inter_ver = (round(inter_vertex[0][ver_idx], 6), round(inter_vertex[1][ver_idx], 6))
        
            # do a check for whether the vertex is already in the dictionary
            if curr_inter_ver not in polytope_vertex_dict:

                # business as usual
                polytope_vertex_dict[curr_inter_ver] = [inters_list[refine_index]]

            elif curr_inter_ver in polytope_vertex_dict:
                polytope_vertex_dict[curr_inter_ver].append(inters_list[refine_index])

    refine_index += 1
tf_intersect = time.time()
time_intersect = tf_intersect - t0_intersect
print(f'Time taken to check intersections: {time_intersect}')
# print("Checkpoint 2")
# print(vertex_dict)

print(f'Length of the refined dictionary is {len(refined_polytope_parent_child_dict)}')

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

# define the nodes

for entry in polytope_vertex_dict:

    node = Node(polytope_vertex_dict[entry], entry)
    nodes.append(node)

# from here we can define the neighbour connections by looking at what nodes have the same neighbours


# we also need to look at what polytopes are in the parent child assignment and assign neighbour that way
# as well 

for node in nodes:

    for node_el in nodes:

        # making sure we don't self-assign neighbours and making sure we don't repear a neighbour assignment
        if (node != node_el) and (node_el not in node.neighbours):

            for pol in node.polytopes:
                
                # check whether any polytope in node is contained in the node_el polytopes list

                if pol in node_el.polytopes: 

                    # they are neighbours
                    node.neighbours.append(node_el)
                    node_el.neighbours.append(node)
                    break

# also need to do a parent-child polytope neighbour assignment
for node in nodes:

    for pol in node.polytopes:

        # check if the polytope is in the dictionary 
        if pol in refined_polytope_parent_child_dict:

            # then we look for all the nodes that contain the parents of the list
            parents = refined_polytope_parent_child_dict[pol]

            # get all the neighbour nodes
            for ele in parents:
                
                more_neighbours = [n for n in nodes if ele in n.polytopes]

                for mo in more_neighbours:
                    if mo not in node.neighbours:
                        mo.neighbours.append(node)
                        node.neighbours.append(mo)

# also suppose the case where 1 parent contains two sub-polytopes. We should also have the ability
# to connect the vertices of the sub-polytopes to each other.

for subpol in refined_polytope_parent_child_dict:

    pols = refined_polytope_parent_child_dict[subpol] # all the parent polytopes in subpolytope 1

    for child in refined_polytope_parent_child_dict:

        if subpol != child:

            pars = refined_polytope_parent_child_dict[child] # all the parent polytopes in subpolytope 2

            for pol in pols:
                if pol in pars: # different children have the same parent polytope

                    # connect all the children vertices to each other
                    subpol_nodes = [n for n in nodes if subpol in n.polytopes]
                    child_nodes = [n for n in nodes if child in n.polytopes]

                    # connect them
                    for subpol_node in subpol_nodes:

                        for child_node in child_nodes:

                            if subpol_node != child_node:
                                subpol_node.neighbours.append(child_node)
                                child_node.neighbours.append(subpol_node)

tf_build = time.time()
time_build = tf_build - t0_build
print(f'Time taken to build the graph: {time_build}')

###############################################################################################
# DEFINING THE IN_POLYTOPE FUNCTION

def in_polytope(node_coords):
    for tope in all_pols:
        if tope.PointInSet(node_coords):
            return True
    return False

###############################################################################################
# RANDOMLY PLACE START AND GOAL NODES

# NEED TO INTRODUCE CONSTRAINT THAT THE START AND GOAL AREN'T GENERATED IN FREESPACE BUT IN AN ACTUAL POLYTOPE
t0_points = time.time()

start = [np.random.uniform(x1_min, x1_max), np.random.uniform(x2_min, x2_max)]

# start = [12.474357775465961, 2.939952153423444]

while check_obstacle_collision(start, obstacles) == True or in_polytope(start) == False: # the start node intersects and obstacles
    start = [round(np.random.uniform(x1_min, x1_max), 6), round(np.random.uniform(x2_min, x2_max), 6)]

goal = [np.random.uniform(x1_min, x1_max), np.random.uniform(x2_min, x2_max)]

while (goal == start) or (check_obstacle_collision(goal, obstacles) == True) or (distance(start, goal) < (x1_max - x1_min)/2) or in_polytope(goal) == False:
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

# New method of defining the start/goal neighbours
# 1. Check if the start/goal node is in a polytope intersection
# Also need to account for duplicates of neighbours

start_neigh_list = []
goal_neigh_list = []

start_tope_list = []

for node in nodes:
    if (node!= startnode) and (node!= goalnode):
        for tope in node.polytopes:
            if tope.PointInSet(start):
                startnode.neighbours.append(node)
                node.neighbours.append(startnode)
                
                if startnode.polytopes == None:
                    startnode.polytopes = [tope]
                else:
                    startnode.polytopes.append(tope)

for node in nodes:
    if (node!= goalnode) and (node != startnode):  # I removed the node!=startnode criterion since now startnode should have polytopes assigned
        for tope in node.polytopes:
            if tope.PointInSet(goal):
                goalnode.neighbours.append(node)
                node.neighbours.append(goalnode)

                if goalnode.polytopes == None:
                    goalnode.polytopes = [tope]
                else:
                    goalnode.polytopes.append(tope)



# also include parent-child relationships for defining the start/goal nodes

# do a check for whether this polytope is a child 

# we look at all the polytopes in the goal node. We check if the goalnode is a parent
# if yes, we connect to the child

print(f'Start neighbours: {startnode.neighbours}')


start_coords = []
for neigh in startnode.neighbours:
    start_coords.append(neigh.coords)

goal_coords = []
for neigh in goalnode.neighbours:
    goal_coords.append(neigh.coords)

for tope in startnode.polytopes:

    for child in refined_polytope_parent_child_dict:

        if tope in refined_polytope_parent_child_dict[child]:

            start_neighbours = [n for n in nodes if child in n.polytopes]

            for start_neigh in start_neighbours:

                if start_neigh.coords not in start_coords:

                    startnode.neighbours.append(start_neigh)
                    start_neigh.neighbours.append(startnode)
                    start_coords.append(start_neigh.coords)

for tope in goalnode.polytopes:

    for child in refined_polytope_parent_child_dict:

        if tope in refined_polytope_parent_child_dict[child]:

            goal_neighbours = [n for n in nodes if child in n.polytopes]

            for goal_neigh in goal_neighbours:

                if goal_neigh.coords not in goal_coords:

                    goalnode.neighbours.append(goal_neigh)
                    goal_neigh.neighbours.append(goalnode)
                    goal_coords.append(goal_neigh.coords)

# check for direct connection between startnode and goal node
if (startnode.polytopes!= None) and (goalnode.polytopes!= None):

    for tope in startnode.polytopes:
        if tope.PointInSet(goal):
            goalnode.neighbours.append(startnode)
            startnode.neighbours.append(goalnode)
            break



tf_points_neighbours = time.time()
time_points_neighbours = tf_points_neighbours - t0_points_neighbours
print(f'Time taken to generate start/goal neighbours: {time_points_neighbours}')

print(f'Number of nodes is: {len(nodes)}')


# remove neighbour duplicated

for node in nodes:
    unique_neighbours = []
    unique_neighbour_coords = []

    for neigh in node.neighbours:

        if (neigh.coords not in unique_neighbour_coords) and (neigh.coords != node.coords):
            unique_neighbours.append(neigh)
            unique_neighbour_coords.append(neigh.coords)

    node.neighbours = None
    node.neighbours = unique_neighbours

                    

###############################################################################################
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
            path = []
            time_found = 0
            return path, time_found
        
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
                print(f'Length of Deck is: {len(onDeck)}')

        else: # node is neither a start nor goalnode
            for element in node.neighbours:
                
                # check whether the element has been seen (whether a parent has been assigned)
                if element.seen == False:
                    element.seen = True
                    element.parent = node
                    element.cost = element.edge_cost(node) + node.cost
                    bisect.insort(onDeck, element)
                    print(f'Length of Deck is: {len(onDeck)}')

                # if the element has already been seen (a parent was assigned)
                elif (element.seen == True) and (element.done == False):
                    new_cost = element.edge_cost(node) + node.cost
                    if new_cost < element.cost:
                        onDeck.remove(element)
                        element.cost = new_cost
                        element.parent = node
                        bisect.insort(onDeck, element)
                        print(f'Length of Deck is: {len(onDeck)}')
                


path, t_plan = run_planner(startnode, goalnode)
print('Did it work?')

###############################################################################################
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

# obs_rect2_pts = obs_rect2.vertices()
# obs_rect2_pts = reorder_verts_2D(obs_rect2_pts)
# plt.fill(obs_rect2_pts[0, :], obs_rect2_pts[1, :], 'r')

# obs_rect3_pts = obs_rect3.vertices()
# obs_rect3_pts = reorder_verts_2D(obs_rect3_pts)
# plt.fill(obs_rect3_pts[0, :], obs_rect3_pts[1, :], 'r')

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

for group_verts in vertex_list:
    group_verts = reorder_verts_2D(group_verts)
    # print('The points')
    # print(group_verts[0, :], group_verts[1, :])
    plt.plot(group_verts[0, :], group_verts[1, :], colour_list[idx % len(vertex_list)], linewidth=2)
    plt.fill(group_verts[0, :], group_verts[1, :], colour_list[int(abs((len(colour_list) - 1) - idx % (len(vertex_list) - 1)))])
    idx += 1

# plot the polytope intersections
for inter in inters_list:

    inter_vertex = inter.vertices()
 
    if inter_vertex.size > 0:
        group_inters = reorder_verts_2D(inter_vertex)
        plt.fill(group_inters[0,:], group_inters[1,:], 'lime')
        # plt.plot(group_inters[0,:], group_inters[1,:], 'orange', linewidth = 2)

# plot all the neighbour connections
for node in nodes:
    ncoords = node.coords

    for neigh in node.neighbours:
        neighcoords = neigh.coords
        
        x_vals = [ncoords[0], neighcoords[0]]
        y_vals = [ncoords[1], neighcoords[1]]

        plt.plot(x_vals, y_vals, 'slategrey')

# print(f'Startode neighbours: {startnode.neighbours}')

# for neigh in startnode.neighbours:
#     print(neigh.coords)

# print(f'Goalnode neighbours: {goalnode.neighbours}')


# print(vertex_dict.keys())

print(f'Startnode: {startnode.coords}')

for n in startnode.neighbours:
    print(n.coords)


print(f'Goalnode: {goalnode.coords}')

for n in goalnode.neighbours:
    print(n.coords)


for node in nodes:
    print(f'Node is: {node.coords}')

    for n in node.neighbours:
        print(n.coords)


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

# plot all the nodes (includes the vertices)
for no in nodes:
    plt.plot(no.coords[0], no.coords[1], 'bo')

# plot the start and goal nodes
plt.plot(start[0], start[1], 'mo')
plt.plot(goal[0], goal[1], 'mo')

print("Time Report")
print(f'Time for Iris: {t_IRIS}')
# print(f'Time for planner: {t_plan}')
print(f'Time taken to check intersections: {time_intersect}')
print(f'Time taken to build the graph: {time_build}')
print(f'Time taken to generate start/goal points: {time_points}')
print(f'Time taken to generate start/goal neighbours: {time_points_neighbours}')
# print(f'Total Time: {t_IRIS + t_plan + time_intersect + time_build + time_points + time_points_neighbours}')
print(f'Total Time: {t_IRIS + time_intersect + time_build + time_points + time_points_neighbours}')


plt.axis('equal')
plt.show()

##########################################333
# THINGS TO NOTE:
# One thing I noted is that in the case where we have a polytope and a polytope intersection
# There were no assignments made such that all the child vertices connect with all the vertices in the
# parents. 



# Let's recall how things are added to the dictionary
# First we add all the parent (and possible non-parent) polytopes to the dictionary
# Next we add the vertices of the childrena nd their polytopes to the dictionary

# The only type of overlap that is considered is if the same vertex is shared by two polytopes

# In the neighbour assignment we only think of assigning the same polytopes as neighbours

# Next we need to consider applying the parent-child definitions to the polytopes so that all the 
# parent vertices are assigned to child vertices and vice versa