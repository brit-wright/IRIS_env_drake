#!/usr/bin/env python3.10
import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import *#!/usr/bin/env python3.8
import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import *#!/usr/bin/env python3.8
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


STEP_SIZE = 0.25
NMAX = 10000
SMAX = 10000

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
rect_pts1 = np.array([[0, 10],
                     [15, 10],
                     [15, 9.9],
                     [0, 9.9]])

rect_pts2 = np.array([[18, 10],
                     [30, 10],
                     [30, 9.9],
                     [18, 9.9]])

rect_pts3 = np.array([[15, 20],
                      [15, 12],
                      [14.9, 12],
                      [14.9, 20]])

rect_pts4 = np.array([[12, 12],
                      [15, 12],
                      [15, 11.9],
                      [12, 11.9]])

rect_pts5 = np.array([[15, 10],
                      [15, 5],
                      [14.9, 5],
                      [14.9, 10]])

rect_pts6 = np.array([[15, 0], 
                    [12, 4.9],
                    [12, 5],
                    [14.9, 0]])

rect_pts7 = np.array([[18, 12],
                      [21, 12],
                      [21, 11.9],
                      [18, 11.9]])

obs_rect1 = VPolytope(rect_pts1.T)
obs_rect2 = VPolytope(rect_pts2.T)
obs_rect3 = VPolytope(rect_pts3.T)
obs_rect4 = VPolytope(rect_pts4.T)
obs_rect5 = VPolytope(rect_pts5.T)
obs_rect6 = VPolytope(rect_pts6.T)
obs_rect7 = VPolytope(rect_pts7.T)

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
obstacles = [obs_rect1, obs_rect2, obs_rect3, obs_rect4, obs_rect5, obs_rect6, obs_rect7]

# choose a sample intial point to do optimization from

sample_pts = []

# let's do 3 sample points

num_samples = 50

for pt in range(num_samples):
    sample_pt = np.array([np.random.uniform(x1_min, x1_max), np.random.uniform(x2_min, x2_max)])

    while (obs_rect1.PointInSet(sample_pt) or obs_rect2.PointInSet(sample_pt) or obs_rect3.PointInSet(sample_pt) 
    or obs_rect4.PointInSet(sample_pt) or obs_rect5.PointInSet(sample_pt) or obs_rect6.PointInSet(sample_pt) 
    or obs_rect7.PointInSet(sample_pt)):
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
        
        current_vertex = (curr_vertex_list[0][curr_idx], curr_vertex_list[1][curr_idx])

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

            curr_inter_ver = (inter_vertex[0][ver_idx], inter_vertex[1][ver_idx])
        
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

# start = [np.random.uniform(x1_min, x1_max), np.random.uniform(x2_min, x2_max)]

# # start = [12.474357775465961, 2.939952153423444]

# while check_obstacle_collision(start, obstacles) == True or in_polytope(start) == False: # the start node intersects and obstacles
#     start = [round(np.random.uniform(x1_min, x1_max), 6), round(np.random.uniform(x2_min, x2_max), 6)]

# goal = [np.random.uniform(x1_min, x1_max), np.random.uniform(x2_min, x2_max)]

# while (goal == start) or (check_obstacle_collision(goal, obstacles) == True) or (distance(start, goal) < (x1_max - x1_min)/2) or in_polytope(goal) == False:
#     goal = [round(np.random.uniform(x1_min, x1_max), 6), round(np.random.uniform(x2_min, x2_max), 6)]

start = [5, 15]
goal = [5, 5]

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

# print(f'Start neighbours: {startnode.neighbours}')


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


# remove neighbour duplicates

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

    start_failure_nodes = []
    start_failure_coords = []

    print("Starting Dijkstra's")

    while True:

        # check that the deck is not empty
        if not (len(onDeck) > 0):
            print('Path not found')
            # print(f'The start_failure_nodes are: {start_failure_coords}')
            path = start_failure_nodes
            time_found = -1

            return start_failure_nodes, time_found
        
        # Pop the next node (state) from the deck
        node = onDeck.pop(0)
        start_failure_coords.append([round(node.coords[0], 6), round(node.coords[0], 6)])
        start_failure_nodes.append(node)

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

                
def run_reverse_planner(goal):
    t0 = time.time()
    goal.seen = True
    goal.cost = 0
    goal.parent = None
    onDeck2 = [goal]
    path2 = []

    goal_failure_nodes = []
    goal_failure_coords = []

    print("Starting Reverse Dijkstra's")

    while True:

        # check that the deck is not empty
        if not (len(onDeck2) > 0):
            print('Path not found')
            # print(f'The goal_failure_nodes are: {goal_failure_coords}')
            path = []

            return goal_failure_nodes
        
        # Pop the next node (state) from the deck
        node = onDeck2.pop(0)
        goal_failure_coords.append([round(node.coords[0], 6), round(node.coords[0], 6)])
        goal_failure_nodes.append(node)

        node.done = True # this means the node has been processed
        
        if node == goal:

            for element in node.neighbours:
                element.seen = True
                element.parent = goal
                element.cost = element.edge_cost(node)
                bisect.insort(onDeck2, element)

        else: # node is not a goalnode
            for element in node.neighbours:
                
                # check whether the element has been seen (whether a parent has been assigned)
                if element.seen == False:
                    element.seen = True
                    element.parent = node
                    element.cost = element.edge_cost(node) + node.cost
                    bisect.insort(onDeck2, element)

                # if the element has already been seen (a parent was assigned)
                elif (element.seen == True) and (element.done == False):
                    new_cost = element.edge_cost(node) + node.cost
                    if new_cost < element.cost:
                        onDeck2.remove(element)
                        element.cost = new_cost
                        element.parent = node
                        bisect.insort(onDeck2, element)

path, t_plan = run_planner(startnode, goalnode)

if t_plan == -1:
    start_fails = path
    # this means that the start to goal Dijkstra's planner failed
    goal_fails = run_reverse_planner(goalnode)

    best_node_pairs = []
    best_distances = []

    node_pairs = []

    distance_list = []

    # find the node to node pairings

    # need to include another distance list

    # this metric is lowkey trash. will come up with a better metric later on
    min_dist = 6
    max_dist = 8

    for start_fail in start_fails:

        for goal_fail in goal_fails:

                dist = distance(start_fail.coords, goal_fail.coords)

                if (dist >= min_dist) and (dist <= max_dist):

                    best_distances.append(dist)
                    best_node_pairs.append([start_fail, goal_fail])

    print(f'The new distance list is: {best_distances}')
    print(f'Best node pairs are: {best_node_pairs}')
    print(f'Length is: {len(best_distances)}')

    start_rrt = []
    goal_rrt = []


    # need to start creating the 'start and goal nodes' based on which nodes are first and second in the pair
    for pair in best_node_pairs:

        start_rrt.append(pair[0].coords)
        goal_rrt.append(pair[1].coords)

    print(f'Start nodes are: {start_rrt}')
    print(f'Goal nodes are: {goal_rrt}')


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
    

    def inFreespace(next_node):
        
        # check that the next point is in-bounds

        # returns False if any condition fails and True if all conditions pass
        in_bounds_mask = ((next_node[:,0] >= x1_min) & (next_node[:,0] <= x1_max)) & ((next_node[:,1] >= x2_min) & (next_node[:,1] <= x2_max))

        # The A_stack_T variable has already been pre-calculated so I can just to the batch matrix multiplication here
        prod = torch.bmm(next_node, A_stack_T)

        mask = (prod <= b_stack.unsqueeze(1))
        mask2 = mask.all(dim=2)
        mask3 = mask2.any(dim=0)

        final_mask = in_bounds_mask & ~mask3

        return final_mask
    
    def connectsTo():
        pass


    # RUN RRT IN GPU (CPU for now)
    def do_rrt(start_r, goal_r):
        device = torch.device('cpu')
        starts = torch.tensor(start_r, dtype=torch.float, device=device)
        goals = torch.tensor(goal_r, dtype=torch.float, device=device)

        # looks good so far
        print(f'Starts are: {starts}')
        print(f'Goals are: {goals}')


        # next need to run the actual RRT algorithm (pray 4 me)

        batch_size = len(start_r)
        node_counts = torch.ones(batch_size, dtype=torch.long, device=device)
        tree_parents = torch.full((batch_size, NMAX), -1, dtype=torch.long, device=device)

        tree_positions = torch.zeros((batch_size, NMAX, 2), device=device)
        tree_positions[:, 0, 0] = startnode.x
        tree_positions[:, 0, 1] = startnode.y

        iter = 0

        def addtotree(valid_batches, valid_nextnodes, nearest_indices):
            # start by assigning the parent of the new_node to be the nearnode
            tree_parents[valid_batches, node_counts[valid_batches]] = nearest_indices

            # add the new node to the tree
            tree_positions[valid_batches, node_counts[valid_batches], :] = valid_nextnodes

            # increment the node cout
            next_node_index = node_counts[valid_batches]
            node_counts[valid_batches] += 1
            
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
            targetnode[mask] = goal[mask]

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

            # Now that we have nextnode, we should then check the validity of next node by checking if it's in
            # free space and if it connects

            # first, check if the node is inFreespace()
            # we first define the node to be checked based on its x and y positions
            
            # right now, nextnode contains the coordinates for each batch
            # we can convert this tensor into a numpy array

            

            # freespace_mask_cpu is a numpy array
            freespace_mask = inFreespace(nextnode)

            # convert the numpy array to a tensor
            

            # next, we have to check if the nearnode connects to the nextnode
            # we follow a similar procesure where we first convert the nearnode tensor into a numpy array
            

            # then we send the nearnode numpy array and the nextnode numpy array to the connectsTo function
            connects_mask = connectsTo(nearnode, nextnode)

        

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


            if valid_nextnodes.shape[0] == 0:
                continue
            
            ##### GOAL CHECKING BLOCK #######

            # first make a mask to check whether the distance between the valid_nextnode and the goalnode
            # is within step_size

            # valid_nextnodes would be a subset of nextnodes which has shape valid_batches, (x,y)

            # Currently, the goalnode is of size (batch_size, (x,y)). Need to define valid_goal
            possible_goal = goal[valid_batches]

            # Find the Euclidean distance between the goal and each node in nextnode

            goal_diff = possible_goal - valid_nextnodes
            goal_distances = torch.norm(goal_diff, dim=1)
        
            # Next we create a mesh to see whether the distance is within the step size
            # print(f'Possible Goal: {possible_goal}')
            # print(f'Valid Next Node: {valid_nextnodes}')
            # print(f'Goal distances: {goal_distances}')
            within_threshold_mask = goal_distances < STEP_SIZE

            # Next we must create another mesh that tests whether the valid nextnode connects to the 
            # goalnode
            goal_connects_mask = connectsTo(valid_nextnodes, possible_goal)
            # compare both masks
            # print('Check mask types')
            # print(dist_mask.dtype)
            # print(f'dist_mask: {dist_mask}')
            # print(goal_connects_mask.dtype)
            # print(f'goal connects mask: {goal_connects_mask}')
            valid_goal_connects_mask = within_threshold_mask & goal_connects_mask

            # get the batch indices where the goal can be added to the tree
            goal_batches = valid_batches[valid_goal_connects_mask]
            
            goal_nodes = goal[goal_batches]
            
            # goal_indices = node_counts[goal_batches] - 1
            # goal_parents = goal_indices - 1
            
            # goal_tree_index = valid_nearest[goal_batches] + 2
            # add goal nodes to the tree
            # goal_tree_index = next_node_index[goal_batches] + 1

            if goal_batches.numel()>0:
                # print(f'Bruhhh: {goal_batches.numel()}')
                addtogoal(goal_batches, goal_nodes)

            # Set the batch to inactive
        
            # the last thing is to check the following criterion for whether to stop a batch
            # 1. The goal has been found
            

            # goal_found_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            # goal_found_mask[goal_batches] = True
                
            
                # print('Issue might come up here!')
                active_batches[goal_batches] = False
                # print(f'Goal found for batch: {goal_batches.tolist()} at Node: {node_counts[goal_batches]} and at Step: {step_counts[goal_batches]}')
                # print(tree_positions[goal_batches])
                # print(tree_parents[goal_batches])
                all_goal_batches = torch.cat([all_goal_batches, goal_batches])

            # Increment the number of steps. Recall the following step tensor definition
            # step_counts = torch.zeros(batch_size, dtype=torch.long, device=device)

            step_counts[active_batches] += 1

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
            if False in active_batches:
                # print(active_batches)
                pass
            if not active_batches.any():
                if (step_counts >= SMAX).all() | (node_counts >= NMAX).all():
                    # print(f'Process Aborted at Node Count = {node_counts} and \nStep Count = {step_counts}. No path found')
                    return None
                break

        tf = time.time()
        print(node_counts, step_counts)
        print(f'Stuck counter: {stuck_counter}')
        node_counts_cpu = node_counts.cpu().numpy()
        tree_parents_cpu = tree_parents.cpu().numpy()
        tree_positions_cpu = tree_positions.cpu().numpy()
        goal_indices = node_counts_cpu - 1
        all_goal_batches_cpu = all_goal_batches.cpu().numpy()
        # Let's send all this stuff to a function outside of the rrt called buildPath
        all_paths = buildPath(goal_indices, tree_parents_cpu, tree_positions_cpu, batch_size, all_goal_batches_cpu)
        return tf, all_paths

    def buildPath(goal_idx, node_parents, node_positions, batch_size, goal_batches):
       
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
            tup_list = [tuple(float(x) for x in point) for point in current_path]
            node_list = [Node(x,y) for (x,y) in tup_list]
            path.append(node_list)
            
            # print('Checking path validity ahhhhh')
            
            for ele in range(len(tup_list) - 1):
                
                (x1, y1) = tup_list[ele]
                (x2, y2) = tup_list[ele+1]

                dist = sqrt((x1 - x2)**2 + (y1 - y2)**2)
                return path


    # just do the call here :/
    do_rrt(start_rrt, goal_rrt)
###################################################################################################################







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
    # plt.plot(group_verts[0, :], group_verts[1, :], colour_list[int(abs((len(colour_list) - 1) - idx % (len(vertex_list) - 1)))], linewidth=2)
    plt.fill(group_verts[0, :], group_verts[1, :], colour_list[int(abs((len(colour_list) - 1) - idx % (len(vertex_list) - 1)))])
    idx += 1

# plot the polytope intersections
for inter in inters_list:

    inter_vertex = inter.vertices()
 
    if inter_vertex.size > 0:
        group_inters = reorder_verts_2D(inter_vertex)
        plt.fill(group_inters[0,:], group_inters[1,:], 'lime')

if t_plan != -1:
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
print(f'Time for planner: {t_plan}')
print(f'Time taken to check intersections: {time_intersect}')
print(f'Time taken to build the graph: {time_build}')
print(f'Time taken to generate start/goal points: {time_points}')
print(f'Time taken to generate start/goal neighbours: {time_points_neighbours}')
print(f'Total Time: {t_IRIS + t_plan + time_intersect + time_build + time_points + time_points_neighbours}')

plt.axis('equal')
plt.show()
