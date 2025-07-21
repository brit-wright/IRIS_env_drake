#!/usr/bin/env python3.8
import numpy as np
import matplotlib.pyplot as plt
import time
from pydrake.all import *
import bisect
from math import sqrt, inf
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

# V_polytope triangles
tri_pts1 = np.array([[1,16],
                     [4,18],
                     [6,12]])

tri_pts2 = np.array([[3,6],
                     [8,10],
                     [8,14]])

tri_pts3 = np.array([[3,2],
                     [6,4],
                     [6,8]])

tri_pts4 = np.array([[7,4],
                     [11,6],
                     [12,2]])

tri_pts5 = np.array([[8,12],
                     [12,12],
                     [14,4]])

tri_pts6 = np.array([[6,18],
                     [11,14],
                     [14,19]])

tri_pts7 = np.array([[13,12],
                     [13,18],
                     [17,18]])

tri_pts8 = np.array([[14,6],
                     [16,2],
                     [16,12]])

tri_pts9 = np.array([[17,8],
                     [19,8],
                     [19,18]])

tri_pts10 = np.array([[16,2],
                     [21,6],
                     [22,16]])

tri_pts11 = np.array([[21,18],
                     [28,18],
                     [29,14]])

tri_pts12 = np.array([[22,5],
                     [25,10],
                     [25,15]])

tri_pts13 = np.array([[20,2],
                     [25,6],
                     [26,2]])

tri_pts14 = np.array([[26,6],
                     [28,2],
                     [28,14]])

obs_tri1 = VPolytope(tri_pts1.T)
obs_tri2 = VPolytope(tri_pts2.T)
obs_tri3 = VPolytope(tri_pts3.T)
obs_tri4 = VPolytope(tri_pts4.T)
obs_tri5 = VPolytope(tri_pts5.T)
obs_tri6 = VPolytope(tri_pts6.T)
obs_tri7 = VPolytope(tri_pts7.T)
obs_tri8 = VPolytope(tri_pts8.T)
obs_tri9 = VPolytope(tri_pts9.T)
obs_tri10 = VPolytope(tri_pts10.T)
obs_tri11 = VPolytope(tri_pts11.T)
obs_tri12 = VPolytope(tri_pts12.T)
obs_tri13 = VPolytope(tri_pts13.T)
obs_tri14 = VPolytope(tri_pts14.T)

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
obstacles = [obs_tri1, obs_tri2, obs_tri3, obs_tri4, obs_tri5, 
             obs_tri6, obs_tri7, obs_tri8, obs_tri9, obs_tri10, 
             obs_tri11, obs_tri12, obs_tri13, obs_tri14]

# choose a sample intial point to do optimization from

sample_pts = []

# let's do 3 sample points

num_samples = 100

for pt in range(num_samples):
    sample_pt = np.array([np.random.uniform(x1_min, x1_max), np.random.uniform(x2_min, x2_max)])

    while (obs_tri1.PointInSet(sample_pt) or obs_tri2.PointInSet(sample_pt) or obs_tri3.PointInSet(sample_pt) or
           obs_tri4.PointInSet(sample_pt) or obs_tri5.PointInSet(sample_pt) or obs_tri6.PointInSet(sample_pt) or 
           obs_tri7.PointInSet(sample_pt) or obs_tri8.PointInSet(sample_pt) or obs_tri9.PointInSet(sample_pt) or 
           obs_tri10.PointInSet(sample_pt) or obs_tri11.PointInSet(sample_pt) or obs_tri12.PointInSet(sample_pt) or 
           obs_tri13.PointInSet(sample_pt) or obs_tri14.PointInSet(sample_pt)):
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

    vertex_list.append(VPolytope(r_H).vertices()) # contains the vertices of all the solution polytopes
    # print(f'Vertices: {VPolytope(r_H).vertices()}')
    center_list.append([x, y])
    r_H_list.append(r_H)
    refined_samples_list.append(sample_pts[alg_num])

    vertex_dict[(x, y)] = VPolytope(r_H).vertices() # saves the solution polytopes as a dictionary indexed by cheb center


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
t0_intersect = time.time()
center_pairs = [] # polytope pairs stores the pairs of chebyshev centers that have overlapping polytopes
polytope_pairs = []
inters_centers = []
inters_list = []
inters_list_hedron = []

intersecting_vertices = []

# check for intersections between the polyhedrons
for hedron_idx in range(len(r_H_list)):
    hedron_counter = 0
    for hedron in r_H_list:
        if hedron != r_H_list[hedron_idx]:
            # check for intersection
            inters = hedron.Intersection(r_H_list[hedron_idx])  # this returns a polyhedron

            inters_list.append(VPolytope(inters))
            intersecting_vertices.append(VPolytope(inters).vertices())
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
refined_intersecting_vertices = []

# print(f'Length of intersecting vertices is {len(intersecting_vertices)}')
# print(f'Length of inters list is: {len(inters_list)}')



# FIXME: for the post-processing, I also need to use the relationship between intersecting_vertices and 
# and refine_index to add to the vertex_dict[(x, y)] = VPolytope(r_H).vertices() dictionary

# print(f'Old vertex dictionary is {len(vertex_dict)}')

# post-process the polytope_pairs and inters_list entries to only consider the valid intersections
refine_index = 0
check_centers_list = []

for inter_el in inters_list:
    inter_vertex = inter_el.vertices()
    if inter_vertex.size > 0:

        # update the refinement lists
        refined_center_pairs.append(center_pairs[refine_index])
        refined_polytope_pairs.append(polytope_pairs[refine_index])
        refined_inters_list.append(inters_list[refine_index])
        refined_intersecting_vertices.append(intersecting_vertices[refine_index])


        # Calculate the Chebyshev center of that intersection
        inters_cheb_center = inters_list_hedron[refine_index].ChebyshevCenter()
        inters_cheb_c = inters_cheb_center.tolist()
        [x_inter, y_inter] = [round(inters_cheb_c[0], 6), round(inters_cheb_c[1], 6)]

        refined_inters_centers.append([x_inter, y_inter])

        check_centers_list.append((x_inter, y_inter))

        # update the vertices dictionary with the overlapping vertices
        if (x_inter, y_inter) not in vertex_dict:
            vertex_dict[(x_inter, y_inter)] = inter_vertex
        else:
            vertex_dict[(x_inter, y_inter)] = np.append(vertex_dict[(x_inter, y_inter)], inter_vertex, axis=1)

    refine_index += 1
tf_intersect = time.time()
time_intersect = tf_intersect - t0_intersect
print(f'Time taken to check intersections: {time_intersect}')
# print(f'New vertex dictionary is {len(vertex_dict)}')
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


# first define every coordinate as a node object

# create the mega-list of all polytopes and all coords
t0_build = time.time()
mega_topes = []
mega_coords = []

for element in refined_center_pairs:
    for coord in element:
        mega_coords.append(coord)
        mega_topes.append([refined_polytope_pairs[refined_center_pairs.index(element)][element.index(coord)]])


# the mega is the sum of the centers from the refined_center_pairs (the polytopes created by Iris that 
# are capable of connection with another polytope)
# and refined_inters_centers (the chebyshev centers of the intersecting polytopes)

mega_coords = mega_coords + refined_inters_centers
mega_topes = mega_topes + refined_polytope_pairs


already_processed_nodes = [] # has the coordinates of the nodes that have already been processed
nodes = []

# print(f'All coords are: {mega_coords}')

# print(f'Refined inters centers: {refined_inters_centers}')
# print(f'Checking all centers: {check_centers_list}')
# k = vertex_dict.keys()
# print(k)
# for knum in k:
#     if [knum[0], knum[1]] not in refined_inters_centers:
#         print(f'{knum} not in refined inters centers') 
# print(f'Check in refined center pairs: {refined_center_pairs}')

# print(f'Vertex keys are: {vertex_dict.keys()}')

# build a node list containing only the polytope and the coordinates of the center

# so mega_coords contains all the centers for the polytopes that intersect and all the centers for the
# polytope intersections. 
# Vertex_dict contains the vertices of all the polytope solutions and the vertices of all the polytopes
# formed from intersections

# this first if statement simply sets up the nodes in mega_coords. mega_coords contains duplicates so
# this just combines all the polytopes. It assigns vertex_neighbours for nodes that are also keys in vertex_dict
for element_idx in range(len(mega_coords)):


    # this allows for the polytope list to be built for nodes that were already added to the list
    # from being members of a key-value pair in a dictionary
    if mega_coords[element_idx] in already_processed_nodes:

        # first find the node in the nodes list
        node = [n for n in nodes if mega_coords[element_idx] == n.coords][0]

        # combine the polytopes list
        for pol in mega_topes[element_idx]:

            if pol not in node.polytopes:
                node.polytopes.append(pol)

        node_poly = node.polytopes


    elif mega_coords[element_idx] not in already_processed_nodes:
        
        # handles the case where the same coordinate will show up multiple times in mega_coords
        indices_found = [index for (index, coord) in enumerate(mega_coords) if coord == mega_coords[element_idx]]

        node_poly = [] # contains all the polytopes of the current node
        
        for ind in indices_found:
            
            node_poly.append(pol for pol in mega_topes[ind] if pol not in node_poly)

        # node_poly = mega_topes[element_idx] 

        node_coord = mega_coords[element_idx]

        node = Node([pol for pol in node_poly], node_coord)
        nodes.append(node)
        already_processed_nodes.append(node_coord)

        # want to also add the vertices as nodes. if the current node is a center of a polytope, we 
        # also add the vertices of the polytope as nodes

        # assigning neighbours also happens. we assign the vertices as the neighbours of the center,
        # the center as the neighbours of the vertices and the vertices as each others neighbours


    # let's cook, i guess. let's pull this block out of the elif so that all vertices and have
    # neighbours assigned?

    # print(f'Vertex centers are: {vertex_dict.keys()}')

    if (mega_coords[element_idx][0], mega_coords[element_idx][1]) in vertex_dict:

        vertex_node_list = []

        items = vertex_dict[(mega_coords[element_idx][0], mega_coords[element_idx][1])]

        # print(f'Coords are: {(mega_coords[element_idx][0], mega_coords[element_idx][1])}')
        # print(f'Items are: {items}')

        for idx in range(len(items[0])):

            pair = [float(items[0][idx]), float(items[1][idx])]

            if pair not in already_processed_nodes:

                newnode = Node([pol for pol in node_poly], pair)
                nodes.append(newnode)
                already_processed_nodes.append(pair)

                # also makes sense assign the neighbours here
                node.neighbours.append(newnode)
                newnode.neighbours.append(node)

                vertex_node_list.append(newnode)

            # need to include another condition where the pair has already been processed
            # but belongs to a different polytope
            elif pair in already_processed_nodes:
                node_pair = [n for n in nodes if n.coords == pair][0]

                # combining to get the full list of polytopes
                for element in node_poly:

                    if element not in node_pair.polytopes:
                        node_pair.polytopes.append(element)

                # create the neighbour connections 
                node.neighbours.append(node_pair)
                node_pair.neighbours.append(node)
                
                vertex_node_list.append(node_pair)

        # let's go through and connect all neighbours
        for element1 in vertex_node_list:
            for element2 in vertex_node_list:
                if element1 != element2:
                    element1.neighbours.append(element2)


        # basically, for the case of [0, 0] another node already had it as a neighbour so it got connected to
        # all of that node's neighbours

        # now, we process [0, 0] as a node on its own and as a key to a dictionary
        # we connect the node [0, 0] to all the members of its dictionary
        # and all the members of the dictionary to [0, 0]

# build the neighbour list based on the presence of the polytope of a node in an intersection with another
# polytope
for node in nodes:

    neighbour_coord_list = []

    # check for the presence of the node in a pair and add the neighbour as the inter point
    for element_idx in range(len(refined_polytope_pairs)):

        for part in range(0, 2):

            if refined_center_pairs[element_idx][part] == node.coords:

                nb_poly_coord = refined_inters_centers[element_idx]

                if (nb_poly_coord not in neighbour_coord_list) and (nb_poly_coord != node.coords):
                    nb = [n for n in nodes if n.coords == nb_poly_coord]
                    node.neighbours.append(nb[0])

                    neighbour_coord_list.append(nb_poly_coord)

# build the neighbour list based on the presence of the node as an intersection
    for inter_idx in range(len(refined_inters_list)):

        if refined_inters_centers[inter_idx] == node.coords:
            
            nb_coord1 = refined_center_pairs[inter_idx][0]
            nb_coord2 = refined_center_pairs[inter_idx][1]

            if (nb_coord1 not in neighbour_coord_list) and (nb_coord1 != node.coords):

                neighbour_coord_list.append(nb_coord1)

                nb1 = [n for n in nodes if n.coords == nb_coord1]
                node.neighbours.append(nb1[0])

            if (nb_coord2 not in neighbour_coord_list) and (nb_coord2 != node.coords):

                neighbour_coord_list.append(nb_coord2)

                nb2 = [n for n in nodes if n.coords == nb_coord2]
                node.neighbours.append(nb2[0])
tf_build = time.time()
time_build = tf_build - t0_build
print(f'Time taken to build the graph: {time_build}')
###############################################################################################
# RANDOMLY PLACE START AND GOAL NODES
t0_points = time.time()
start = [np.random.uniform(x1_min, x1_max), np.random.uniform(x2_min, x2_max)]

while check_obstacle_collision(start, obstacles) == True: # the start node intersects and obstacles
    start = [np.random.uniform(x1_min, x1_max), np.random.uniform(x2_min, x2_max)]

goal = [np.random.uniform(x1_min, x1_max), np.random.uniform(x2_min, x2_max)]

while (goal == start) or (check_obstacle_collision(goal, obstacles) == True) or (distance(start, goal) < (x1_max - x1_min)/2):
    goal = [np.random.uniform(x1_min, x1_max), np.random.uniform(x2_min, x2_max)]

# define the start and goal as nodes
startnode = Node(None, start)
goalnode = Node(None, goal)
tf_points = time.time()
time_points = tf_points - t0_points
print(f'Time taken to generate start/goal points: {time_points}')
###############################################################################################
# DEFINE THE NEIGHBOURS OF THE START AND GOAL NODE
t0_points_neighbours = time.time()
for node in nodes:
    if (check_interpolation([[round(node.coords[0], 6), round(node.coords[1], 6)], start], obstacles) == True) and node.coords != start:
        startnode.neighbours.append(node)
        node.neighbours.append(startnode)

for node in nodes:
    if (check_interpolation([[round(node.coords[0], 6), round(node.coords[1], 6)], goal], obstacles) == True) and node.coords != goal:
        goalnode.neighbours.append(node)
        node.neighbours.append(goalnode)

# also check for start-goal connection (though the points should not be chosed to let this happen)
if (check_interpolation([start, goal], obstacles) == True):
    goalnode.neighbours.append(startnode)
    startnode.neighbours.append(goalnode)
tf_points_neighbours = time.time()
time_points_neighbours = tf_points_neighbours - t0_points_neighbours
print(f'Time taken to generate start/goal neighbours: {time_points_neighbours}')
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

###############################################################################################
# PLOTTING
plt.figure()

# plot the domain
domain_V = VPolytope(domain)
domain_pts = domain_V.vertices()
domain_pts = reorder_verts_2D(domain_pts)
plt.fill(domain_pts[0, :], domain_pts[1, :], 'white')


# plot the obstacles (the walls)
obs_tri1_pts = obs_tri1.vertices()
obs_tri1_pts = reorder_verts_2D(obs_tri1_pts)
plt.fill(obs_tri1_pts[0, :], obs_tri1_pts[1, :], 'r')

obs_tri2_pts = obs_tri2.vertices()
obs_tri2_pts = reorder_verts_2D(obs_tri2_pts)
plt.fill(obs_tri2_pts[0, :], obs_tri2_pts[1, :], 'r')

obs_tri3_pts = obs_tri3.vertices()
obs_tri3_pts = reorder_verts_2D(obs_tri3_pts)
plt.fill(obs_tri3_pts[0, :], obs_tri3_pts[1, :], 'r')

obs_tri4_pts = obs_tri4.vertices()
obs_tri4_pts = reorder_verts_2D(obs_tri4_pts)
plt.fill(obs_tri4_pts[0, :], obs_tri4_pts[1, :], 'r')

obs_tri5_pts = obs_tri5.vertices()
obs_tri5_pts = reorder_verts_2D(obs_tri5_pts)
plt.fill(obs_tri5_pts[0, :], obs_tri5_pts[1, :], 'r')

obs_tri6_pts = obs_tri6.vertices()
obs_tri6_pts = reorder_verts_2D(obs_tri6_pts)
plt.fill(obs_tri6_pts[0, :], obs_tri6_pts[1, :], 'r')

obs_tri7_pts = obs_tri7.vertices()
obs_tri7_pts = reorder_verts_2D(obs_tri7_pts)
plt.fill(obs_tri7_pts[0, :], obs_tri7_pts[1, :], 'r')

obs_tri8_pts = obs_tri8.vertices()
obs_tri8_pts = reorder_verts_2D(obs_tri8_pts)
plt.fill(obs_tri8_pts[0, :], obs_tri8_pts[1, :], 'r')

obs_tri9_pts = obs_tri9.vertices()
obs_tri9_pts = reorder_verts_2D(obs_tri9_pts)
plt.fill(obs_tri9_pts[0, :], obs_tri9_pts[1, :], 'r')

obs_tri10_pts = obs_tri10.vertices()
obs_tri10_pts = reorder_verts_2D(obs_tri10_pts)
plt.fill(obs_tri10_pts[0, :], obs_tri10_pts[1, :], 'r')

obs_tri11_pts = obs_tri11.vertices()
obs_tri11_pts = reorder_verts_2D(obs_tri11_pts)
plt.fill(obs_tri11_pts[0, :], obs_tri11_pts[1, :], 'r')

obs_tri12_pts = obs_tri12.vertices()
obs_tri12_pts = reorder_verts_2D(obs_tri12_pts)
plt.fill(obs_tri12_pts[0, :], obs_tri12_pts[1, :], 'r')

obs_tri13_pts = obs_tri13.vertices()
obs_tri13_pts = reorder_verts_2D(obs_tri13_pts)
plt.fill(obs_tri13_pts[0, :], obs_tri13_pts[1, :], 'r')

obs_tri14_pts = obs_tri14.vertices()
obs_tri14_pts = reorder_verts_2D(obs_tri14_pts)
plt.fill(obs_tri14_pts[0, :], obs_tri14_pts[1, :], 'r')

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

# print(len(vertex_list))
for group_verts in vertex_list:
    group_verts = reorder_verts_2D(group_verts)
    # print('The points')
    # print(group_verts[0, :], group_verts[1, :])
    # print(abs(len(colour_list) - idx % len(vertex_list)))
    # print(type(abs(len(colour_list) - idx % len(vertex_list))))

    # plt.plot(group_verts[0, :], group_verts[1, :], colour_list[int(abs((len(colour_list) - 1) - idx % (len(vertex_list) - 1)))], linewidth=2)
    plt.fill(group_verts[0, :], group_verts[1, :], colour_list[int(abs((len(colour_list) - 1) - idx % (len(vertex_list) - 1)))])
    idx += 1

for inter in inters_list:

    inter_vertex = inter.vertices()

    if inter_vertex.size > 0:
        group_inters = reorder_verts_2D(inter_vertex)
        plt.fill(group_inters[0,:], group_inters[1,:], 'lime')
        plt.plot(group_inters[0,:], group_inters[1,:], 'slategrey', linewidth = 2)

# # plot all the neighbour connections
# for node in nodes:
#     ncoords = node.coords

#     for neigh in node.neighbours:
#         neighcoords = neigh.coords
        
#         x_vals = [ncoords[0], neighcoords[0]]
#         y_vals = [ncoords[1], neighcoords[1]]

#         plt.plot(x_vals, y_vals, 'slategrey')

# print(vertex_dict.keys())

# print(f'Startnode: {startnode.coords}')

# for n in startnode.neighbours:
#     print(n.coords)

# print(f'Goalnode: {goalnode.coords}')

# for n in goalnode.neighbours:
#     print(n.coords)


# for node in nodes:
#     print(f'Node is: {node.coords}')

#     for n in node.neighbours:
#         print(n.coords)


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

for ver in intersecting_vertices:
    plt.plot(ver[0], ver[1], 'yo')

for vert_group in vertex_list:
    for el_ind in range(len(vert_group[0])):
        plt.plot(vert_group[0][el_ind], vert_group[1][el_ind], 'ko')


# for key_num in vertex_dict:
#     vvert_group = vertex_dict[key_num]
#     for ell_ind in range(len(vvert_group[0])):
#         plt.plot(vvert_group[0][ell_ind], vvert_group[1][ell_ind], 'ro')

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

# ###### SUMMARY OF DIJKSTRA'S/NODE STUFF
# print(f'Startnode: {startnode.coords}')

# for n in startnode.neighbours:
#     print(n.coords)


# print(f'Goalnode: {goalnode.coords}')

# for n in goalnode.neighbours:
#     print(n.coords)


# for node in nodes:
#     print(f'Node is: {node.coords}')

#     for n in node.neighbours:
#         print(n.coords)