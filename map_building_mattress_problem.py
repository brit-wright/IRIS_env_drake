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
seed = 2040
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
# IRIS ALGORITHM

# list of all the obstalces
obstacles = [obs_rect1, obs_rect2, obs_rect3, obs_rect4, obs_rect5, obs_rect6]

# choose a sample intial point to do optimization from

sample_pts = []

# let's do 3 sample points

num_samples = 300

for pt in range(num_samples):
    sample_pt = np.array([np.random.uniform(x1_min, x1_max), np.random.uniform(x2_min, x2_max)])

    while (obs_rect1.PointInSet(sample_pt) or obs_rect2.PointInSet(sample_pt) or obs_rect3.PointInSet(sample_pt) 
    or obs_rect4.PointInSet(sample_pt) or obs_rect5.PointInSet(sample_pt) or obs_rect6.PointInSet(sample_pt)):
        sample_pt = np.array([np.random.uniform(x1_min, x1_max), np.random.uniform(x2_min, x2_max)])
        
    sample_pts.append(sample_pt)



# sample_pt = np.array([np.random.uniform(x1_min, x1_max), np.random.uniform(x2_min, x2_max)])

# iris options
options = IrisOptions()
options.termination_threshold = 1e-3
options.iteration_limit = 200
options.configuration_obstacles = obstacles

refined_samples_list = []
r_H_list = []
center_list = []
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

    center_list.append([x, y])
    r_H_list.append(r_H)
    refined_samples_list.append(sample_pts[alg_num])


###############################################################################################
# PATH TRACING

def build_path(start, best_start_center, goal, best_goal_center):

    # Begin to trace the path from the start to the goal

    print(f'Best start center is {best_start_center}')

    path = [start, best_start_center]

    print(f'Initial path is {path}')

    # next, get the Chebyshev center that connects best to the path

    # recall that best_start_center is in connections

    curr = best_start_center
    stuck = False

    while stuck == False:

        # look for the neighbours of the current node
        neighbour_list = []

        for connection in connections:
            if curr in connection:
                interest_idx = int(abs(1 - (connection.index(curr)))) # quick way of saying I want the other element (neighbour) in the connection pair
                neighbour = connection[interest_idx]

                if neighbour not in path: #and distance(neighbour, goal) < distance(curr, goal)
                    neighbour_list.append(neighbour)

        # check which is the best neighbour
        if neighbour_list != []:
            best_neighbour = neighbour_list[0]
            for neighbour_idx in range(len(neighbour_list)):
                if distance(best_neighbour, goal) > distance(neighbour_list[neighbour_idx], goal):
                    best_neighbour = neighbour_list[neighbour_idx]
            path.append(best_neighbour)

            curr = best_neighbour

            if curr == best_goal_center:
                path.append(goal)
                stuck = True
        
        else:
            stuck = True
            
    return path

   
###############################################################################################
# CONNECTING ADJACENT CHEBYSHEV CENTERS

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

def connect_centers(center_list, obstacles):
    connect_pairs = []

    for center_idx in range(len(center_list)):

        x1, y1 = center_list[center_idx][0], center_list[center_idx][1]

        for center_el in center_list:

            x2, y2 = center_el[0], center_el[1]

            if center_list[center_idx] == center_el:
                continue
            
            line = create_coords(x1, y1, x2, y2)
            
            intersects = False

            for coord in line:
                intersects = check_obstacle_collision(coord, obstacles)
                if intersects == True:
                    break

            if intersects == False:

                # check if the connection pair is present already
                pair = [center_list[center_idx], center_el]

                if [pair[1], pair[0]] not in connect_pairs:
                    connect_pairs.append([center_list[center_idx], center_el])
                    print(f'{center_list[center_idx]} connects to {center_el}')

    return connect_pairs


connections = connect_centers(center_list, obstacles)

print('Connections')
print(connections)

###############################################################################################
# RANDOMLY PLACE START AND GOAL NODES

def distance(point1, point2):

    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]

    distance = sqrt((x1 - x2)**2 + (y1 - y2)**2) 
    return distance


start = [np.random.uniform(x1_min, x1_max), np.random.uniform(x2_min, x2_max)]

while check_obstacle_collision(start, obstacles) == True: # the start node intersects and obstacles
    start = [np.random.uniform(x1_min, x1_max), np.random.uniform(x2_min, x2_max)]

goal = [np.random.uniform(x1_min, x1_max), np.random.uniform(x2_min, x2_max)]

while (goal == start) or (check_obstacle_collision(goal, obstacles) == True) or (distance(start, goal) < (x1_max - x1_min)/2):
    goal = [np.random.uniform(x1_min, x1_max), np.random.uniform(x2_min, x2_max)]

print(f'Start is {start}')
print(f'Goal is {goal}')

###############################################################################################
# ATTEMPT TO CONNECT THE START AND GOAL NODES TO THE NEAREST CHEBYSHEV CENTERS

# start by trying to connect the start node
center_goal = []
center_start = []
start_connect_candidate = []
goal_connect_candidate = []

# get the distances for each center to the start/goal node
for center in center_list:
    # want to check center-start distance and center-goal distance
    center_start_distance = distance(center, start)
    center_goal_distance = distance(center, goal)
    center_start.append(center_start_distance)
    center_goal.append(center_goal_distance)

# get the best Chebyshev center candidates for the start node
for idx in range(len(center_start)):
    line = create_coords(start[0], start[1], center_list[idx][0], center_list[idx][1])
    intersects = False
    for coord in line:
        intersects = check_obstacle_collision(coord, obstacles)
        if intersects == True:
            break
    # check that the start is closer and that the start and center don't interact
    if (center_start[idx] < center_goal[idx]) and (intersects == False): # first ensure that the node is closer to the start node than the goal
        start_connect_candidate.append(idx)

# the best candidate is the one closest to the start
best_start_distance = center_start[start_connect_candidate[0]]
best_start_center = center_list[start_connect_candidate[0]]
for ele in range(1, len(start_connect_candidate)):

    if center_start[start_connect_candidate[ele]] < best_start_distance:
        best_start_distance = center_start[start_connect_candidate[ele]]
        best_start_center = center_list[start_connect_candidate[ele]]

# get the best Chebyshev center candidates for the goal node
for idx in range(len(center_goal)):
    line = create_coords(goal[0], goal[1], center_list[idx][0], center_list[idx][1])
    intersects = False
    for coord in line:
        intersects = check_obstacle_collision(coord, obstacles)
        if intersects == True:
            break
    # check that the start is closer and that the start and center don't interact
    if (center_goal[idx] < center_start[idx]) and (intersects == False): # first ensure that the node is closer to the start node than the goal
        goal_connect_candidate.append(idx)

# the best candidate is the one closes to the goal
best_goal_distance = center_goal[goal_connect_candidate[0]]
best_goal_center = center_list[goal_connect_candidate[0]]
for ele in range(1, len(goal_connect_candidate)):

    if center_goal[goal_connect_candidate[ele]] < best_goal_distance:
        best_goal_distance = center_goal[goal_connect_candidate[ele]]
        best_goal_center = center_list[goal_connect_candidate[ele]]


print(f'Best start center: {best_start_center}')
print(f'Best goal center: {best_goal_center}')

###############################################################################################
# GET THE PATH

path = build_path(start, best_start_center, goal, best_goal_center)
print(f'PATH IS: {path}')

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

# plot the IRIS answers 
for answer in range(len(r_H_list)):
    r_V = VPolytope(r_H_list[answer])
    r_pts = r_V.vertices()
    r_pts = reorder_verts_2D(r_pts)
    plt.fill(r_pts[0, :], r_pts[1, :], 'g')


# plot the Chebyshev centers
# print('Centers')
for center in center_list:
    plt.plot(center[0], center[1], 'bo')
    # print(center)


# plot the connections
for connection in connections:
    x_values = [connection[0][0], connection[1][0]]
    y_values = [connection[0][1], connection[1][1]]

    plt.plot(x_values, y_values, 'b')

# plot the start and goal nodes
plt.plot(start[0], start[1], 'ko')
plt.plot(goal[0], goal[1], 'ko')


# plot the start and goal to center connections
plt.plot([start[0], best_start_center[0]], [start[1], best_start_center[1]], 'm')
plt.plot([goal[0], best_goal_center[0]], [goal[1], best_goal_center[1]], 'm')


plt.axis('equal')
plt.show()