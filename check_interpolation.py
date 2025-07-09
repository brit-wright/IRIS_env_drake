#############################################################################################
# IMPORTS

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

################################################################################################
# DEFINE THE OBSTACLES

# V_polytope rectangles
rect1_pts = np.array([[5, 15],
                      [10, 15],
                      [5, 5],
                      [10, 5]])

rect2_pts = np.array([[15, 15],
                      [25, 15],
                      [15, 5],
                      [25, 5]])

obs_rect1 = VPolytope(rect1_pts.T)

obs_rect2 = VPolytope(rect2_pts.T)

obstacles = [obs_rect1, obs_rect2]

#################################################################################################
# PERFORMING INTERPOLATION BETWEEN TWO POINTS

def check_obstacle_collision(coord, obstacles):
    
    for v_pol in obstacles:
        # do an early check for whether the point intersects
        if v_pol.PointInSet(coord) == True:
            return coord, True
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
# returns true if the points can connect without intersection
def check_interpolation(pair, obstacles):

    x1, y1 = int(pair[0][0]), int(pair[0][1])
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

# wanna do a test case
collides = check_interpolation([[25, 5], [27.086549285345033, 16.444554633099873]], obstacles)
print(f'Clear Path = {collides}')


"""
Found the issue: The issue is with rounding.

The collision coordinate is [10, 5]

The thing is that sometimes the goal/start won't connect will connect with the obstacles because they
will register the neighbour (which is also basically a vertex) as [24.99999, 4.999999] instead of as [25, 5]


Instead of turning all the vertices into integers (which I think I might regret later) I'll just introduce another
exception about if the collision coordinate is a vertex then it shouldn't count
"""