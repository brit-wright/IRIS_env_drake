#!/usr/bin/env python3.8
import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import *

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
    
    print(circle.shape)

    # scale the circle
    ellipse = B @ circle
    
    # translate the ellipse
    ellipse = ellipse + c

    print(ellipse.shape)

    return ellipse

###############################################################################################
# DOMAIN AND OBSTACLES

# define the domain of the problem, Xcal = {x | Ax <= b}
x1_min = -4
x1_max = 4
x2_min = -2
x2_max = 5

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

# creating my convex sets, H = {x | Ax <= b}
square_A = np.array([[1, 0], 
              [0, 1], 
              [-1, 0], 
              [0, -1]])
square_b = np.array([[1],
              [1],
              [1],
              [1]])
obs_square = HPolyhedron(square_A, square_b)

# V_polytope triangle, V = conv({v1, v2,..., vn})
tri_pts = np.array([[1, 2.5],
                    [2, 1.5],
                    [3, 4]])
obs_triangle = VPolytope(tri_pts.T)

# HyperEllipsoid, E = {x | (x - c)^T A' * A (x - c) <= 1}
ellipse_B = np.array([[1, 1],        # affine map of unit circle
                      [0, 1.5]])
ellipse_A = np.linalg.inv(ellipse_B) # A = inv(B)
ellipse_center = np.array([[-2],
                           [2.5]])
obs_ellipse = Hyperellipsoid(ellipse_A, ellipse_center)

###############################################################################################
# IRIS ALGORITHM

# list of all the obstalces
obstacles = [obs_square, obs_triangle, obs_ellipse]

# choose a sample intial point to do optimization from
sample_pt = np.array([np.random.uniform(x1_min, x1_max), np.random.uniform(x2_min, x2_max)])

# iris options
options = IrisOptions()
options.termination_threshold = 1e-3
options.iteration_limit = 200
options.configuration_obstacles = obstacles


# run the algorithm
r_H = Iris(obstacles, # obstacles list
           sample_pt, # sample point, (intial condition)
           domain,    # domain of the problem
           options)   # options

###############################################################################################
# PLOTTING
plt.figure()

# plot the domain
domain_V = VPolytope(domain)
domain_pts = domain_V.vertices()
domain_pts = reorder_verts_2D(domain_pts)
plt.fill(domain_pts[0, :], domain_pts[1, :], 'gray')

# plot the obstacles
obs_square_pts = VPolytope(obs_square).vertices()
obs_square_pts = reorder_verts_2D(obs_square_pts)
plt.fill(obs_square_pts[0, :], obs_square_pts[1, :], 'r')

obs_triangle_pts = obs_triangle.vertices()
obs_triangle_pts = reorder_verts_2D(obs_triangle_pts)
plt.fill(obs_triangle_pts[0, :], obs_triangle_pts[1, :], 'r')

obs_ellipse_pts = get_ellipse_pts_2D(ellipse_B, ellipse_center)
plt.fill(obs_ellipse_pts[0, :], obs_ellipse_pts[1, :], 'r')

# plot the IRIS answer
r_V = VPolytope(r_H)
r_pts = r_V.vertices()
r_pts = reorder_verts_2D(r_pts)
plt.fill(r_pts[0, :], r_pts[1, :], 'g')

# plot the intial condition
plt.plot(sample_pt[0], sample_pt[1], 'bo')

plt.axis('equal')
plt.show()