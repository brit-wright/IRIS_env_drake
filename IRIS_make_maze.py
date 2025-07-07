#!/usr/bin/env python3.8
import numpy as np
import matplotlib.pyplot as plt
import time
from pydrake.all import *

# LET'S BUILD A MAZE :D

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

rect1_pts = np.array([[2, 18],
                      [2, 17],
                      [12, 18],
                      [12, 17]])

rect2_pts = np.array([[3, 17],
                      [6, 17],
                      [3, 14],
                      [6, 14]])

rect3_pts = np.array([[4, 14],
                      [5, 14],
                      [5, 6],
                      [4, 6]])

rect4_pts = np.array([[3, 6],
                      [6, 6],
                      [3, 3],
                      [6, 3]])

rect5_pts = np.array([[3, 3],
                      [3, 1],
                      [21, 3],
                      [21, 1]])

rect6_pts = np.array([[11, 11],
                      [11, 6],
                      [12, 6],
                      [12, 11]])

rect7_pts = np.array([[10, 14],
                      [10, 11],
                      [13, 14],
                      [13, 11]])

rect8_pts = np.array([[11, 15],
                      [11, 14],
                      [18, 15],
                      [18, 14]])

rect9_pts = np.array([[18, 16],
                      [18, 13],
                      [21, 16],
                      [21, 13]])

rect10_pts = np.array([[21, 15],
                       [21, 14],
                       [27, 15],
                       [27, 14]])

rect11_pts = np.array([[27, 16],
                       [27, 13],
                       [30, 16],
                       [30, 13]])

rect12_pts = np.array([[17, 10],
                       [17, 3],
                       [19, 10],
                       [19, 3]])

rect13_pts = np.array([[23, 11],
                       [25, 11],
                       [23, 6],
                       [25, 6]])

rect14_pts = np.array([[22, 3],
                       [22, 1],
                       [28, 3],
                       [28, 1]])


obs_rect1 = VPolytope(rect1_pts.T)
obs_rect2 = VPolytope(rect2_pts.T)
obs_rect3 = VPolytope(rect3_pts.T)
obs_rect4 = VPolytope(rect4_pts.T)
obs_rect5 = VPolytope(rect5_pts.T)
obs_rect6 = VPolytope(rect6_pts.T)
obs_rect7 = VPolytope(rect7_pts.T)
obs_rect8 = VPolytope(rect8_pts.T)
obs_rect9 = VPolytope(rect9_pts.T)
obs_rect10 = VPolytope(rect10_pts.T)
obs_rect11 = VPolytope(rect11_pts.T)
obs_rect12 = VPolytope(rect12_pts.T)
obs_rect13 = VPolytope(rect13_pts.T)
obs_rect14 = VPolytope(rect14_pts.T)

###############################################################################################
# IRIS ALGORITHM

# list of all the obstalces
obstacles = [obs_rect1, obs_rect2, obs_rect3, obs_rect4, obs_rect5, 
             obs_rect6, obs_rect7, obs_rect8, obs_rect9, obs_rect10, 
             obs_rect11, obs_rect12, obs_rect13, obs_rect14]

# choose a sample intial point to do optimization from

sample_pts = []

# let's do 3 sample points

num_samples = 100

for pt in range(num_samples):
    sample_pt = np.array([np.random.uniform(x1_min, x1_max), np.random.uniform(x2_min, x2_max)])

    while (obs_rect1.PointInSet(sample_pt) or obs_rect2.PointInSet(sample_pt) or obs_rect3.PointInSet(sample_pt) or
           obs_rect4.PointInSet(sample_pt) or obs_rect5.PointInSet(sample_pt) or obs_rect6.PointInSet(sample_pt) or 
           obs_rect7.PointInSet(sample_pt) or obs_rect8.PointInSet(sample_pt) or obs_rect9.PointInSet(sample_pt) or 
           obs_rect10.PointInSet(sample_pt) or obs_rect11.PointInSet(sample_pt) or obs_rect12.PointInSet(sample_pt) or 
           obs_rect13.PointInSet(sample_pt) or obs_rect14.PointInSet(sample_pt)):
        sample_pt = np.array([np.random.uniform(x1_min, x1_max), np.random.uniform(x2_min, x2_max)])
        
    sample_pts.append(sample_pt)



# sample_pt = np.array([np.random.uniform(x1_min, x1_max), np.random.uniform(x2_min, x2_max)])

# iris options
options = IrisOptions()
options.termination_threshold = 1e-3
options.iteration_limit = 200
options.configuration_obstacles = obstacles

r_H_list = []
center_list = []
t0 = time.time()
for alg_num in range(num_samples):
    # run the algorithm
    r_H = Iris(obstacles, # obstacles list
            sample_pts[alg_num], # sample point, (intial condition)
            domain,    # domain of the problem
            options)   # options


    cheb_center = r_H.ChebyshevCenter()
    r_H_list.append(r_H)
    center_list.append(cheb_center)
tf = time.time()
t_taken = tf - t0
print(f'Time Taken: {t_taken}')
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

# plot the IRIS answers 

for answer in range(len(r_H_list)):

    r_V = VPolytope(r_H_list[answer])
    r_pts = r_V.vertices()
    r_pts = reorder_verts_2D(r_pts)
    plt.fill(r_pts[0, :], r_pts[1, :], 'g')

# plot the intial condition
for sample in sample_pts:
    plt.plot(sample[0], sample[1], 'bo')

plt.axis('equal')
plt.show()

print('Hello World!')