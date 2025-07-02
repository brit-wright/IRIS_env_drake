#!/usr/bin/env python3.8
import numpy as np
import matplotlib.pyplot as plt
import time
from pydrake.all import *

print('Hello!')

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