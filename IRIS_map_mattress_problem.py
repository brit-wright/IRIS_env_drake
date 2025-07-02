#!/usr/bin/env python3.8
import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import *#!/usr/bin/env python3.8
import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import *#!/usr/bin/env python3.8
import numpy as np
import matplotlib.pyplot as plt

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
   
print('Here')
print(center_list)
print(r_H_list)
print(refined_samples_list)

###############################################################################################
# CONNECTING ADJACENT NODES


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

# # plot the intial condition
# for sample in refined_samples_list:
#     plt.plot(sample[0], sample[1], 'bo')


# plot the Chebyshev centers
for center in center_list:
    plt.plot(center[0], center[1], 'bo')

plt.axis('equal')
plt.show()