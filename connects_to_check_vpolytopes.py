import torch
from pydrake.all import *
device = torch.device('cpu')

x1_min = 0
x2_min = 0
x1_max = 30
x2_max = 20

# define the obstacles
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

obstacles = [obs_rect1, obs_rect2, obs_rect3, obs_rect4, obs_rect5, obs_rect6, obs_rect7]

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

# from there we should discretize the line and look for intersections

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



def get_coords(tens_start, tens_goal):

    # do a mask to check for whether the line is vertical
    # act accordingly


    # then do the calculations for non-vertical lines by using the interpolation
    # using the formula. It doesn't matter which x/y values are bigger as long as 
    # you use the correct pair-wise description

    pass


def check_interpolation(tstart, tgoal):

    # start be getting the coordinates between each pairwise connection
    coords = get_coords(tstart, tgoal)




# define a random tensor with points to test the inFreespace claim
starts = [[0, 0], [5, 8], [20, 7]]
goals = [[5, 5], [5, 12], [25, 13]]

start_tens = torch.tensor(starts, dtype=torch.float, device=device)
goal_tens = torch.tensor(goals, dtype=torch.float, device=device)

mask = check_interpolation(start_tens, goal_tens)

# how to do the parallelization?
