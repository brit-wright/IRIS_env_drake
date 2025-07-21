import torch
from pydrake.all import *
import time
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
        
        # start_tens = start_tens.unsqueeze(0).expand(len(obstacles), len(starts), 2) # shape is (7, 4, 2)

        # need to expand next_node so that it has the correct shape
        

        # print(len(next_node))

        # print(f'Shape of tensor is: {next_node.shape}')
        # print(f'Shape of A_stack_T is: {A_stack_T.shape}')

        
        # check that the next point is in-bounds

        # returns False if any condition fails and True if all conditions pass

        # individual checks
        # print(next_node[:, 0])
        # print(next_node[:, 1])

        # print(next_node[:,0] >= x1_min)
        # print(next_node[:,0] <= x1_max)
        # print(next_node[:,1] >= x2_min)
        # print(next_node[:,1] <= x2_max)

        in_bounds_mask = ((next_node[:,0] >= x1_min) & (next_node[:,0] <= x1_max)) & ((next_node[:,1] >= x2_min) & (next_node[:,1] <= x2_max))

        next_node = next_node.unsqueeze(0).expand(len(obstacles), len(next_node), 2)

        # print(len(next_node))

        # print(f'Shape of tensor is: {next_node.shape}')
        # print(f'Shape of A_stack_T is: {A_stack_T.shape}')
       
       
        # The A_stack_T variable has already been pre-calculated so I can just to the batch matrix multiplication here
        prod = torch.bmm(next_node, A_stack_T)

        mask = (prod <= b_stack.unsqueeze(1))
        mask2 = mask.all(dim=2)
        mask3 = mask2.any(dim=0)

        final_mask = in_bounds_mask & ~mask3

        return final_mask



# def check_interpolation(tens_start, tens_goal):

#     # Here we take in the tensors defining the start and goal points for each RRT
#     # We start by doing an initial check to see which pairs of start-goal entries have the same x-value (vertical line check)

#     vertical_mask = tens_start[:, 0] == tens_goal[:, 0]

#     indices = np.arange(len(tens_start))

#     vertical_indices = indices[vertical_mask]
#     non_vertical_indices = indices[~vertical_mask]

#     num_divisions = 5

#     start_non_vert = tens_start[~vertical_mask]
#     goal_non_vert = tens_goal[~vertical_mask]

#     x_divs = torch.linspace(0, 1, steps=num_divisions, device=device).view(-1, 1)

#     x_vals_non_vert = start_non_vert[:, 0] + (goal_non_vert[:, 0] - start_non_vert[:, 0]) * x_divs

#     # check if we can calculate the y_values using the same method
#     y_vals_non_vert = start_non_vert[:, 1] + (goal_non_vert[:, 1] - start_non_vert[:, 1]) * x_divs

#     # checked with a plot and they look good. send them over to the interpolator
#     connects_non_vert = torch.ones(len(start_non_vert), dtype=torch.bool, device=device)
    
#     for b in range(num_divisions):

#         new = torch.tensor([[x_vals_non_vert[b][a], y_vals_non_vert[b][a]] for a in range(len(start_non_vert))], dtype=torch.float, device=device)
        
#         result = inFreespace(new)

#         connects_non_vert = connects_non_vert & result

#     # find the corresponding y-values by doing the same interpolation

#     start_vert = tens_start[vertical_mask]
#     goal_vert = tens_goal[vertical_mask]

#     # update 2: using num_divisions instead of step_size
    
#     # start by choosing the divisions?
#     y_divs = torch.linspace(0, 1, steps=num_divisions, device=device).view(-1, 1)
    
#     # apply these divisions to the linear interpolation between the y_start and y_goal

#     y_vals_vert = start_vert[:, 1] + (goal_vert[:, 1] - start_vert[:, 1]) * y_divs

#     # define x_vals_vert based on this new definition
#     x_vals_vert = [torch.tensor(start_vert[:, 0]).unsqueeze(0).expand(num_divisions, len(start_vert))]

#     # next we do the freespace test on these pairs of x and y values

#     connects_vert = torch.ones(len(start_vert), dtype=torch.bool, device=device)


#     x_vals_vert = x_vals_vert[0]

#     for b in range(num_divisions):

#         new = torch.tensor([[x_vals_vert[b][a], y_vals_vert[b][a]] for a in range(len(start_vert))], dtype=torch.float, device=device)
        
#         result = inFreespace(new)

#         connects_vert = connects_vert & result

#     # next we need to re-combine based on the indices that are and aren't vertical
#     connects_to_result = torch.zeros(len(indices), dtype=torch.bool, device=device)

#     connects_to_result[vertical_indices] = connects_vert
#     connects_to_result[non_vertical_indices] = connects_non_vert

#     return connects_to_result



def check_interpolation(tens_start, tens_goal):

    # OPTIMIZING THIS

    t_connectsto_start = time.time()
    counting_infreespace = 0
    timing_infreespace = 0

    num_divisions = 200
    divs = torch.linspace(0, 1, steps=num_divisions, device=device).view(-1, 1)

    # Here we take in the tensors defining the start and goal points for each RRT
    # We start by doing an initial check to see which pairs of start-goal entries have the same x-value (vertical line check)

    vertical_mask = tens_start[:, 0] == tens_goal[:, 0]
    non_vertical_mask = ~vertical_mask

    
    # FOR THE NON-VERTICAL LINES
    start_non_vert = tens_start[non_vertical_mask]
    goal_non_vert = tens_goal[non_vertical_mask]

    # get the x and y values for the non-vertical case
    x_vals_non_vert = (start_non_vert[:, 0]).unsqueeze(0) + (goal_non_vert[:, 0] - start_non_vert[:, 0]).unsqueeze(0) * divs

    # check if we can calculate the y_values using the same method
    y_vals_non_vert = (start_non_vert[:, 1]).unsqueeze(0) + (goal_non_vert[:, 1] - start_non_vert[:, 1]).unsqueeze(0) * divs

    
    # collect all the test points
    new_non_vert = torch.stack([x_vals_non_vert, y_vals_non_vert], dim=2).reshape(-1, 2)

    # call inFreespace
    result_non_vert = inFreespace(new_non_vert)

    # re-shape the result
    result_non_vert = result_non_vert.view(num_divisions, -1)

    # define the mask
    connects_non_vert = result_non_vert.all(dim=0)

    # FOR THE VERTICAL LINES
    start_vert = tens_start[vertical_mask]
    goal_vert = tens_goal[vertical_mask]

    y_vals_vert = (start_vert[:, 1]).unsqueeze(0) + (goal_vert[:, 1] - start_vert[:, 1]).unsqueeze(0) * divs
    x_vals_vert = start_vert[:, 0].unsqueeze(0).expand(num_divisions, -1)
    
    new_vert = torch.stack([x_vals_vert, y_vals_vert], dim=2).reshape(-1, 2)

    result_vert = inFreespace(new_vert)

    result_vert = result_vert.view(num_divisions, -1)

    connects_vert = result_vert.all(dim=0)


    # next we need to re-combine based on the indices that are and aren't vertical
    connects_to_result = torch.zeros(len(tens_start), dtype=torch.bool, device=device)

    connects_to_result[vertical_mask] = connects_vert
    connects_to_result[non_vertical_mask] = connects_non_vert

    t_connectsto_end = time.time()
    t_connectsto = t_connectsto_end - t_connectsto_start
    # print(f'def connectsTo: {t_connectsto}')

    return connects_to_result



# define a random tensor with points to test the inFreespace claim
starts = [[0, 0], [5, 8], [20, 7], [4, 9], [21, 5], [12, 7]]
goals = [[5, 5], [5, 12], [25, 13], [4, 10], [23, 5], [18, 7]]

# Expected result: True, False, False, False, True, False

start_tens = torch.tensor(starts, dtype=torch.float, device=device)
goal_tens = torch.tensor(goals, dtype=torch.float, device=device)

mask = check_interpolation(start_tens, goal_tens)

print(f'The mask is: {mask}')









# how to do the parallelization?

"""

HERE ARE ALL THE OLD COMMENTS

"""
    # for the pairs that have a non-vertical mask, we want to collect the points for linear interpolation
    # step_s = 0.01
    # start_not_vert = tens_start[~vertical_mask]
    # goal_not_vert = tens_goal[~vertical_mask]

    # num_steps = np.ceil(np.abs(start_not_vert[:, 0] - goal_not_vert[:, 0])/step_s)
    # num_steps = num_steps.type(torch.int64)

    # # this section is lowkey not vectorized. can't be vectorized :\

    # # next we get the x-values by calling np.linspace
    # x_vals = [np.linspace(xstart, xgoal, steps) for xstart, xgoal, steps in zip(start_not_vert[:,0], goal_not_vert[:,0], num_steps)]

    # # get the corresponding y_values by using the linear interpolation equation
    # y_vals = [(ymin + (ymax - ymin)/(xmax - xmin) * (xval - xmin)) for (xmin, ymin), (xmax, ymax), xval in zip(start_not_vert, goal_not_vert, x_vals)]


    # # now that we've assembled the x and y values, we can call inFreespace on each x-y pair
    # print('XVALS')
    # print(x_vals)

    # print('YVALS')
    # print(y_vals)

    # print(len(x_vals[0]))

    # # inFreespace works by taking in a tensor and for that tensor, checking if any of the points fails the
    # # check

    # # how to define the tensor that gets sent over --> We can define the tensor as one x-y pair in the group

    # # we can define each tensor pair by looping through the list

    # # we want to perform the loop such that we're defining the tensor to be an x-y value pair for each entry

    # inner = num_steps[0]
    # outer = len(x_vals)

    # # wanna update such that each tensor has the form torch.tensor([x_vals[a][b], y_vals[a][b]] for a in outer)

    # # wanna define an initial connects mask, where we assume the values to be true
    # connects = torch.ones(len(start_not_vert), dtype=torch.bool, device=device)
    
    # # slight issue for definition of new
    # """
    # it's defining the nodes like this 
    # tensor([[x1, y1] ,[x2, y2]])

    # but when it sends over to inFreespace, this causes
    # nextnode[:, 0] = [x1, y1] and nextnode[:, 1] == [x2, y2]
    # """
    
    
    # for b in range(inner):

    #     new = torch.tensor([[x_vals[a][b], y_vals[a][b]] for a in range(outer)], dtype=torch.float, device=device)
    #     print(new)

    #     # from here, we wanna send the tensor over to inFreespace so that the necessary checks can be done there
    #     # result is True if the point is valid and False if the point is invalid
    #     result = inFreespace(new)

    #     # the raw result of result is a 7 by 2 tensor, where 7 is the number of obstacles and 2 is the number of points
    #     # that are non-vertical. So basically each entry says that for the nth obstacle and for the ith non-vertical point
    #     # the test point is or is not in freespace

    #     # we can further alter the result by saying that we wanna check for all in terms of the obstacles like so:
    #     # result2 = result.all(dim=0)

    #     print(f'result1: {result}')
    #     # print(f'result2: {result2}')

    #     connects = connects & result

    # print(f'Final Answer is ver 1: {connects}')


    # NEXT, WE NEED TO WORK ON THE CASE WITH VERTICAL LINES :)


    # altering so instead I just need to select a certain number of divisions (should make the tensorizing process easier)

    
    # changing the non-vertical line check so that it also does the new check with num_divs