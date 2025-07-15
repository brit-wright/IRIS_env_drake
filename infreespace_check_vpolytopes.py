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

# define a random tensor with points to test the inFreespace claim
starts = [[-0.5, 3], [19, 12], [21, 11.9], [25, 15]]

start_tens = torch.tensor(starts, dtype=torch.float, device=device)

# True means that the point is within bounds
in_bounds_mask = ((start_tens[:,0] >= x1_min) & (start_tens[:,0] <= x1_max)) & ((start_tens[:,1] >= x2_min) & (start_tens[:,1] <= x2_max))

# want to expand the start_tens variable to accommodate batch processing
start_tens = start_tens.unsqueeze(0).expand(len(obstacles), len(starts), 2) # shape is (7, 4, 2)

# also want to expand the A and b attributes of the obstacles

# we could try looping through the list of obstacles and stacking the A and b parameters
# ideally, this should be done when first declaring the obstacle variables


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

# check the shapes to see if they're correct
print(f'Shape of A stack: {A_stack.shape}')  # got the expected shape (7, 4, 2)
print(f'Shape of b stack: {b_stack.shape}') # got the expected shape (7, 4)


# next we do the matrix multiplication between start_tens and A
# take the transpose of the A matrix (we want to transpose the 1st and 2nd indices)
A_stack_T = A_stack.transpose(1, 2)

print(f'Checking the shape: {A_stack_T.shape}')

prod = torch.bmm(start_tens, A_stack_T)

print(f'Checking the shape of the product: {prod.shape}')

# next do the comparison between the product and the b matrix
mask = (prod <= b_stack.unsqueeze(1))
# the mask returns a shape of (7, 3, 4) --> So it has a True/False value for each obstacle
# and each point in start and each side(?) of the obstacle

# want the final version of mask to be of dimension 3
# if we take mask.all(dim = 2), then we're checking along the 2nd dimension (the constraint dimension)
# that all the results are True. This allows us to check, for a given obstacles-point combination, whether
# the constraints are satisfied
mask2 = mask.all(dim=2)

# this mask should then have size (7, 3)
# if we then take mask.all(dim = 0), we perform the mask along the 0th dimension (obstacles) to check if any
# of the results is True
mask3 = mask2.any(dim=0)


# Recall the inFreespace means that it is within bounds AND doesn't collide

print(f'The final mask is : {mask3}')

final_mask = in_bounds_mask & ~mask3

print(f'The bounds mask is {in_bounds_mask}')

print(f'The combined mask is: {final_mask}')

