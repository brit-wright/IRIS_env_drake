### TENSORIZED AND PRE-PROCESSED WALLS

import matplotlib.pyplot as plt
import numpy as np
import random
import time
from math               import pi, sin, cos, atan2, sqrt, ceil, dist
from scipy.spatial      import KDTree
from shapely.geometry   import Point, LineString, Polygon, MultiPolygon, MultiLineString
from shapely.prepared   import prep
from vandercorput       import vandercorput
import cProfile
import torch
# seed = int(random.random()*10000)
# seed = 9676
seed = 3331
random.seed(seed)
# print(f"{seed=}")


STEP_SIZE = 0.25
# NMAX = 90   # Set the maximum number of nodes
# SMAX = 90   # Set the maximum number of steps

NMAX = 10000
SMAX = 10000


(xmin, xmax) = (0, 30)
(ymin, ymax) = (0, 20)

(xA, xB, xC, xD, xE) = ( 5, 12, 15, 18, 21)
(yA, yB, yC, yD)     = ( 5, 10, 12, 15)

xlabels = (xmin, xA, xB, xC, xD, xE, xmax)
ylabels = (ymin, yA, yB, yC, yD,     ymax)

# Draw the outer boundary/walls
outside = LineString([[xmin, ymin], [xmax, ymin], [xmax, ymax],
                      [xmin, ymax], [xmin, ymin]])

# Draw the interior walls that the mattress will have to move around
wall1   = LineString([[xmin, yB], [xC, yB]])
wall2   = LineString([[xD, yB], [xmax, yB]])
wall3   = LineString([[xB, yC], [xC, yC]])
wall4   = LineString([[xC, yC], [xC, ymax]])
wall5   = LineString([[xC, yB],[xC, yA]])
wall6   = LineString([[xC, ymin], [xB, yA]])
bonus   = LineString([[xD, yC], [xE, yC]])

device='cpu'
twall1   = [[xmin, yB], [xC, yB]]
twall2   = [[xD, yB], [xmax, yB]]
twall3   = [[xB, yC], [xC, yC]]
twall4   = [[xC, yC], [xC, ymax]]
twall5   = [[xC, yB],[xC, yA]]
twall6   = [[xC, ymin], [xB, yA]]
tbonus   = [[xD, yC], [xE, yC]]
twalls = [twall1, twall2, twall3, twall4, twall5, twall6, tbonus]
wall_coords = torch.tensor([[[twall[0][0],twall[0][1]], [twall[1][0],twall[1][1]]] for twall in twalls], dtype=torch.float, device=device)
W = wall_coords.shape[0]

# Collect all the walls and prepare(?). I'm including the bonus wall because why not?
walls = prep(MultiLineString([outside, wall1, wall2, wall3, wall4, wall5, wall6, bonus]))
# walls = prep(MultiLineString([outside, wall1, wall2, wall3]))

# Define the start/goal states (x, y, theta) of the mattress
(xstart, ystart) = (xA, yD)
(xgoal, ygoal) = (5, 5)


# Visualization Utility
class Visualization:
    def __init__(self):

        # Clear the current, or create a new figure
        plt.clf()

        # Create new axes, enable the grid, and set the axis limits.
        plt.axes()
        plt.grid(True)
        plt.gca().axis('on')
        plt.gca().set_xlim(xmin, xmax)
        plt.gca().set_ylim(ymin, ymax)
        plt.gca().set_xticks(xlabels)
        plt.gca().set_yticks(ylabels)
        plt.gca().set_aspect('equal')

        # Show the walls
        for l in walls.context.geoms:
            plt.plot(*l.xy, 'k', linewidth=2)
        if bonus in walls.context.geoms:
            plt.plot(*bonus.xy, 'b:', linewidth=3)

        # Show
        self.show()

    def show(self, text = ''):
        # Show the plot
        plt.pause(0.001)

        # If text is specified, print and wait for confirmation
        if len(text)>0:
            input(text + ' (hit return to continue)')

    def drawNode(self, node, *args, **kwargs):
        plt.plot(node.x, node.y, *args, **kwargs)

    def drawEdge(self, head, tail, *args, **kwargs):
        plt.plot((head.x, tail.x), (head.y, tail.y), *args, **kwargs)

    def drawPath(self, path, *args, **kwargs):
        for i in range(len(path)-1):
            self.drawEdge(path[i], path[i+1], *args, **kwargs)

# NODE DEFINITION

class Node:
    def __init__(self, x, y):
        # Define a parent 
        self.parent = None

        # Define/remember the state/coordinates (x,y,theta) of the node
        self.x = x
        self.y = y
        
    # Node Utilities
    # To print the node
    def __repr__(self):
        return("<Node %5.2f,%5.2f>" % (self.x, self.y))
    
    # Compute/create an intermediate node for checking the local planner
    def intermediate(self, other, alpha):
        return Node(self.x + alpha * (other.x - self.x),
                self.y + alpha * (other.y - self.y))
    
    # Return a tuple of coordinates to compute Euclidean distances
    def coordinates(self):
        return(self.x, self.y)
    
    # Compute the relative distance Euclidean distance to another node
    def distance(self, other):
        return dist(self.coordinates(), other.coordinates())
    
# Converting the vectorized (orientation-version of) Freespace function into tensor logic
def inFreespace(next_node):

    # Returns False if any of the conditions fails
    in_bounds_mask = ((next_node[:,0] >= xmin) & (next_node[:,0] <= xmax)) & ((next_node[:,1] >= ymin) & (next_node[:,1] <= ymax))
    B = next_node.shape[0]
    
    p2 = next_node.unsqueeze(1).expand(-1, W, -1) # This is supposed to change the dimension of nextnode to (B, W, 2)
    # also want to change the wall coords to have the same (B, W, 2) shape
    # the shape of wall_coords is, by default, (W, 2, 2)

    # We separate the walls into their end-points q1 and q1. We then want to unsqueeze and expand each in order to 
    # get the (B, W, 2) shape. Note that unsqueeze(0) gives the shape [1, 7, 2]. Expansing (B, -1, -1) means that we 
    # wanna change the first dimension to B and leave the other dimensions as they are

    q1 = wall_coords[:,0,:].unsqueeze(0).expand(B, -1, -1)
    q2 = wall_coords[:,1,:].unsqueeze(0).expand(B, -1, -1)
    
    def orientation(p, q, r):
        """Return orientation of triplet (p, q, r) as 0, 1, or 2"""
        # 0 --> colinear, 1 --> clockwise turn, 2 --> counterclockwise turn
        # val performs the cross product to get the orientation
        # val == 0 --> colinear, val > 0 --> clockwise, val < 0 --> counterclockwise
        val = (q[:,:,1] - p[:,:,1]) * (r[:,:,0] - q[:,:,0]) - (q[:,:,0] - p[:,:,0]) * (r[:,:,1] - q[:,:,1])
        return torch.where(
            torch.abs(val) < 1e-8,
            torch.tensor(0, device=val.device),  # colinear
            torch.where(val > 0, torch.tensor(1, device=val.device), torch.tensor(2, device=val.device))  # clockwise/counterclockwise
        )
    
    
    def on_segment(p, q, r):
        """Check if point q lies on segment pr"""
        return (
            (q[:,:,0] <= torch.max(p[:,:,0], r[:,:,0])) & (q[:,:,0] >= torch.min(p[:,:,0], r[:,:,0])) &
            (q[:,:,1] <= torch.max(p[:,:,1], r[:,:,1])) & (q[:,:,1] >= torch.min(p[:,:,1], r[:,:,1]))
        )

    o1 = orientation(q1, q2, p2)
    # a general intersection test
    # (o1 == 0) means there is colinearity and on_segment means there's overlap. --> True means not in Freespace
    intersects = (o1 == 0) & on_segment(q1, p2, q2)
    wall_intersect = intersects.any(dim=1)
    return ~wall_intersect & in_bounds_mask

def connectsTo(nearnode, nextnode):
    """
    Returns a boolean tensor indicating which node pairs can connect without intersecting walls.
    
    Inputs:
    - nearnode: (B, 2) tensor of start points
    - nextnode: (B, 2) tensor of end points
    
    Output:
    - (B,) boolean tensor: True if the segment is not blocked, False if it intersects any wall
    """
    B = nearnode.shape[0]

    # Let's go back to our everything True assumption

    # Okay this is the weird part. Basically need to do some re-sizing to prep for broadcasting 
    # so that dimensions can become compatible.

    # Overall, we want to expand to the shape (B, W, 2) in order to compare the Batches to the Walls

    p1 = nearnode.unsqueeze(1).expand(-1, W, -1) # This is supposed to change the dimension of nearnode to (B, W, 2)
    p2 = nextnode.unsqueeze(1).expand(-1, W, -1) # This is supposed to change the dimension of nextnode to (B, W, 2)

    # also want to change the wall coords to have the same (B, W, 2) shape
    # the shape of wall_coords is, by default, (W, 2, 2)

    # We separate the walls into their end-points q1 and q1. We then want to unsqueeze and expand each in order to 
    # get the (B, W, 2) shape. Note that unsqueeze(0) gives the shape [B, W, 2]. Expanding (B, -1, -1) means that we 
    # wanna change the first dimension to B and leave the other dimensions as they are

    q1 = wall_coords[:,0,:].unsqueeze(0).expand(B, -1, -1) # this contains one endpoint, has shape (B, W, 2)
    q2 = wall_coords[:,1,:].unsqueeze(0).expand(B, -1, -1) # contains the other endpoint, has shape (B, W, 2)

    # let's try the bounding box method...

    # Wall: define minimum and maximum wall coordinates
    wall_mins = wall_coords.min(dim=1).values  # should be of shape (W,2)
    wall_maxs = wall_coords.max(dim=1).values  # should be of shape (W,2)

    # Wall: expand the min and max for broadcasting
    wall_mins_e = wall_mins.unsqueeze(0).expand(B, -1, -1)  # should be of shape (B, W, 2)
    wall_maxs_e = wall_maxs.unsqueeze(0).expand(B, -1, -1)  # should be of shape (B, W, 2)

    # Nodes: define the minimum and maximum coordinates
    seg_mins = torch.minimum(p1, p2) # Shape is B, W, 2
    seg_maxs = torch.maximum(p1, p2) # Shape is B, W, 2

    # Define the bounding box mask. If any entry in disjoint mask is true, it means we know for sure
    # that this wall will not intersect (without having to call the orientation function on it)

    disjoint_mask = (
        (seg_maxs[:,:,0] < wall_mins_e[:,:,0]) |
        (seg_mins[:,:,0] > wall_maxs_e[:,:,0]) |
        (seg_maxs[:,:,1] < wall_mins_e[:,:,1]) |
        (seg_mins[:,:,1] > wall_maxs_e[:,:,1])
    )

    valid_mask = ~disjoint_mask # True means that the wall should be processed

    # We now go into vectorizing the orientation operation. Each parameter passed into orientation should have shape
    # (B, W, 2)
    def orientation(p, q, r):
        """Return orientation of triplet (p, q, r) as 0, 1, or 2"""
        # 0 --> colinear, 1 --> clockwise turn, 2 --> counterclockwise turn
        # val performs the cross product to get the orientation
        # val == 0 --> colinear, val > 0 --> clockwise, val < 0 --> counterclockwise
        val = (q[:,1] - p[:,1]) * (r[:,0] - q[:,0]) - (q[:,0] - p[:,0]) * (r[:,1] - q[:,1])
        
        return torch.where(
            torch.abs(val) < 1e-8,
            torch.tensor(0, device=device),  # colinear
            torch.where(val > 0, torch.tensor(1, device=device), torch.tensor(2, device=device))  # clockwise/counterclockwise
        )
    
    """Returns a (B,) boolean tensor if segment p1-p2 intersects q1-q2"""

    # Need to re-define p1, p2, q1, q2 with valid_mask applied
    b_idx, w_idx = torch.nonzero(valid_mask, as_tuple=True) # returns the batch and wall indices where valid_mask is True

    # Get the masked p1, p2, q1, q2 tensors
    p1_masked = p1[b_idx, w_idx]
    p2_masked = p2[b_idx, w_idx]
    q1_masked = q1[b_idx, w_idx]
    q2_masked = q2[b_idx, w_idx]


    # Apply the disjoint mask to p1, p2, p3, p4 so that irrelevant walls don't get processed

    o1 = orientation(p1_masked, p2_masked, q1_masked)
    o2 = orientation(p1_masked, p2_masked, q2_masked)
    o3 = orientation(q1_masked, q2_masked, p1_masked)
    o4 = orientation(q1_masked, q2_masked, p2_masked)

    intersects = (o1 != o2) & (o3 != o4) # shape is (b, w). True if line intersects with wall

    result = torch.ones(B, dtype=torch.bool, device=device)

    if intersects.any():    # if any entry in intersect is True
        batch_collide = b_idx[intersects] # define batch_collide as the batch_indices where there are node-wall collisions
        result[batch_collide] = False # for those batches, connectsTo is False
    return result

# RRT Function

def rrt(startnode, goalnode, visual):
    
    # Leaving many more comments because I'm genuinely confused lol

    ################### LET'S START BY DEFINING THE IMPORTANT VARIABLES FOR THIS PROBLEM ####################

    # first we need to indicate that we're changing the processing/device from CPU to GPU (cuda)
    device = torch.device('cpu')   

    # Batch Size is basically the number of RRTs I want to have running in parallel
    batch_size = 2

    # Now I'm defining node counts which basically tells us, how many nodes are in each tree
    # where it's one tree per batch
    node_counts = torch.ones(batch_size, dtype=torch.long, device=device)

    # Defining the start and the goal node as tensors containing the x and y positions of the
    # nodes
    start = torch.tensor([[startnode.x, startnode.y]], dtype=torch.float, device=device).repeat(batch_size, 1)
    goal = torch.tensor([[goalnode.x, goalnode.y]], dtype=torch.float, device=device).repeat(batch_size, 1)

    # Next, I'm creating the tree_parents tensor of size batch_size, NMAX
    # How it works is that for each nth node (which is defined based on the location in NMAX)
    # It stores the index of that node's parent. Currently initialized to -1 meaning that 
    # None is assigned to parents.
    tree_parents = torch.full((batch_size, NMAX), -1, dtype=torch.long, device=device)

    # Next we define the tree. For each batch, the tree will store the (x,y) positions of the 
    # nth node that has been processed. The nth node is defined based on the location of the 
    # node in NMAX

    # Tree positions is also initialized for the zeroth node as the startnode. This is equivalent
    # to saying tree = [startnode]

    tree_positions = torch.zeros((batch_size, NMAX, 2), device=device)
    tree_positions[:, 0, 0] = startnode.x
    tree_positions[:, 0, 1] = startnode.y

    iter = 0
    ########### NOW LET'S GET TO WRITING THE addtotree FUNCTION ##################
    def addtotree(valid_batches, valid_nextnodes, nearest_indices):
        # start by assigning the parent of the new_node to be the nearnode
        tree_parents[valid_batches, node_counts[valid_batches]] = nearest_indices

        # add the new node to the tree
        tree_positions[valid_batches, node_counts[valid_batches], :] = valid_nextnodes

        # increment the node cout
        next_node_index = node_counts[valid_batches]
        node_counts[valid_batches] += 1
        
        # print(tree_parents[valid_batches,:])
        # print(tree_positions[valid_batches,:])

    def addtogoal(goal_batches, goal_nodes):

        index_of_next_node = node_counts[goal_batches] - 1
        # print(f'goal_batches: {goal_batches}')
        # print(f'goal_nodes: {goal_nodes}')
        # print(f'index_of_next_node: {index_of_next_node}')

        tree_parents[goal_batches, node_counts[goal_batches]] = index_of_next_node
        
        tree_positions[goal_batches, node_counts[goal_batches],:] = goal_nodes

        node_counts[goal_batches] += 1


    ########### OKAY LET'S GO INTO THE ACUTAL LOOP LOGIC #####################
    # Also define the number of steps that have been made, since we have a constraint on the number
    # steps that are supposed to be done for each tree
    step_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    
    # ALSO DEFINE THE PROBABILITY
    p = 0.3

    # DEFINE THE FLAGS/STOPPERS
    active_batches = torch.ones(batch_size, dtype=torch.bool, device=device)
    all_goal_batches = torch.tensor([], dtype=torch.long, device=device)
    stuck_counter = 0
    ######### ENTER THE WHILE LOOP LOL #######################
    while True:
        iter += 1
        # Define the random.random() tensor
        random_tensor = torch.rand(batch_size, device=device)

        # Define the boolean meask which compares each value in the random tensor to the probability variable
        mask = (random_tensor <= p) & active_batches

        # We update the mask to consider which batches are active where when mask returns true it means 
        # that we want to sample the goalnode and we can the node is active. If we wanna sample the goal 
        # node but the active_batches for some batch is false, then the target node remains as the sample?
        

        # Define the samples tensor which does non-uniform stampling
        samples = torch.empty((batch_size, 2), device=device)
        samples[:, 0] = torch.rand(batch_size, device=device) * (xmax - xmin) + xmin
        samples[:, 1] = torch.rand(batch_size, device=device) * (ymax - ymin) + ymin
        
        # Logic to check whether mask is true or not

        # Select rows in the random_tenser where mask is true. For those rows, copy the row from goal
        # and assign it to targetnode
        targetnode = torch.clone(samples)
        targetnode[mask] = goal[mask]

        # Now that we have the targetnode for each batch, we can move on to calculating what the next node is

        # we start by doing an 'unsqueeze' process on the targetnode. I don't really know how it works but this is
        # necessary for targetnode and tree_positions to be compatible for the subtraction
        targetnode_compat = targetnode.unsqueeze(1)

        # we then calculate the differenct between the tree_positions and targetnode_compat
        diff = targetnode_compat - tree_positions

        # now we have a difference tensor with the same size as tree_positions

        # next we square each element in the difference tensor
        squared_diff = diff**2

        # and then we take the square-root of the squared_diff tensor to get the distances
        # for inactive batches, the distance is infinity
        distances = torch.sqrt(torch.sum(squared_diff, dim=2))

        # Create a mask based on nodecounts where any index in distances beyond nodecounts has its distance set to
        # zero

        # this defines the number of nodes as a column vector. Shape is (1, NMAX)
        node_indices = torch.arange(NMAX, device=device).unsqueeze(0)

        # we then define a mask to test whether node_indices its less that node_counts.unsqueeze(1). Shape is (B, NMAX)
        # node indices is a list from 0 to NMAX-1
        # node_counts defines how many nodes are currently in the tree
        # valid_mask ensures that empty unseen node spaces in node_indices (the buffer up to the max number of nodes) are not included
        # in the distance calculation. Instead the distances for those node indices are set to infinity
        
        masked_node_counts = node_counts.clone()
        masked_node_counts[~active_batches] = 0 # zero out the inactive nodes
        
        # valid mask is used to say which values should be considered finding the smallest distance by taking 
        # into account whether the node number has been seen (we don't calculate distance for nodes that don't
        # yet exist). Masked node counts has 0 where the batch is inactive an integer where the batch is active

        # So for an inactive node, node_indices < masked_node_counts will always be False for inactive batches 
        # and so the distance will be set to infinity for all nodes in that batch

        # It also retains its original function where in active batches, any node count past the actual number of 
        # nodes in the tree is set to False
        valid_mask = node_indices < masked_node_counts.unsqueeze(1)

        # apply the mask to distances to check indices
        distances[~valid_mask] = float('inf')

        # after calculating the distance we can then look for the index where the distance is the smallest
        # this is of size (batch_size, 1)
        nearest_indices = torch.argmin(distances, dim=1) # Shape is (batch_size,)
        
        batch_indices = torch.arange(batch_size, device=device) # Shape is (batch_size,)

        # next we find the node (in tree positions) that corresponds to that index
        nearnode = tree_positions[batch_indices, nearest_indices, :] # this should be of shape (batch_size, 1, 2 -->(x,y))

        # next we get the minimum distance
        min_dist = distances[batch_indices, nearest_indices]

        # next we calculate the new x and y coordinates of the next node
        nextnode = nearnode + (STEP_SIZE/min_dist).unsqueeze(1) * diff[batch_indices, nearest_indices,:]

        # Now that we have nextnode, we should then check the validity of next node by checking if it's in
        # free space and if it connects

        # first, check if the node is inFreespace()
        # we first define the node to be checked based on its x and y positions
        
        # right now, nextnode contains the coordinates for each batch
        # we can convert this tensor into a numpy array

        

        # freespace_mask_cpu is a numpy array
        freespace_mask = inFreespace(nextnode)

        # convert the numpy array to a tensor
        

        # next, we have to check if the nearnode connects to the nextnode
        # we follow a similar procesure where we first convert the nearnode tensor into a numpy array
        

        # then we send the nearnode numpy array and the nextnode numpy array to the connectsTo function
        connects_mask = connectsTo(nearnode, nextnode)

       

        # next we need to combine the masks to see, for each batch, whether the node found is valid
        next_valid_mask = freespace_mask & connects_mask & active_batches

        # get the batches where the nextnode is valid
        valid_batches = torch.where(next_valid_mask)[0]

        # get the valid nextnodes
        valid_nextnodes = nextnode[valid_batches]
        if valid_nextnodes.shape[0] == 0:
            # print('No valid nodes found - stuck')
            stuck_counter += 1

        # get only the valid nearest indices
        valid_nearest = nearest_indices[valid_batches]

        # call addtotree for only the valid batches and valid nodes
        addtotree(valid_batches, valid_nextnodes, valid_nearest)
        
        if(iter % 500 == 0):
            print(f'Now at {iter} iterations')


        if valid_nextnodes.shape[0] == 0:
            continue
        
        ##### GOAL CHECKING BLOCK #######

        # first make a mask to check whether the distance between the valid_nextnode and the goalnode
        # is within step_size

        # valid_nextnodes would be a subset of nextnodes which has shape valid_batches, (x,y)

        # Currently, the goalnode is of size (batch_size, (x,y)). Need to define valid_goal
        possible_goal = goal[valid_batches]

        # Find the Euclidean distance between the goal and each node in nextnode

        goal_diff = possible_goal - valid_nextnodes
        goal_distances = torch.norm(goal_diff, dim=1)
    
        # Next we create a mesh to see whether the distance is within the step size
        # print(f'Possible Goal: {possible_goal}')
        # print(f'Valid Next Node: {valid_nextnodes}')
        # print(f'Goal distances: {goal_distances}')
        within_threshold_mask = goal_distances < STEP_SIZE

        # Next we must create another mesh that tests whether the valid nextnode connects to the 
        # goalnode
        goal_connects_mask = connectsTo(valid_nextnodes, possible_goal)
        # compare both masks
        # print('Check mask types')
        # print(dist_mask.dtype)
        # print(f'dist_mask: {dist_mask}')
        # print(goal_connects_mask.dtype)
        # print(f'goal connects mask: {goal_connects_mask}')
        valid_goal_connects_mask = within_threshold_mask & goal_connects_mask

        # get the batch indices where the goal can be added to the tree
        goal_batches = valid_batches[valid_goal_connects_mask]
        
        goal_nodes = goal[goal_batches]
        
        # goal_indices = node_counts[goal_batches] - 1
        # goal_parents = goal_indices - 1
        
        # goal_tree_index = valid_nearest[goal_batches] + 2
        # add goal nodes to the tree
        # goal_tree_index = next_node_index[goal_batches] + 1

        if goal_batches.numel()>0:
            # print(f'Bruhhh: {goal_batches.numel()}')
            addtogoal(goal_batches, goal_nodes)

        # Set the batch to inactive
    
        # the last thing is to check the following criterion for whether to stop a batch
        # 1. The goal has been found
        

        # goal_found_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        # goal_found_mask[goal_batches] = True
            
        
            # print('Issue might come up here!')
            active_batches[goal_batches] = False
            # print(f'Goal found for batch: {goal_batches.tolist()} at Node: {node_counts[goal_batches]} and at Step: {step_counts[goal_batches]}')
            # print(tree_positions[goal_batches])
            # print(tree_parents[goal_batches])
            all_goal_batches = torch.cat([all_goal_batches, goal_batches])

        # Increment the number of steps. Recall the following step tensor definition
        # step_counts = torch.zeros(batch_size, dtype=torch.long, device=device)

        step_counts[active_batches] += 1

        # Test to see if the step counts for any batch has exceeded the maximum steps
        step_mask = step_counts >= SMAX

        # Test to see if the node counts for any batch has exceeded the maximum nodes
        node_mask = node_counts >= NMAX

        # Create the overall expired mask
        expired_mask = step_mask | node_mask

        # Apply the expired mask to active batches
        active_batches[expired_mask] = False

        # And then the last thing is that I should break/end the loop if all the batches
        # have been stopped
        if False in active_batches:
            # print(active_batches)
            pass
        if not active_batches.any():
            if (step_counts >= SMAX).all() | (node_counts >= NMAX).all():
                # print(f'Process Aborted at Node Count = {node_counts} and \nStep Count = {step_counts}. No path found')
                return None
            break

    # We're finally out of the loop!!!

    # Now we need to create the path
    
    # This path re-contruction is done not in GPU because it can't be done simultaneously
    # As a result, this means we need to convert nodecounts, tree_parents, and tree_positions
    # to numpy arrays
    tf = time.time()
    print(node_counts, step_counts)
    print(f'Stuck counter: {stuck_counter}')
    node_counts_cpu = node_counts.cpu().numpy()
    tree_parents_cpu = tree_parents.cpu().numpy()
    tree_positions_cpu = tree_positions.cpu().numpy()
    goal_indices = node_counts_cpu - 1
    all_goal_batches_cpu = all_goal_batches.cpu().numpy()
    # Let's send all this stuff to a function outside of the rrt called buildPath
    all_paths = buildPath(goal_indices, tree_parents_cpu, tree_positions_cpu, batch_size, all_goal_batches_cpu)
    return tf, all_paths


def buildPath(goal_idx, node_parents, node_positions, batch_size, goal_batches):

    # goal_idx = np_goal_idx.tolist()
    # node_parents = np_node_parents.tolist()
    # node_positions = np_node_positions.tolist()

    
    # print('Now building the path')
    path = []
    # loop through each batch
    for batch_num in range(batch_size):
        if batch_num not in goal_batches:
            print(f'No path found for batch {batch_num}')
            path.append(None)
            continue
        current_path = []
        idx = goal_idx[batch_num]
        # print(idx)
        # Skip if no path was found:
        if idx < 0 or node_parents[batch_num, idx] == -1 and idx != 0:
            path.append(None)
            continue

        while idx != -1:
            # the parent node hasn't been encountered
            # add the node to the path
            current_path.append(node_positions[batch_num, idx])
            idx = node_parents[batch_num, idx]

        current_path.reverse()
        tup_list = [tuple(float(x) for x in point) for point in current_path]
        node_list = [Node(x,y) for (x,y) in tup_list]
        path.append(node_list)
        # print(node_list)
        
        # print('Checking path validity ahhhhh')
        
        for ele in range(len(tup_list) - 1):
            
            (x1, y1) = tup_list[ele]
            (x2, y2) = tup_list[ele+1]

            dist = sqrt((x1 - x2)**2 + (y1 - y2)**2)
            # print(f'The distance between {(x1, y1)} and {(x2, y2)} is {dist}')
    return path
            
 
# Post process the path.
def PostProcess(path):
    i = 0
    while (i < len(path)-2):
        if path[i].connectsTo(path[i+2]):
            path.pop(i+1)
        else:
            i = i+1

# MAIN
def main():
    print('Running with step size ', STEP_SIZE, ' and up to ', NMAX, ' nodes.')

    # Create the figure
    visual = Visualization()

    # Create the start and goal nodes
    startnode = Node(xstart, ystart)
    goalnode = Node(xgoal, ygoal)

    # Visualize the start and goal nodes
    visual.drawNode(startnode, color='orange', marker='o')
    visual.drawNode(goalnode, color='purple', marker='o')
    visual.show('Showing basic world') 

    # Call the RRT function
    print('Running RRT')
    t0 = time.time()
    tf, paths = rrt(startnode, goalnode, visual)
    # tf = time.time()
    time_taken = tf-t0
    print(f'Time taken: {time_taken}')

    # If unable to connect path, note this
    if not paths:
        visual.show('NO PATHS FOUND')
        return
    
    # Otherwise, show the path created
    # colors = ['r', 'b', 'g', 'y','r', 'b', 'g', 'y','r', 'b', 'g', 'y','r', 'b', 'g', 'y']
    index = 0
    for batch_num, path in enumerate(paths):
        if path:
            visual.drawPath(path, color='r', linewidth=1)
            visual.show(f'Showing the raw path for batch {batch_num}')
            index += 1
        else:
            print(f'No path found for batch {batch_num}')


    # Post-process the path
    # PostProcess(path)
        
    # Show the post-processed path
    # visual.drawPath(paths, color='b', linewidth=2)
    # visual.show('Showing the post-processed path')

if __name__ == "__main__":
    # main()
    cProfile.run('main()')