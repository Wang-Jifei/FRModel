import logging
from frmodel.base.kdclusterer.candidate import Candidate
from frmodel.base.kdclusterer import kdnode 
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

def get_closest_candidate(candidate_centers_set, actual_cent):
    """ During the filtering algorithm, finds the z star for a certain cell. Refer to Fig 2 in the paper. 

    :param candidate_centers_set: Set of Candidates for the cell 
    :param actual_centroid: np.ndarray of u.wgt_cent/u.count where u is the node in the kd tree that is associated with this cell
    Returns the Candidate that is the closest to the actual center of the cell, i.e. z star
    """
    min_dist = float('inf')
    zstar = None 
    for cand in candidate_centers_set:
        dist = np.linalg.norm(cand.candidate_pos - actual_cent)
        print("minused: ", cand.candidate_pos - actual_cent)
        print("cand pos: ", cand.candidate_pos)
        print("actual_cent: ", actual_cent)
        print("dist is ", dist)
        if dist < min_dist:
            min_dist = dist
            zstar = cand
    return zstar

def child_axis_calculator(parent_axis, axes_to_pick_from):
    """Determines the next axis to split the cell by, by just taking turns. 
    
    :param parent_axis: The most recent axis that the cell has been split by
    :param axes_to_pick_from: List of axis indices to pick from in turn sequence (starting from 0) 
   
    TODO: Implement sliding midpoint rule, in which case the 
    child_axis_calculator must return the longest axis of the cell, not just taking turns for each axis
    """
    current_idx = axes_to_pick_from.index(parent_axis)
    next_idx = (current_idx+1)%len(axes_to_pick_from)
    return axes_to_pick_from[next_idx]

def construct_kdtree(point_arr, axis, x_pos=0, y_pos=1, leaf_size=1, child = None):
    """Creates a kd tree from an array of points.
    :param point_arr: Flattened image array of points, where each point is in the format of e.g. [xyrgb]
    :param axis: Axis to split the cell by
    
    Note regarding point_arr: For a 2D image, the frame data would be a 3D array of shape (height, width, num_channels)
    where num_channels is obtained by calling frame.get_chns(). Before a kd tree can be constructed, 
    flatten the image to 2D array of shape (height * width, num_channels), using frame.data_flatten().  
    Then call construct_kdtree().

    Given the specified axis to split the cell by, we find the median value of the cell in this axis.
    A new node will be created, using the point that has the median value. 
    The LHS and RHS children of this node are constructed post-order recursively.
    THE X AND Y COORDINATES OF THE POINTS ARE NOT USED FOR SPLITTING. 

    # TODO: Instead of median, the sliding midpoint rule takes midpoint, while also ensuring not all points lie to one side
    See here for illustration: https://www.researchgate.net/figure/The-Sliding-Midpoint-rule-avoids-trivial-splits-2_fig3_251492991
    And here for explanation: https://github.com/halfhorst/katy
    """
    
    if point_arr.size == 0: # If previous node is leaf, then the current point_arr is empty
        return None
    
    # Find the point that has the median value in the specified splitting axis
    median_idx = point_arr.shape[0] // 2
    partial_sort_idx = np.argpartition(point_arr[:, axis], median_idx)
    median_point = point_arr[partial_sort_idx][median_idx,:] # xyrgb point
    # print(f"median point: {median_point}")
    partial_sorted_point_arr = point_arr[partial_sort_idx]

    # Recursively create the child nodes
    dimensions = point_arr.shape[-1]
    axes_to_pick_from = [i for i in range(dimensions) if i not in [x_pos, y_pos]]
    next_axis = child_axis_calculator(axis, axes_to_pick_from) # if there are 3 dimensions (rgb), next_axis is 0, 1 or 2
    left = construct_kdtree(partial_sorted_point_arr[:median_idx], axis = next_axis, x_pos = x_pos, y_pos = y_pos)
    # TODO: if there is more than one node with the same value in that splitting axis, 
    # then the node at the median_idx will be different each time -->
    # which means that kd trees are not unique, if the value of the nodes (at each different splitting axis) is not unique (?).
    # Is it possible to reconstruct the image from the kd tree if the kd tree is different each time?
    #  --> i think must we record the actual ax and y coords of each kd node of the kd tree

    right = construct_kdtree(partial_sorted_point_arr[median_idx+1:], axis = next_axis, x_pos = x_pos, y_pos = y_pos)

    newnode = kdnode.KDNode(median_point, x_pos = x_pos, y_pos = y_pos, left = left, right = right, axis = axis)
    if left:
        newnode.left.parent = newnode
        newnode.left.child = "left"
    if right:
        newnode.right.parent = newnode
        newnode.right.child = "right"

    return newnode


# def make_plot(root, width, height, mapping_dict):
#     #https://stackoverflow.com/questions/62362807/plotting-multi-class-semantic-segmentation-transparent-overlays-over-rgb-image
#     """Allows us to see what clusters have been generated
#     :param root: The root node of the kd tree.
#     Note that this should only be run after running root.filter(candidate_centers_set)
#     """
    
#     all_assigned = [n.assigned for n in root.cell]
#     def cand_to_int(mapping_dict, cand):
#         return mapping_dict[cand]

#     all_assigned = [cand_to_int(mapping_dict, c) for c in all_assigned]
   
#     all_x = np.arange(width)
#     all_x = np.tile(all_x,height)
#     all_y = np.arange(height-1, stop = -1, step = -1)[np.newaxis, :].transpose()
#     all_y = np.tile(all_y, (1,width)).flatten()
#     print(all_y)
#     fig, ax = plt.subplots()
#     ax.scatter(all_x, all_y, c = all_assigned,  alpha = 0.5)
  
#     ax.set_aspect("equal")
#     plt.show()

def get_minmax(kd_node):
    """Find the minimum and maximum value of the cell whose root is `kd_node`
    i.e. calculate [Cmin_i, Cmax_i] for each dimension i

    :param kd_node: The node that is the root of the (sub)tree that represents the cell
    """

    # Initializing the minimum and maximum values of the cell in each dimension
    cmin = np.full(shape = kd_node.wgt_cent.shape, fill_value = np.inf)
    cmax = np.full(shape = kd_node.wgt_cent.shape, fill_value = -np.inf)
    
    # Get the actual minimum and maximum coordinate of the cell 
    for n in kd_node.cell:

        for idx, val in np.ndenumerate(n.data): # TODO: optimize
            if val < cmin[idx]:
                cmin[idx] = val
            if val > cmax[idx]:
                cmax[idx] = val
    print("cmin ", cmin)
    print("cmax ", cmax)
    return cmin, cmax

