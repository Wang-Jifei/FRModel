from candidate import Candidate
from kdnode import KDNode
from math import sqrt
import numpy as np

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
        if dist < min_dist:
            min_dist = dist
            zstar = cand
    return zstar

def child_axis_calculator(parent_axis, dimensions):
    """Determines the next axis to split the cell by, by just taking turns. 
    
    :param parent_axis: The most recent axis that the cell has been split by
    :param dimensions: The number of dimensions in each point (due to get_chns())
   
    TODO: Implement sliding midpoint rule, in which case the 
    child_axis_calculator must return the longest axis of the cell, not just taking turns for each axis
    """
    return (parent_axis+1)%dimensions

def construct_kdtree(point_arr, axis, leaf_size=1):
    """Creates a kd tree from an array of points.
    :param point_arr: Flattened image array of points
    :param axis: Axis to split the cell by

    Note regarding point_arr: For a 2D image, the frame data would be a 3D array of shape (height, width, num_channels)
    where num_channels is obtained by calling frame.get_chns(). Before a kd tree can be constructed, 
    flatten the image to 2D array of shape (heigth * width, num_channels), using frame.data_flatten().  
    Then call construct_kdtree().

    Given the specified axis to split the cell by, we find the median value of the cell in this axis.
    A new node will be created, using the point that has the median value. 
    The LHS and RHS children of this node are constructed pre-order recursively.

    # TODO: Instead of median, the sliding midpoint rule takes midpoint, while also ensuring not all points lie to one side
    See here for illustration: https://www.researchgate.net/figure/The-Sliding-Midpoint-rule-avoids-trivial-splits-2_fig3_251492991
    And here for explanation: https://github.com/halfhorst/katy
    """
    if point_arr.size == 0: # If previous node is leaf, then the current point_arr is empty
        return None

    
    # Find the point that has the median value in the specified splitting axis
    median_idx = point_arr[0].shape[0] // 2
    partial_sort_idx = np.argpartition(point_arr[:, axis], median_idx)
    median_point = point_arr[partial_sort_idx][median_idx,:]
    partial_sorted_point_arr = point_arr[partial_sort_idx]
    
    # Recursively create the child nodes
    dimensions = point_arr[0].shape[1]
    next_axis = child_axis_calculator(axis, dimensions)
    left = construct_kdtree(partial_sorted_point_arr[:median_idx], next_axis)
    right = construct_kdtree(partial_sorted_point_arr[median_idx+1:], next_axis)

    return KDNode(median_point, left, right, axis)
