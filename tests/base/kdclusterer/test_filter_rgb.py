# See that the kd tree is constructed correctly
# And that the 4 candidates are at R,G,B,W respectively.
# Then reconstruct img from kd tree

# NOTE: THIS TESTING WAS DONE BEFORE KDNode class was defined to hold attributes x and y

from frmodel.base.kdclusterer.kanungo_utils import *
from frmodel.base.kdclusterer.kdnode import KDNode
from frmodel.base.kdclusterer.candidate import Candidate
from frmodel.base.D2.frame2D import Frame2D
import numpy as np
from PIL import Image
import sys
import gdal

def main():
    img_arr = []
    '''
    point_dict = {
        "R1": [255, 0, 0], "R2": [255, 0, 0],
        "G1": [0, 255, 0], "G2": [0, 255, 0],
        "B1": [0, 0, 255], "B2": [0, 0, 255],
        "W1": [255, 255, 255], "W2": [255, 255, 255]
    }

    # Why the filtering algorithm doesn't work well if we just use those 8 points:
    # After the filtering algorithm, 
    # the coordinates of the candidate centers are not updated enough, since we only have 8 points and thus 
    # there are very few rounds of coordinate updation.
    # Kanungo's kmeans is not iterative, but rather we just go down the kd tree once,
    # and whatever candidate center coordinates we get are final.
    # As a result, the coordinates of the candidate centers obtained after the filtering algorithm 
    # are not a good reflection of whether Kanungo's kmeans is working.

    Thus, let's use 4000 points instead:
    '''
    for p in [[255,0,0], [0,255,0], [0,0,255], [0,0,0]]:
        for i in range(1000):
            img_arr.append(p)

    img_arr = np.array(img_arr).reshape(4000,1,3) # pretend its a 4000x1 image
    img = Image.fromarray(img_arr.astype(np.uint8))

    img_path = sys.path[0] + "/testimgs/rgb_test.png"
    img.save(img_path)

    frame = Frame2D.from_image(img_path)
    flatdata = frame.normalize().data_flatten() # TODO: Normalization should be along per-channel mean and std dev, not per pixel?
  
    root = construct_kdtree(flatdata)

    '''
    # UNCOMMENT THIS TO SEE KD TREE BUILDING
    for p in root.cell: # updated: changed root.cell to in-order traversal 
        if p is root:
            print("ROOT", end = " ")
        else:
            print(p.child, " child of ", p.parent)
        print(p, p.data, p.axis)
        print()
    '''

    k = 4 # RGBW
    candidate_centers_set = set()
    mapping_dict = {}

    for i in range(1, k+1):
        '''RANDOM CENTROID INITIALIZATION IN THE VECTOR SPACE'''
        # new_cand = Candidate(initial_pos = np.random.random(size = flatdata.shape[-1]))

        '''MIN-MAX CENTROID INITIALIZATION'''
        rmin, rmax = get_minmax(root) # root min, root max
        new_cand = Candidate(initial_pos = np.random.uniform(low= rmin, high = rmax, size = flatdata.shape[-1]))

        print("Initial position of candidate: ", new_cand.candidate_pos)
        candidate_centers_set.add(new_cand)
        mapping_dict[new_cand] = i

    root.filter(candidate_centers_set)


    all_assigned = [mapping_dict[cand] for cand in [n.assigned for n in root.cell]]
    print(f"All assigned: {all_assigned}")

    unique, counts = np.unique(all_assigned, return_counts=True)
    print(f"Number of points assigned to each Candidate: {dict(zip(unique, counts))}")

    for c in root.candidate_centers:
        print(c.candidate_pos)

    '''
    Output with random centroid initialization in the vector space:

    all assigned [all 4]
    [0.0079524  0.00792994 0.00802135]
    --> i.e. we pruned away 3 Candidates right at the root node! This is because for the root.cell, we have
    cmin  [0. 0. 0.]
    cmax  [0.03162278 0.03162278 0.03162278]

    Then the initial positions of the candidates were:
    1: [0.50428013 0.04648459 0.62319163]
    2: [0.85799202 0.88322259 0.22078305]
    3: [0.32975803 0.16971674 0.60219882]
    4: [0.19475901 0.10490933 0.47063211]

    So Candidate 4 is the centroid of all the points, starting at the root node itself....
    '''

    '''
    Output with min-max centroid initialization:

    Number of points assigned to each Candidate: {1: 1000, 2: 1000, 3: 999, 4: 1001}

    Final positions of Candidates:
    [3.59851841e-06 7.97518144e-06 1.56925009e-06] --> [0,0,0]
    [3.16216404e-02 7.26893404e-06 1.90527575e-05] --> [255,0,0]
    [1.92995782e-05 1.21297311e-05 3.16013754e-02] --> [0,0,255]
    [5.17085352e-06 3.16102713e-02 6.95368188e-06] --> [0,255,0]
    '''

    # Q: Can we reconstruct a point cloud from a kd tree? 
    # No, because we did not use x and y information in the points of the kd tree.
    # By using just RGB data in the kd tree, we can only rebuild a weird RGB representation of the point cloud that 
    # does not reflect any real-world locality information.

    # TODO: Think of a way to give the original points in the point cloud a "point id",
    # and create kd nodes without the x and y information, but still carry the "point id"
    # so that we can map the assigned Candidates (obtained after Kanungo's kmeans) back to the 
    # original points in the point cloud.
    # --> see test_filter_tif.py

if __name__ == "__main__":
    main()