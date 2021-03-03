import numpy as np

class Candidate:
    """ A Candidate center is similar to a KDNode, but only contains a wgt_cent np.ndarray, and a 'count' integer.
    """
    def __init__(self, dimensions = None, initial_pos = None):
        """
        :param initial_pos: np.ndarray of a specific point at which to initialize the candidate center, if any. 
        NOTE THAT THE POINT MUST EXCLUDE X AND Y, e.g. if the tif data has points of the form [xyrgb], then each Candidate point should be initialized with only [rgb]
        :param dimensions: The number of dimensions in each point [R,G,B,H,S,V,...] (X AND Y SHOULD NOT BE DIMENSIONS OF THE CANDIDATE)
        Either initial_pos or dimensions must be specified during initialization.  
        
        candidate.wgt_cent is the np.ndarray of the vector sum of all the points assigned to the candidate so far.
        """
        if initial_pos is not None:
            self.wgt_cent = initial_pos
        elif dimensions is not None:
            self.wgt_cent = np.zeros(dimensions)
        else:
            raise AttributeError("Either dimensions or initial_pos must be specified.")
        self.count = 1
   
    def addtree(self, tree_node):
        """
        Assign all the points of a (sub)tree to the candidate.
        :param tree_node: KDNode. Root node of the (sub)tree, whereby all points in the subtree are to be assigned to the candidate center.
        The assigning should be done if the current KDNode is a leaf node, or if there is only one candidate center remaining for the cell.
        """ 
        self.wgt_cent += tree_node.wgt_cent
        self.count += tree_node.count
    
    @property
    # This must be a property because wgt_cent and count are updated gradually
    def candidate_pos(self):
       
        return self.wgt_cent / self.count

    def is_farther(self, other_cand, cmin, cmax):
        """Check whether this candidate is further from the actual centroid of the cell than is the other candidate other_cand.
        :param other_cand: Candidate
        # :param cell: iterator over all nodes within the cell of the node that called filter().
        :param cmin: np.ndarray of the minimum value of the cell in each dimension
        :param cmax: np.ndarray of the maximum value of the cell in each dimension

        PROCESS:

        Calculate the vector z-z*, as required in Fig 2
        Let H be the bisecting hyperplane of vec.
        Find the vertex v(H) of the cell, that is EXTREME IN THE DIRECTION OF VEC.
        "Extreme" means that the k-dimensional position of v(H) has values that are 
        the highest in the direction of positive vec, compared to other vertices.
        "We take the ith coordinate of v(H) to be cmin[i] if the ith coordinate of u is negative and cmax[i] otherwise."
        z is pruned if and only if dist(z,v(H))≥dist(z*, v(H))
        because this would mean that the cell lies entirely to one side of the bisecting hyperplane H of vec.
        """
     
        # Calculating the vector z-z*, as required in Fig 2
        vec = self.candidate_pos - other_cand.candidate_pos
        print("\ncandidate pos ", self.candidate_pos)
        print("vec", vec)
        
        # Find the extreme vertex v(H)
        vH = np.zeros(shape = vec.shape)

        for idx, vec_val in np.ndenumerate(vec):
            if vec_val < 0:
                vH[idx] = cmin[idx]
            else:
                vH[idx] = cmax[idx]

        print("Extreme vertex vH is ", vH)
        # z is pruned if and only if dist(z,v(H)) >= dist(z_star, v(H)), squared distance may be used to avoid square root
        dist1 = np.linalg.norm(self.candidate_pos-vH) # Euclidean distance
        print("dist1 ", dist1)

        dist2 = np.linalg.norm(other_cand.candidate_pos-vH) # other_cand is z_star
        print("dist2", dist2)
        return dist1 >= dist2

