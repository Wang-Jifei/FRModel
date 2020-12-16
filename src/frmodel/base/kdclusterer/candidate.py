import numpy as np

class Candidate:
    """ A Candidate center is similar to a KDNode, but only contains a wgt_cent np.ndarray, and a 'count' integer.
    """
    def __init__(self, dimensions = None, initial_pos = None):
        """
        :param initial_pos: A specific point at which to initialize the candidate center, if any. 
        :param dimensions: The number of dimensions in each point [R,G,B,X,Y,H,S,V,...]
        Either initial_pos or dimensions must be specified during initialization.  
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

    def is_farther(self, other_cand, cell):
        """Check whether this candidate is further from the actual centroid of the cell than is the other candidate other_cand.
        :param other_cand: Candidate
        :param cell: iterator

        PROCESS:

        Calculate the vector z-z*, as required in Fig 2
        Let H be the bisecting hyperplane of vec.
        Find the minimum and maximum value of the cell in each dimension
        i.e. calculate [Cmin_i, Cmax_i] for each dimension i
        so that we can find the vertex v(H) of the cell, that is EXTREME IN THE DIRECTION OF VEC.
        This means that the k-dimensional position of v(H) has values that are 
        the highest in the direction of positive vec, compared to other vertices.
        z is pruned if and only if dist(z,v(H))≥dist(z*, v(H))
        because this would mean that the cell lies entirely to one side of the bisecting hyperplane H of vec.
        """
        if self == other_cand:
            return False
        
        # Calculating the vector z-z*, as required in Fig 2
        vec = self.candidate_pos - other_cand.candidate_pos
        
        # Initializing the minimum and maximum values of the cell in each dimension
        cmin = next(cell).data.tolist()
        cmax = cmin.copy()

        # Get the actual minimum and maximum coordinate of the cell so as to find the extreme vertex v(H)
        for kd_node in cell:
            for idx, val in enumerate(kd_node.data): # TODO: optimize
                cmin[idx] = min(cmin[idx], val) 
                cmax[idx] = max(cmax[idx], val)

        # Find the extreme vertex v(H)
        # "We take the ith coordinate of v(H) to be Cmin[i] if the ith coordinate of u is negative and Cmax[i] otherwise."
        vH = np.array([])
        for idx, vec_val in vec:
            if vec_val < 0:
                np.append(vH, cmin[idx])
            else:
                np.append(vH, cmax[idx])
        
        # z is pruned if and only if dist(z,v(H)) >= dist(z_star, v(H)), squared distance may be used to avoid squaroot
        dist1 = np.linalg.norm(self.candidate_pos, vH) # Euclidean distance
        dist2 = np.linalg.norm(other_cand.candidate_pos, vH)
        return dist1 >= dist2

