import numpy as np
from itertools import chain
from frmodel.base.kdclusterer import kanungo_utils
from frmodel.base.kdclusterer.candidate import Candidate

class KDNode:
    def __init__(self, data, left=None, right=None, axis=0, child = None, parent = None):
        """ Creates a new node in the kd-tree, with implementation of the filtering algorithm,
        whereby candidate centers for each node are pruned, or "filtered",
        as they are propagated to the node's children.

        :param data: np.ndarray # One point [R,G,B,X,Y,H,S,V,EX_G, ...] as defined in _frame_channel.py
        :param left: KDNode
        :param right: KDNode
        :param axis: int # axis of splitting the kd space
        :param child: "left" or "right", i.e. whether the root node of this kd subtree is the left of right child of the parent node. None if the node is the root node
        """
       
        self.data = data
        self.left = left
        self.right = right
        self.axis = axis
        self.child = child
        self.parent = parent
        self.dimensions = len(self.data.shape)

        self.count = 1 # u.count in the paper, the number of points in the cell whose root is this KDNode
        self.wgt_cent = np.array(data, copy = True) # u.wgtCent in the paper, the vector sum of all points in the cell
        # NOT self.wgt_cent = data because otherwise any updates to wgt_cent will cause self.data to change as well

        if left is not None:
            self.count += left.count
            self.wgt_cent += left.wgt_cent
        if right is not None:
            self.count += right.count
            self.wgt_cent += right.wgt_cent
        
        self.actual_cent = self.wgt_cent / self.count # the actual centroid of the cell whose root is this KDnode

        self.candidate_centers = []
        self.assigned = None

    def is_leaf(self):
        return self.count ==1

    @property
    def children(self):
        """
        Returns an iterator for the non-empty children of the Node
        The children are returned as (Node, pos) tuples where pos is 0 for the
        left subnode and 1 for the right.
        """

        if self.left and self.left.data:
            yield self.left, 0
        if self.right and self.right.data:
            yield self.right, 1
    @property
    def cell(self):
        """
        Returns an iterator for over all the nodes contained in the tree
        """

        def me():
            yield self

        # iterator = me() # can do in-order???

        # if self.right:
        #     iterator = chain(iterator, self.right.cell)#, iterator
        # if self.left:
        #     iterator = chain(iterator, self.left.cell) #, iterator

        if self.left:
            iterator = chain(self.left.cell, me())
        else:
            iterator = me()
        if self.right:
            iterator = chain(iterator, self.right.cell)
        
        return iterator
   
    def height(self):
        """
        Returns height of the (sub)tree, without considering empty leaf-nodes
        """
        min_height = int(bool(self))
        return max([min_height] + [c.height() + 1 for c, p in self.children])
    
    def assign(self,z_star):
        """Assign a node to the closest center. This occurs if 
        1. The node is a leaf node OR
        2. A (grand)parent node above this node has only one candidate center remaining, i.e. we can assign the entire subtree to this center
        
        :param z_star: The Candidate that the node is assigned to
        """
        self.assigned = z_star
   
    def filter(self, candidate_centers_set):    
        """Prunes the candidate centers for this node. 
        The candidate centers of the node is stored in self.candidate_centers, which is a set.
        If there is only one possible center in self.candidate_centers, 
        (i.e. if the node is a leaf node, or if every point in the cell is closer to z_star than to other Candidates), 
        then the assigned Candidate of the node is stored in self.assigned.

        It is only necessary to call rootnode.filter(), because all nodes of the kd tree will recursively undergo the pruning.
        
        :param candidate_centers_set: Set of Candidates that we can choose from (inherited from the parent node)
        
        
        TODO: Implement density-based cluster seeding to generate the candidate_centers_set.
        """
        print("self.data: ", self.data)
        z_star = kanungo_utils.get_closest_candidate(candidate_centers_set, self.actual_cent)
        print("z_star.candidate_pos: ", z_star.candidate_pos)
        if self.is_leaf():
            # self.assign(z_star)
            new_candidate_centers_set = set([z_star])
        else:
            new_candidate_centers_set = candidate_centers_set.copy() 
            # must .copy() because we use filter() for left child branch, then for right child branch. Do not want left child filter to affect right child filter
            # Python is a PASS-BY-ASSIGNMENT programming language 
            temp = candidate_centers_set.copy()
            temp.discard(z_star)
            cmin, cmax = kanungo_utils.get_minmax(self)
            
            for z in temp:
                if z.is_farther(z_star, cmin, cmax):
                    new_candidate_centers_set.discard(z)
                    print(f"Pruned {z.candidate_pos}")
            
        if len(new_candidate_centers_set) == 1:
            z_star.addtree(self) # update the position of z_star
            for node in self.cell: # assign the leaf node, or assign the entire subtree, to this candidate
                node.assign(z_star)
                node.candidate_centers = new_candidate_centers_set
        else:
            self.candidate_centers = new_candidate_centers_set
            if self.left:
                print(f"\nAt left of {self.data} now\n")
                self.left.filter(new_candidate_centers_set) 
            if self.right:
                print(f"\nAt right of {self.data} now\n")
                self.right.filter(new_candidate_centers_set)

            #  ADD THE ASSIGNMENT OF THE PARENT NODES HERE
            self.assigned = z_star # TODO: help idk
        

    



