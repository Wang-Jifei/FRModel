# Copyright Anne M. Archibald 2008
# Released under the scipy license
import numpy as np
import warnings
from .ckdtree import cKDTree, cKDTreeNode

__all__ = ['minkowski_distance_p', 'minkowski_distance', 'Rectangle', 'KDTree']


def minkowski_distance_p(x, y, p=2):
    """
    Compute the pth power of the L**p distance between two arrays.
    For efficiency, this function computes the L**p distance but does
    not extract the pth root. If `p` is 1 or infinity, this is equal to
    the actual L**p distance.
    Parameters
    ----------
    x : (M, K) array_like
        Input array.
    y : (N, K) array_like
        Input array.
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use.
    Examples
    --------
    >>> from scipy.spatial import minkowski_distance_p
    >>> minkowski_distance_p([[0,0],[0,0]], [[1,1],[0,1]])
    array([2, 1])
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Find smallest common datatype with float64 (return type of this function) - addresses #10262.
    # Don't just cast to float64 for complex input case.
    common_datatype = np.promote_types(np.promote_types(x.dtype, y.dtype), 'float64')

    # Make sure x and y are NumPy arrays of correct datatype.
    x = x.astype(common_datatype)
    y = y.astype(common_datatype)

    if p == np.inf:
        return np.amax(np.abs(y-x), axis=-1)
    elif p == 1:
        return np.sum(np.abs(y-x), axis=-1)
    else:
        return np.sum(np.abs(y-x)**p, axis=-1)


def minkowski_distance(x, y, p=2):
    """
    Compute the L**p distance between two arrays.
    Parameters
    ----------
    x : (M, K) array_like
        Input array.
    y : (N, K) array_like
        Input array.
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use.
    Examples
    --------
    >>> from scipy.spatial import minkowski_distance
    >>> minkowski_distance([[0,0],[0,0]], [[1,1],[0,1]])
    array([ 1.41421356,  1.        ])
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if p == np.inf or p == 1:
        return minkowski_distance_p(x, y, p)
    else:
        return minkowski_distance_p(x, y, p)**(1./p)


class Rectangle(object):
    """Hyperrectangle class.
    Represents a Cartesian product of intervals.
    """
    def __init__(self, maxes, mins):
        """Construct a hyperrectangle."""
        self.maxes = np.maximum(maxes,mins).astype(float)
        self.mins = np.minimum(maxes,mins).astype(float)
        self.m, = self.maxes.shape

    def __repr__(self):
        return "<Rectangle %s>" % list(zip(self.mins, self.maxes))

    def volume(self):
        """Total volume."""
        return np.prod(self.maxes-self.mins)

    def split(self, d, split):
        """
        Produce two hyperrectangles by splitting.
        In general, if you need to compute maximum and minimum
        distances to the children, it can be done more efficiently
        by updating the maximum and minimum distances to the parent.
        Parameters
        ----------
        d : int
            Axis to split hyperrectangle along.
        split : float
            Position along axis `d` to split at.
        """
        mid = np.copy(self.maxes)
        mid[d] = split
        less = Rectangle(self.mins, mid)
        mid = np.copy(self.mins)
        mid[d] = split
        greater = Rectangle(mid, self.maxes)
        return less, greater

    def min_distance_point(self, x, p=2.):
        """
        Return the minimum distance between input and points in the hyperrectangle.
        Parameters
        ----------
        x : array_like
            Input.
        p : float, optional
            Input.
        """
        return minkowski_distance(0, np.maximum(0,np.maximum(self.mins-x,x-self.maxes)),p)

    def max_distance_point(self, x, p=2.):
        """
        Return the maximum distance between input and points in the hyperrectangle.
        Parameters
        ----------
        x : array_like
            Input array.
        p : float, optional
            Input.
        """
        return minkowski_distance(0, np.maximum(self.maxes-x,x-self.mins),p)

    def min_distance_rectangle(self, other, p=2.):
        """
        Compute the minimum distance between points in the two hyperrectangles.
        Parameters
        ----------
        other : hyperrectangle
            Input.
        p : float
            Input.
        """
        return minkowski_distance(0, np.maximum(0,np.maximum(self.mins-other.maxes,other.mins-self.maxes)),p)

    def max_distance_rectangle(self, other, p=2.):
        """
        Compute the maximum distance between points in the two hyperrectangles.
        Parameters
        ----------
        other : hyperrectangle
            Input.
        p : float, optional
            Input.
        """
        return minkowski_distance(0, np.maximum(self.maxes-other.mins,other.maxes-self.mins),p)


class KDTree(cKDTree):
    """
    kd-tree for quick nearest-neighbor lookup
    This class provides an index into a set of k-dimensional points
    which can be used to rapidly look up the nearest neighbors of any
    point.
    Parameters
    ----------
    data : array_like, shape (n,m)
        The n data points of dimension m to be indexed. This array is
        not copied unless this is necessary to produce a contiguous
        array of doubles, and so modifying this data will result in
        bogus results. The data are also copied if the kd-tree is built
        with copy_data=True.
    leafsize : positive int, optional
        The number of points at which the algorithm switches over to
        brute-force.  Default: 10.
    compact_nodes : bool, optional
        If True, the kd-tree is built to shrink the hyperrectangles to
        the actual data range. This usually gives a more compact tree that
        is robust against degenerated input data and gives faster queries
        at the expense of longer build time. Default: True.
    copy_data : bool, optional
        If True the data is always copied to protect the kd-tree against
        data corruption. Default: False.
    balanced_tree : bool, optional
        If True, the median is used to split the hyperrectangles instead of
        the midpoint. This usually gives a more compact tree and
        faster queries at the expense of longer build time. Default: True.
    boxsize : array_like or scalar, optional
        Apply a m-d toroidal topology to the KDTree.. The topology is generated
        by :math:`x_i + n_i L_i` where :math:`n_i` are integers and :math:`L_i`
        is the boxsize along i-th dimension. The input data shall be wrapped
        into :math:`[0, L_i)`. A ValueError is raised if any of the data is
        outside of this bound.
    Notes
    -----
    The algorithm used is described in Maneewongvatana and Mount 1999.
    The general idea is that the kd-tree is a binary tree, each of whose
    nodes represents an axis-aligned hyperrectangle. Each node specifies
    an axis and splits the set of points based on whether their coordinate
    along that axis is greater than or less than a particular value.
    During construction, the axis and splitting point are chosen by the
    "sliding midpoint" rule, which ensures that the cells do not all
    become long and thin.
    The tree can be queried for the r closest neighbors of any given point
    (optionally returning only those within some maximum distance of the
    point). It can also be queried, with a substantial gain in efficiency,
    for the r approximate closest neighbors.
    For large dimensions (20 is already large) do not expect this to run
    significantly faster than brute force. High-dimensional nearest-neighbor
    queries are a substantial open problem in computer science.
    Attributes
    ----------
    data : ndarray, shape (n,m)
        The n data points of dimension m to be indexed. This array is
        not copied unless this is necessary to produce a contiguous
        array of doubles. The data are also copied if the kd-tree is built
        with `copy_data=True`.
    leafsize : positive int
        The number of points at which the algorithm switches over to
        brute-force.
    m : int
        The dimension of a single data-point.
    n : int
        The number of data points.
    maxes : ndarray, shape (m,)
        The maximum value in each dimension of the n data points.
    mins : ndarray, shape (m,)
        The minimum value in each dimension of the n data points.
    size : int
        The number of nodes in the tree.
    """

    class node:
        @staticmethod
        def _create(ckdtree_node=None):
            """Create either an inner or leaf node, wrapping a cKDTreeNode instance"""
            if ckdtree_node is None:
                return KDTree.node(ckdtree_node)
            elif ckdtree_node.split_dim == -1:
                return KDTree.leafnode(ckdtree_node)
            else:
                return KDTree.innernode(ckdtree_node)

        def __init__(self, ckdtree_node=None):
            if ckdtree_node is None:
                ckdtree_node = cKDTreeNode()
            self._node = ckdtree_node

        def __lt__(self, other):
            return id(self) < id(other)

        def __gt__(self, other):
            return id(self) > id(other)

        def __le__(self, other):
            return id(self) <= id(other)

        def __ge__(self, other):
            return id(self) >= id(other)

        def __eq__(self, other):
            return id(self) == id(other)

    class leafnode(node):
        @property
        def idx(self):
            return self._node.indices

        @property
        def children(self):
            return self._node.children

    class innernode(node):
        def __init__(self, ckdtreenode):
            assert isinstance(ckdtreenode, cKDTreeNode)
            super().__init__(ckdtreenode)
            self.less = KDTree.node._create(ckdtreenode.lesser)
            self.greater = KDTree.node._create(ckdtreenode.greater)

        @property
        def split_dim(self):
            return self._node.split_dim

        @property
        def split(self):
            return self._node.split

        @property
        def children(self):
            return self._node.children

    @property
    def tree(self):
        if not hasattr(self, "_tree"):
            self._tree = KDTree.node._create(super().tree)

        return self._tree

    def __init__(self, data, leafsize=10, compact_nodes=True, copy_data=False,
                 balanced_tree=True, boxsize=None):
        data = np.asarray(data)
        if data.dtype.kind == 'c':
            raise TypeError("KDTree does not work with complex data")

        # Note KDTree has different default leafsize from cKDTree
        super().__init__(data, leafsize, compact_nodes, copy_data,
                         balanced_tree, boxsize)
