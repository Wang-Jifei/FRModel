# Testing for construct_kdtree() in kanungo_utils
import unittest
import logging
import sys
from frmodel.base.kdclusterer.kanungo_utils import *
from frmodel.base.D2 import Frame2D
from tests.base.D2.test_d2 import TestD2
import numpy as np


class TestKUtils(TestD2, unittest.TestCase):

    # def setUp(self): 
    #     super(TestD2, self).setUp()
    #     self.flatdata = self.frame2.data_flatten()
    #     self.root = construct_kdtree(self.flatdata)

    # FOR TESTING QUICKLY
    def setUp(cls): # setUp, not setUpClass, otherwise each child instance will have different data
        cls.num_points = 100*100
        cls.num_channels = 2
        cls.flatdata = np.random.random(size=(cls.num_points, cls.num_channels)) # assume minmax scaling of frame.data
        cls.root = construct_kdtree(cls.flatdata)
   
    def test_kutils(self):
        log = logging.getLogger("TestKUtils.setUp")
        log.debug(f"flat data shape: {self.flatdata.shape}")
        log.debug(f"flat data: {self.flatdata}")
        
        log.debug(f"root.data: {self.root.data}")
        log.debug(f"root.wgt_cent: {self.root.wgt_cent}")
        log.debug(f"root.count: {self.root.count}")
        log.debug(f"root.actual_cent: {self.root.actual_cent}")

        log.debug(f"root.left: {self.root.left}")
        log.debug(f"root.right: {self.root.right}")
        log.debug(f"root.left.data: {self.root.left.data}")
        # log.debug("root.cell:\n") # includes the root node of the (sub)tree 
        # for n in self.root.cell:
        #     log.debug(n.data)
        self.assertEqual(1,1)

if __name__ == "__main__":
    logging.basicConfig(stream = sys.stderr)
    import os
    logging.getLogger("TestKUtils.setUp").setLevel(logging.DEBUG)
    log = logging.getLogger("TestKUtils.setUp")
    log.debug(f"In test_utils.py: {os.getcwd()}")
    unittest.main()