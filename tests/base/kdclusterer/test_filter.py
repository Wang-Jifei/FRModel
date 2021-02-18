# Testing out the filtering algorithm where Candidates are successively pruned
import logging
from frmodel.base.kdclusterer.kanungo_utils import *
from frmodel.base.kdclusterer.kdnode import KDNode
from frmodel.base.kdclusterer.candidate import Candidate
from test_utils import TestKUtils
from tests.base.D2.test_d2 import TestD2
import unittest
import numpy as np
import sys
from PIL import Image

class TestFilter(TestKUtils, TestD2, unittest.TestCase):
    def test_filter(self):
        log = logging.getLogger("TestFilter.test_filter")
    
        k = 3
        candidate_centers_set = set()
        mapping_dict = {}
        for i in range(1, k+1):
            new_cand = Candidate(initial_pos = np.random.random(size = self.flatdata[0].shape))
            print("Initial position of candidate: ", new_cand.candidate_pos)
            candidate_centers_set.add(new_cand)
            mapping_dict[new_cand] = i

        log.debug(f"flatdata: {self.flatdata}")
        self.root.filter(candidate_centers_set)
        log.debug(f"\n\nroot.candidate_centers: {self.root.candidate_centers}")
        log.debug(f"root.right.candidate_centers: {self.root.right.candidate_centers}")
        log.debug(f"root.assigned: {self.root.assigned}")
        log.debug(f"root.right.assigned: {self.root.right.assigned}")
        log.debug(f"root.left.assigned: {self.root.left.assigned}")
        log.debug(f"root.left.left.assigned: {self.root.left.left.assigned}")
        
        make_plot(self.root, self.frame_window.width(), self.frame_window.height(), mapping_dict)


if __name__ == "__main__":
    logging.basicConfig(stream = sys.stderr)
    logging.getLogger("TestFilter.test_filter").setLevel(logging.DEBUG)

    unittest.main()
