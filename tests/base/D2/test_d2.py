import unittest
import logging
from frmodel.base.D2 import Frame2D
import os
import sys

_RSC = os.path.dirname(os.path.realpath(__file__)) + "/../../../rsc"

class TestD2(unittest.TestCase):

    @classmethod
    def setUp(cls) -> None:
        cls.frame = Frame2D.from_image(f"{_RSC}/imgs/chestnut_0/screenshot1.png")
        cls.frame.data = cls.frame.data[...,:3] # REMOVE THIS if not a screenshot
        cls.frame_window = cls.frame.split_xy(100)[0][0]
        cls.window = 100
        cls._RSC = _RSC
        cls.channels = 3
        cls.frame2 = cls.frame.get_chns(xy=True, hsv=True, mex_g=True, ex_gr=True, ndi=True)

    def runTest(self):
        log= logging.getLogger("TestD2.setUp")
        log.debug(f"original frame shape: {self.frame.shape}, len(frame_window): {len(self.frame.split_xy(100))}, \
                    len(frame_window[0]: {len(self.frame.split_xy(100)[0])}, shape of one Frame2D: {self.frame_window.shape} ({type(self.frame_window)})")
        log.debug(f"frame2.data.shape: {self.frame2.data.shape}")
        
        self.assertEqual(1,1)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr)
    logging.getLogger("TestD2.setUp").setLevel(logging.DEBUG)
    unittest.main()
