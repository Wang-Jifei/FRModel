import unittest
import logging

from frmodel.base.D2.frame2D import Frame2D
from frmodel.base.consts import CONSTS
from tests.base.D2.test_d2 import TestD2


class FrameTest(TestD2):

    def test_split_xy(self):
        # Split by both X and Y
        frames = self.frame.split_xy(by=self.window, method=Frame2D.SplitMethod.DROP)

        self.assertEqual((self.window, self.window, self.channels), frames[0][0].shape)

    def test_split(self):
        # Split by X Axis only, horizontal slices
        frames = self.frame.split(by=self.window, method=Frame2D.SplitMethod.DROP, axis_cut=CONSTS.AXIS.X)

        # Hence the number of rows (index 0) must be equal to window size
        self.assertEqual(self.window, frames[0].shape[0])

    def test_slide_xy(self):
        # Slide by both X and Y
        frames = self.frame.slide_xy(by=self.window, stride=self.window // 2)

        self.assertEqual((self.window, self.window, self.channels), frames[0][0].shape)

    def test_slide(self):
        # Slide by X Axis only, horizontal slices
        frames = self.frame.slide(by=self.window, stride=self.window // 2, axis_cut=CONSTS.AXIS.X)

        # Hence the number of rows (index 0) must be equal to window size
        self.assertEqual(self.window, frames[0].shape[0])
    
    def runTest(self):
        log= logging.getLogger("TestD2.setUp")
        log.debug(f"original frame shape: {self.frame.shape}, len(frame_window): {len(self.frame.split_xy(100))}, len(frame_window[0]: {len(self.frame.split_xy(100)[0])}, shape of one Frame2D: {self.frame_window.shape} ({type(self.frame_window)})")
        self.assertEqual(1,1)

if __name__ == '__main__':
    unittest.main()
c