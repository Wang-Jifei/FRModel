import unittest

from rsc.rsc_paths import *
from FRModel.base.consts import CONSTS
from FRModel.base.D2.frame2D import Frame2D
import numpy as np
class GLCMTest(unittest.TestCase):
    def test(self):

        frame = Frame2D.from_image(SAMPLE_CHESTNUT_0S_IMG)
        frames = frame.split_xy(100)

        # print(frame.shape())
        # print(np.array(frames).shape)
        # print(np.array(frames[0]).shape)
        # print(frames[0][0].shape()) # (100, 100, 1)
        # print(frames[0][0].data) # np array of one 2D Frame where each value is (r,g,b)
        print(frames[0][0].data[0])
        # print(frames[0][0].data[0].shape) # (100, 1) where each value is (r,g,b)
        
        for xframes in frames:
            for frame in xframes:
                frame_red = frame.channel(CONSTS.CHANNEL.RED)
                glcm = frame_red.glcm(by=1, axis=CONSTS.AXIS.Y)
                glcm.contrast()
                glcm.correlation()
                glcm.entropy()

if __name__ == '__main__':
    unittest.main()
