import sys
sys.path.insert(1, '/Users/phoebezhouhuixin/FRModel')
import kd
from frmodel.base.D2.frame2D import Frame2D
from frmodel.base.D2.channel2D import Channel2D
from rsc.samples.frames import chestnut_0
import scipy.stats as ss
import numpy as np
from sklearn.neighbors import KDTree

frame = chestnut_0(0)
# print(frame.data.shape, frame.data[0])
r_chan = frame.channel("R")
r_vals = (r_chan.data - r_chan.data.mean())/r_chan.data.std()
r_vals = np.expand_dims(r_vals, axis = 2)

# print(r_chan.data.shape)
# print(r_chan.data[0])
# print(r_vals.shape)
# print(r_vals[0])

g_chan = frame.channel("G")
g_vals = (g_chan.data - g_chan.data.mean())/g_chan.data.std()
g_vals = np.expand_dims(g_vals, axis = 2)


x_vals = np.array([[x for x in range(frame.width())],]*frame.height())
x_vals = (x_vals - x_vals.mean())/x_vals.std()
x_vals = np.expand_dims(x_vals, axis = 2)
# print(x_vals, x_vals.shape)

y_vals = np.array([[y for y in range(frame.height())],]*frame.width()).transpose()
y_vals = (y_vals - y_vals.mean())/y_vals.std()
y_vals = np.expand_dims(y_vals, axis = 2)

vals = np.concatenate((x_vals, y_vals, r_vals, g_vals), axis = 2)
vals = vals.reshape(frame.height()*frame.width(), -1)
# print(vals.shape, vals[0])

# tree = KDTree(vals, leaf_size = 100) 
# # cannot input shape of [imgheight, imgwidth, noOfIndices], i.e. we can only have
# # [[x y r g], [x y r g], ...] which is of shape [imgheight*imgwidth, noOfIndices]
# # i.e. an array of pixels

tree = kd.create(dimensions = 4, point_list = vals.tolist())
kd.visualize(tree)

