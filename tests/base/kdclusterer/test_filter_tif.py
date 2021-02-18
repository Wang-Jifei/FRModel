# See that the kd tree is constructed correctly
# And that the 4 candidates are at R,G,B,W respectively.
# Then reconstruct img from kd tree
import sys
from frmodel.base.kdclusterer.kanungo_utils import *
from frmodel.base.kdclusterer.kdnode import KDNode
from frmodel.base.kdclusterer.candidate import Candidate
from frmodel.base.D2.frame2D import Frame2D
import numpy as np
from sklearn.preprocessing import StandardScaler
from PIL import Image
import seaborn as sb
sb.set()
import pandas as pd
import gdal

def main():
    # TODO: permute features
    bands_str = ["rededge", "NIR"]
    bands = []

    for band_str in bands_str:
        band_ds: gdal.Dataset = gdal.Open(sys.path[0] + f"/testimgs/{band_str}.tif")
        band: gdal.Band = band_ds.GetRasterBand(1)
        bands.append(band.ReadAsArray())
    for band in bands:
        print(band.shape)
    bands_ar = np.moveaxis(np.stack(bands), 0, -1)
    print(f"bands_ar.shape: {bands_ar.shape}")
    ratio = bands_ar.shape[1] / bands_ar.shape[0] # width / height
    print(ratio)

    '''Filter out pixels that are all nans'''
    notnan_pos = np.nonzero(np.all(np.isnan(bands_ar), axis=2) == False)
    # Returns a tuple of arrays, one for each dimension of a, containing the indices of the non-zero elements in that dimension.
    # Thus, nonnan_pos is  he row_idx, col_idx of pixels that are not [nan nan nan] 
    bands_ar = bands_ar[notnan_pos[0], notnan_pos[1], :]
    print("Shape of notnan bands_ar: ", bands_ar.shape)
    
    '''
    # There STILL are nans (e.g. [nan, nan, 700]) in the pixels at this step, just that the pixel cannot be [R G B] = [nan nan nan]
    for pixel in bands_ar:
        if True in np.isnan(pixel):
            print(pixel)
            break
    # Thus, if we construct_kdtree() with the bands_ar and filter() the Candidates according to the kdtree now, we will get an error,
    # since the filtering algorithm relies on the calculation of distance between the cell's actual_cent and each Candidate's cand_pos, 
    # and we would have nan actual_cent.
    # We need a way to deal with nans. Should we replace with zero?
    '''
    np.nan_to_num(bands_ar, copy=False, nan=0.0)
    
    '''Apply standard normalization to each point's features'''
    scaler = StandardScaler()
    bands_ar = scaler.fit_transform(bands_ar)

    # TODO GLCM

    '''Concat the x y coordinates of each point'''
    concat_x = np.expand_dims(notnan_pos[1], axis = 1) # height, width is y,x rather than x, y
    concat_y = np.expand_dims(np.flip(notnan_pos[0]), axis = 1)
    bands_ar = np.concatenate([concat_x, concat_y, bands_ar], axis=1)
   
    '''Sample just a few points'''
    bands_ar = bands_ar[np.random.choice(np.arange(0, len(bands_ar)), 10000)]  # 10000 points
    # TODO: use more points, but without printing, can log instead
    print(bands_ar.shape)
    print()
    print(bands_ar[0])


    '''Initialize centroids'''
    k = 4 
    candidate_centers_set = set()
    mapping_dict = {} # so that we know which cluster each  cluster id ("1","2","3","4") corresponds to
    for i in range(1, k+1):
        '''MIN-MAX CENTROID INITIALIZATION'''
        # rmin, rmax = get_minmax(root) # root min, root max
        # new_cand = Candidate(initial_pos = np.random.uniform(low= rmin, high = rmax, size = bands_ar.shape[-1]-2)) # -2 because we do not count x and y 

        '''FORGY INITIALIZATION'''
        new_cand_pos = bands_ar[np.random.choice(np.arange(0, len(bands_ar)), 1)].flatten()[2:] # idx 0 and 1 are x and y
        new_cand = Candidate(initial_pos=new_cand_pos)

        # TODO: centroid initialization
    
        print("Initial position of candidate: ", new_cand.candidate_pos)
        candidate_centers_set.add(new_cand)
        mapping_dict[new_cand] = i
    
    # Make kd tree
    root = construct_kdtree(bands_ar, axis=2, x_pos=0, y_pos=1)
    
    # Do the filtering algorithm
    root.filter(candidate_centers_set)

    # Plot the results
    # now, each [xyrgb] point has self.assigned to a certain Candidate
    plot_df = pd.DataFrame(columns=["x", "y", "c"])
    for i,node in enumerate(root.cell):
        x = node.x
        y = node.y
        cand = mapping_dict[node.assigned] # candidate id (int)
        plot_df.loc[i] = [x, y, cand]
    print(plot_df["x"].max(), plot_df["x"].min(), plot_df["y"].max(), plot_df["y"].min())
    f, ax = plt.subplots()
    sb.lmplot(x="x", y="y", hue="c", data=plot_df, fit_reg=False, scatter_kws={"s": 3.5}, aspect =ratio).savefig(sys.path[0] + "/forgy4.png")
    ax.set_aspect("equal") # TODO: why is the shape of the area not the same as original?
    
if __name__ == "__main__":
    main()