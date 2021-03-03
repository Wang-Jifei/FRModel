# Testing out the filtering algorithm where Candidates are successively pruned
import logging
from frmodel.base.kdclusterer.kanungo_utils import *
from frmodel.base.kdclusterer.kdnode import KDNode
from frmodel.base.kdclusterer.candidate import Candidate
from frmodel.base.D2.frame2D import Frame2D
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

log = logging.getLogger("TestFilter.test_filter")

fig, ax = plt.subplots(2, 7)

for e, img in enumerate(("simple.png",
            "simple2.png",
            "simple3.png",
            "simple4.png",
            "simple5.png",
            "simple6.png",
            "simple7.png")):


    f = Frame2D.from_image(os.path.join(sys.path[0], "testimgs", img))
    flat = f.normalize().data[..., 0].reshape([-1, 1])  # just the red channel for easier debugging
    
    root = construct_kdtree(flat)

    k = 2

    candidate_centers_set = set()
    mapping_dict = {}
    for i in range(1, k+1):
        new_cand = Candidate(initial_pos = np.random.random(size = flat.shape[-1]))
        print("Initial position of candidate: ", new_cand.candidate_pos)
        candidate_centers_set.add(new_cand)
        mapping_dict[new_cand] = i

    log.debug(f"flatdata: {flat}")
    root.filter(candidate_centers_set)
    log.debug(f"\n\nroot.candidate_centers: {root.candidate_centers}")
    log.debug(f"root.right.candidate_centers: {root.right.candidate_centers}")
    log.debug(f"root.assigned: {root.assigned}")
    log.debug(f"root.right.assigned: {root.right.assigned}")
    log.debug(f"root.left.assigned: {root.left.assigned}")
    log.debug(f"root.left.left.assigned: {root.left.left.assigned}")

    all_assigned = [mapping_dict[cand] for cand in [n.assigned for n in root.cell]]
    print(f"all assigned {all_assigned}")
    ax[0][e].imshow(255 - f.data[...,0])
    ax[0][e].axis('off')
    ax[1][e].imshow(np.asarray(all_assigned).reshape((f.width(), f.height())).transpose())
    ax[1][e].axis('off')

fig.savefig(os.path.join(sys.path[0],"result.png"))