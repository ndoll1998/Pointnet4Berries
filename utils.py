# import numpy
import numpy as np

def normalize_pc(pc, axis=1):

    # transform relative to centroid and resclae
    pc = pc - np.mean(pc, axis=0, keepdims=True)
    pc = pc / np.max(np.linalg.norm(pc, axis=axis, keepdims=True))

    return pc