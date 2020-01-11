# import numpy
import numpy as np
# import utils
from .utils import rotationMatrix

# *** Augmenter ***

class Augmenter(object):

    def __init__(self, augment_fn, feats, apply_count=1, **kwargs):
        # save function
        self.fn = augment_fn
        self.feats = feats
        # how many augmentations are created by this
        self.count = apply_count
        # parameters for augmentation function
        self.fn_kwargs = kwargs

    def apply(self, pc):
        return [self.fn(pc.copy(), feats=self.feats, **self.fn_kwargs) for _ in range(self.count)]

    def __call__(self, pc):
        return self.apply(pc)

    def dict(self):
        return {
            "Augmentation": self.fn.__name__,
            "Count":        self.count,
            "args":         self.fn_kwargs
        }

# *** Augmentation Functions ***

def augment_mirror_pointcloud(pc, feats):
    for f in ['x', 'nx']:
        if f in feats:
            # get index of feat and invert it
            i = feats.index(f)
            pc[:, i] *= -1
    # return augmented pc
    return pc

def augment_rotate_pointcloud(pc, feats, rot_axis=['x', 'y', 'z']):
    # assertion
    assert all([(f in feats) for f in rot_axis]), "All rotation axis must be features"
    # get random rotations for 3 dimensions
    alphas = [(np.random.uniform(0, np.pi) if f in rot_axis else 0) for f in 'xyz']
    # create rotation matrix
    R = rotationMatrix(*alphas)
    # find features to rotate and apply rotation matrix
    idx = [feats.index(f) for f in 'xyz']
    pc[:, idx] = pc[:, idx] @ R.T
    # find normal features to rotate
    if all([('n'+f) in feats for f in 'xyz']):
        idx = [feats.index('n'+f) for f in 'xyz']
        # apply rotation
        pc[:, idx] = pc[:, idx] @ R.T
    # return rotated pointcloud
    return pc