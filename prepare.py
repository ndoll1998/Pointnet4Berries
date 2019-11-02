# import numpy
import numpy as np
# import utils
from utils import normalize_pc, voxel_down_sample

# import others
from tqdm import tqdm

""" Perpare Data for segmentation """

def create_segmentation_pointcloud(original_file, segmentation_file, save_file):

    # read files
    original = np.loadtxt(original_file).astype(np.float32)
    segmentation = np.loadtxt(segmentation_file).astype(np.float32)
    # separate points from colors
    orig_points, orig_colors = original[:, :3], original[:, 3:]
    seg_points, seg_sematic = segmentation[:, :3], segmentation[:, 3:]
    # free memmory
    del original, segmentation
    # normalize points
    orig_points, seg_points = normalize_pc(orig_points), normalize_pc(seg_points)
    # downsample
    orig_points, orig_colors = voxel_down_sample(orig_points, orig_colors, voxel_size=4e-3)
    seg_points, seg_sematic = voxel_down_sample(seg_points, seg_sematic, voxel_size=4e-3)
    # compute squared distances from each point in 
    # original to each point in segmentation point cloud
    dist = -2 * (orig_points @ seg_points.T)
    dist += np.sum(orig_points ** 2, axis=-1)[:, None]
    dist += np.sum(seg_points ** 2, axis=-1)[None, :]
    # for each point in segmentation find the nearest point in original
    arg_min = np.argmin(dist, axis=0)
    # get colors of nearest neighbors
    seg_colors = orig_colors[arg_min, :]
    # stack all together and save to file
    all_together = np.hstack((seg_points, seg_colors, seg_sematic))
    np.savetxt(save_file, all_together)


if __name__ == '__main__':

    """ combine semantic labels and actual colors in one class """

    # map files
    files = [
        # Calardis Blanc
        ("C:/Users/doll0/Documents/Grapes/Skeletons/CalardisBlanc/1E.xyzrgb", "C:/Users/doll0/Documents/Grapes/GroundTruth/CalardisBlanc_1E.xyzrgb", "C:/Users/doll0/Documents/Grapes/Skeletons_Full/CalardisBlanc_1.xyzrgb"),
        ("C:/Users/doll0/Documents/Grapes/Skeletons/CalardisBlanc/2E.xyzrgb", "C:/Users/doll0/Documents/Grapes/GroundTruth/CalardisBlanc_2E.xyzrgb", "C:/Users/doll0/Documents/Grapes/Skeletons_Full/CalardisBlanc_2.xyzrgb"),
        ("C:/Users/doll0/Documents/Grapes/Skeletons/CalardisBlanc/3E.xyzrgb", "C:/Users/doll0/Documents/Grapes/GroundTruth/CalardisBlanc_3E.xyzrgb", "C:/Users/doll0/Documents/Grapes/Skeletons_Full/CalardisBlanc_3.xyzrgb"),
        ("C:/Users/doll0/Documents/Grapes/Skeletons/CalardisBlanc/4E.xyzrgb", "C:/Users/doll0/Documents/Grapes/GroundTruth/CalardisBlanc_4E.xyzrgb", "C:/Users/doll0/Documents/Grapes/Skeletons_Full/CalardisBlanc_4.xyzrgb"),
        # Dornfelder
        ("C:/Users/doll0/Documents/Grapes/Skeletons/Dornfelder/1D.xyzrgb", "C:/Users/doll0/Documents/Grapes/GroundTruth/Dornfelder_1E.xyzrgb", "C:/Users/doll0/Documents/Grapes/Skeletons_Full/Dornfelder1.xyzrgb"),
        ("C:/Users/doll0/Documents/Grapes/Skeletons/Dornfelder/2D.xyzrgb", "C:/Users/doll0/Documents/Grapes/GroundTruth/Dornfelder_2E.xyzrgb", "C:/Users/doll0/Documents/Grapes/Skeletons_Full/Dornfelder2.xyzrgb"),
        ("C:/Users/doll0/Documents/Grapes/Skeletons/Dornfelder/3D.xyzrgb", "C:/Users/doll0/Documents/Grapes/GroundTruth/Dornfelder_3E.xyzrgb", "C:/Users/doll0/Documents/Grapes/Skeletons_Full/Dornfelder3.xyzrgb"),
        ("C:/Users/doll0/Documents/Grapes/Skeletons/Dornfelder/4D.xyzrgb", "C:/Users/doll0/Documents/Grapes/GroundTruth/Dornfelder_4E.xyzrgb", "C:/Users/doll0/Documents/Grapes/Skeletons_Full/Dornfelder4.xyzrgb"),
        ("C:/Users/doll0/Documents/Grapes/Skeletons/Dornfelder/5D.xyzrgb", "C:/Users/doll0/Documents/Grapes/GroundTruth/Dornfelder_5E.xyzrgb", "C:/Users/doll0/Documents/Grapes/Skeletons_Full/Dornfelder5.xyzrgb"),
        # Pinot Noir
        ("C:/Users/doll0/Documents/Grapes/Skeletons/PinotNoir/1.xyzrgb", "C:/Users/doll0/Documents/Grapes/GroundTruth/PinotNoir_1.xyzrgb", "C:/Users/doll0/Documents/Grapes/Skeletons_Full/PinotNoir1.xyzrgb"),
        ("C:/Users/doll0/Documents/Grapes/Skeletons/PinotNoir/2.xyzrgb", "C:/Users/doll0/Documents/Grapes/GroundTruth/PinotNoir_2.xyzrgb", "C:/Users/doll0/Documents/Grapes/Skeletons_Full/PinotNoir2.xyzrgb"),
        ("C:/Users/doll0/Documents/Grapes/Skeletons/PinotNoir/3.xyzrgb", "C:/Users/doll0/Documents/Grapes/GroundTruth/PinotNoir_3.xyzrgb", "C:/Users/doll0/Documents/Grapes/Skeletons_Full/PinotNoir3.xyzrgb"),
        ("C:/Users/doll0/Documents/Grapes/Skeletons/PinotNoir/4.xyzrgb", "C:/Users/doll0/Documents/Grapes/GroundTruth/PinotNoir_4.xyzrgb", "C:/Users/doll0/Documents/Grapes/Skeletons_Full/PinotNoir4.xyzrgb"),
        ("C:/Users/doll0/Documents/Grapes/Skeletons/PinotNoir/5.xyzrgb", "C:/Users/doll0/Documents/Grapes/GroundTruth/PinotNoir_5.xyzrgb", "C:/Users/doll0/Documents/Grapes/Skeletons_Full/PinotNoir5.xyzrgb")
    ]

    for (orig, seg, save) in tqdm(files):
        # create data
        create_segmentation_pointcloud(orig, seg, save)
