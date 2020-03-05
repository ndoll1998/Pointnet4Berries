# import numpy
import numpy as np
# import torch
import torch
# import NearestNeighbors-Algorithm from sklearn
from sklearn.neighbors import NearestNeighbors
# import utils
if __name__ == '__main__':
    from utils import normalize_pc, rotationMatrix, estimate_curvature_and_normals, group_points_by_grid, group_points_by_octree
else:
    from .utils import normalize_pc, rotationMatrix, estimate_curvature_and_normals, group_points_by_grid, group_points_by_octree

# import others
from tqdm import tqdm
from random import sample
from collections import OrderedDict

# segmentation color-map
class2color = OrderedDict({
    'twig':     (255, 0, 0), 
    'subtwig':  (0, 255, 0), 
    'rachis':   (0, 0, 255), 
    'hook':     (200, 140, 0),
    'berry':    (0, 200, 200), 
    'peduncle': (255, 0, 255), 
    "None":     (255, 255, 255)
})

# list of features in files
cls_file_features = ['x', 'y', 'z', 'r', 'g', 'b']
seg_file_features = ['x', 'y', 'z', 'r', 'g', 'b', 'nx', 'nx', 'nx', 'curvature']
# list of all features
cls_features = cls_file_features + ['length-xy', 'length-xyz']
seg_features = seg_file_features + ['length-xy', 'length-xyz']


# *** DATA GENERATION HELPERS ***

def apply_bins(x, bins, task_classes):
    binned_x = x.copy()
    for i, bin in enumerate(bins.values()):
        # apply bin
        for n in bin:
            # get index of class and apply bin
            j = task_classes.index(n)
            binned_x[x == j] = i
    # return 
    return binned_x

def apply_augmentations(pointclouds, augmentations):
    augmented_pointclouds = []
    # apply all augmentations on all pointclouds
    for augment in augmentations:
        augmented_pointclouds.extend(
            sum([augment.apply(pc) for pc in pointclouds], [])
        )
    # return list of all augmented pointclouds
    return augmented_pointclouds

def get_subsamples(pc, n_points, n_samples):
    """ randomly select an equal amount of points from each class """
    # create subsamples of points equally distributed over all classes
    samples = []
    # check if asked for full pointcloud
    if n_points == -1:
        return [pc]
    # check if pointcloud consists of enough points
    if pc.shape[0] < n_points:
        return samples
    # separate points by class
    class_idx = [list(np.where(pc[:, -1] == i)[0]) for i in np.unique(pc[:, -1])]
    n_points_per_class = n_points // len(class_idx)
    # create multiple subclouds from one cloud
    for _ in range(min(n_samples, (pc.shape[0] // n_points)**2)):
        # get random subset of points in each class
        idx = sum([sample(idx_, min(n_points_per_class, len(idx_))) for idx_ in class_idx], [])
        # fill with more random points
        idx.extend(sample(range(pc.shape[0]), n_points - len(idx)))
        # get random subset of points
        samples.append(pc[idx, :])
    # return samples
    return samples

def get_voxel_subsamples(pc, n_points, n_samples):
    """ select random points from each voxel """
    # check if asked for full pointcloud
    if n_points == -1:
        return [pc]
    # check if pointcloud consists of enough points
    if pc.shape[0] < n_points:
        return []
    # get points
    points, n = normalize_pc(pc[:, :3]), pc.shape[0]
    # build voxels
    # voxel_points = group_points_by_grid(points, voxel_grid_size=0.2)
    voxel_points = group_points_by_octree(points, min_points=1_000)
    # get the number of points selected from each voxel
    weights = np.asarray([len(voxel)/n for voxel in voxel_points])
    points_per_voxel = np.ceil(weights * n_points).astype(np.int32)
    # create samples
    sample_idx = []
    for _ in range(n_samples):
        # select random points from each voxel and add to list
        idx = sum([sample(list(v), n) for v, n in zip(voxel_points, points_per_voxel)], [])
        sample_idx.append(idx)

    # select random points from and build samples from indices
    sample_idx = [sample(idx, n_points) for idx in sample_idx]
    samples = [pc[idx, :] for idx in sample_idx]

    # return samples
    return samples

def build_data(pointclouds_per_class, n_points, n_samples, classes=None):

    if classes is not None:
        # sort classes such that target y matches the order of classes
        grouped_pointclouds = [(pointclouds_per_class[c] if c in pointclouds_per_class else []) for c in classes]
    else:
        # group all pointclouds in a single class - usually applies for segmenation task
        classes = ['pc']
        grouped_pointclouds = [sum(pointclouds_per_class.values(), [])]

    # build data
    x, y = [], []
    for i, pcs in enumerate(grouped_pointclouds):
        # build data
        for pc in tqdm(pcs, desc=str(classes[i])):
            # create multiple subclouds from one cloud
            x += get_voxel_subsamples(pc, n_points, n_samples)
            y += [i] * n_samples

    return x, y

def get_feature(data, feature):
    
    # features directly from data
    if feature in ['x', 'y', 'z', 'r', 'g', 'b', 'nx', 'nx', 'nx', 'curvature']:
        i = ['x', 'y', 'z', 'r', 'g', 'b', 'nx', 'nx', 'nx', 'curvature'].index(feature)
        return data[:, :, i:i+1]

    # length feature
    if feature == 'length-xyz':
        return np.linalg.norm(data[..., :3], axis=2, keepdims=True)
    # add 2d-length feature
    if feature == 'length-xy':
        return np.linalg.norm(data[..., (0, 2)], axis=2, keepdims=True)
    
def combine_features(data, features):
    # collect features
    feats = [get_feature(data, f) for f in features]
    # stack features
    return np.concatenate(feats, axis=2)


# *** GENERATE TRAINING / TESTING DATA ***

def build_data_cls(pointclouds_per_class, n_points, n_samples, class_bins=None, features=cls_features, augmentations=None):
    """ Create Data for classification task """

    # make sure all requested features are available
    assert all([f in cls_features for f in features]), "Only features from cls_features are valid"
    # get all classes
    classes = list(pointclouds_per_class.keys())
    if class_bins is None:
        # standard class bins
        class_bins = OrderedDict({c: [c] for c in classes})

    # apply augmentations
    if augmentations is not None:
        pointclouds_per_class = {key: pcs + apply_augmentations(pcs, augmentations) for key, pcs in pointclouds_per_class.items()}
    # build data from given pointclouds
    x, y = build_data(pointclouds_per_class, n_points, n_samples, classes=classes)
    # convert lists to numpy
    x, y = np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)
    # normalize values
    x[..., :3] = normalize_pc(x[..., :3], 1, 2) # positions
    x[..., 3:6] /= 255                          # colors
    # apply bins to y
    y = apply_bins(y, class_bins, list(classes))
    # create input-feature-vectors
    x = combine_features(x, features=features)
    # convert to tensors and copy to device
    x, y = torch.from_numpy(x), torch.from_numpy(y).long()
    # transpose to match shape (batch, features, points)
    x = x.transpose(1, 2)

    return x, y

def build_data_seg(pointclouds, n_points, n_samples, class_bins=None, features=seg_features, augmentations=None):
    """ Create Data for segmentation task """

    # make sure all requested features are available
    assert all([f in seg_features for f in features]), "Only features from seg_features are valid"
    if class_bins is None:
        # standard class-bins keeping all classes as they are 
        class_bins = OrderedDict({c: [c] for c in class2color.keys()})
    # remove points of classes not contained in any bin
    class_ids_of_interest = [list(class2color.keys()).index(n) for bin in class_bins.values() for n in bin]
    pointclouds = {name: [pc[np.isin(pc[:, -1], class_ids_of_interest)] for pc in pcs] for name, pcs in pointclouds.items()}
    # apply augmentations
    if augmentations is not None:
        pointclouds = {key: pcs + apply_augmentations(pcs, augmentations) for key, pcs in pointclouds.items()}

    # build data from given pointclouds
    x, _ = build_data(pointclouds, n_points, n_samples)
    # convert to numpy array
    x = np.array(x, dtype=np.float32)
    # separate input and class
    x, y, = x[..., :-1], x[..., -1:]
    # normalize values
    x[..., :3] = normalize_pc(x[..., :3], 1, 2)     # positions
    x[..., 3:6] /= 255                              # colors
    x[..., -1] /= np.abs(x[..., -1]).max()          # curvatures
    # apply bins to y
    y = apply_bins(y, class_bins, list(class2color.keys()))
    # create input feature vector
    x = combine_features(x, features=features)
    # convert to tensors
    x, y, = torch.from_numpy(x), torch.from_numpy(y).long()
    # transpose to match shape (batch, features, points)
    x = x.transpose(1, 2)

    return x, y


# *** BUILD TRAINING FILES ***

def get_color_from_nearest(query_points, points, colors):
    # build tree
    tree = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(points)
    # get nearest neighbor indices
    idx = tree.kneighbors(query_points, return_distance=False)
    # return colors from indices
    return colors[idx.flatten()]

def create_segmentation_pointcloud(original_file, segmentation_file, save_file, use_nearest_color=False):
    # read files
    original = np.loadtxt(original_file).astype(np.float32)
    segmentation = np.loadtxt(segmentation_file).astype(np.float32)
    # create feature-stack
    stack = (original,)

    # map colors to classes by index
    color2class = list(class2color.values())
    # get segmentations
    if use_nearest_color:
        # sort out points with colors not matching to any class
        idx = [i for i, x in enumerate(segmentation) if tuple(x[3:6]) in color2class]
        seg_sematic = get_color_from_nearest(original[:, :3], segmentation[idx, :3], segmentation[idx, 3:6])
    else:
        # get color by row
        seg_sematic = segmentation[:, 3:]

    # estimate curvature and normals
    curvature, normals = estimate_curvature_and_normals(original[:, :3])
    stack += (normals, curvature.reshape(-1, 1), )

    # map segmentation to class
    get_class = lambda c: color2class.index(tuple(c))
    classes_ = np.apply_along_axis(get_class, 1, seg_sematic)
    stack += (classes_.reshape(-1, 1), )

    # stack segmentation to array and sace
    combined = np.hstack(stack)
    np.savetxt(save_file, combined)


# *** SCRIPT ***

if __name__ == '__main__':

    # import os to work with directories
    import os
    # import tqdm
    from tqdm import tqdm

    # *** Helpers ***

    def create_segmentation_data_from_path(path, use_nearest_color):
        # create skeleton-data
        for sub_dir in os.listdir(path):

            # log
            print(sub_dir)

            full_path = os.path.join(path, sub_dir)
            # get all original pointcloud-files
            files = [fname for fname in os.listdir(full_path) if 'OR' in fname]

            # process files
            for orig in tqdm(files):
                # create full path to original file
                orig_file = os.path.join(full_path, orig)
                # create path to ground truth
                gt_file = os.path.join(full_path, orig.replace('OR', 'GT'))
                # get index name of current pointcloud
                index = orig.split('_')[1].split('.')[0]
                # build path to save-file
                save_file = os.path.join(path, "Processed", sub_dir + f"_{index}.feats")
                
                # skip if there is no ground thruth for current pointcloud
                if not os.path.exists(gt_file):
                    continue
                
                # create pointcloud
                create_segmentation_pointcloud(orig_file, gt_file, save_file, use_nearest_color=use_nearest_color)

    # base directories
    dir_bunchs = "I:/Pointclouds/Bunch"
    dir_skeletons = "I:/Pointclouds/Skeleton"

    print("CREATE DATA:\n")
    # create segmentation data
    # create_segmentation_data_from_path(dir_bunchs, use_nearest_color=False)
    create_segmentation_data_from_path(dir_skeletons, use_nearest_color=True)

