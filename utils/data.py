# import numpy
import numpy as np
# import torch
import torch
# import NearestNeighbors-Algorithm from sklearn
from sklearn.neighbors import NearestNeighbors
# import utils
from .utils import normalize_pc, rotationMatrix, estimate_curvature

# import others
from random import sample
from collections import OrderedDict

# map color to class by index
class2color = OrderedDict({
    'twig':     (255, 0, 0), 
    'subtwig':  (0, 255, 0), 
    'rachis':   (0, 0, 255), 
    'peduncle': (200, 140, 0), 
    'berry':    (0, 200, 200), 
    'hook':     (255, 0, 255),
    "None":     (255, 255, 255)
})

# list of all features
cls_features = ['x', 'y', 'z', 'r', 'g', 'b', 'length-xy', 'length-xyz']
seg_features = ['x', 'y', 'z', 'r', 'g', 'b', 'length-xy', 'length-xyz', 'curvature']

# *** DATA GENERATION HELPERS ***

def apply_bins(x, bins):
    binned_x = x.copy()
    for i, bin in enumerate(bins.values()):
        # apply bin
        for n in bin:
            # get index of class and apply bin
            j = list(class2color.keys()).index(n)
            binned_x[x == j] = i
    # return 
    return binned_x

def get_subsamples(pc, n_points, n_samples):
    """ randomly select an equal amount of points from each class """
    # create subsamples of points equally distributed over all classes
    samples = []
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

def build_data(pointclouds_per_class, n_points, n_samples):

    x, y = [], []
    for i, pcs in enumerate(pointclouds_per_class.values()):
        # build data
        for pc in pcs:
            # create multiple subclouds from one cloud
            x += get_subsamples(pc, n_points, n_samples)
            y += [i] * n_samples

    return x, y

def get_feature(data, feature):
    
    # features directly from data
    if feature in ['x', 'y', 'z', 'r', 'g', 'b', 'curvature']:
        i = ['x', 'y', 'z', 'r', 'g', 'b', 'curvature'].index(feature)
        return data[:, :, i:i+1]

    # length feature
    if feature == 'length-xyz':
        return np.linalg.norm(data[..., :3], axis=2, keepdims=True)
    # add 2d-length feature
    if feature == 'length-xy':
        return np.linalg.norm(data[..., :2], axis=2, keepdims=True)
    
def combine_features(data, features):
    # collect features
    feats = [get_feature(data, f) for f in features]
    # stack features
    return np.concatenate(feats, axis=2)


# *** GENERATE TRAINING / TESTING DATA ***

def build_data_cls(pointclouds_per_class, n_points, n_samples, features=cls_features):
    """ Create Data for classification task """

    # make sure all requested features are available
    assert all([f in cls_features for f in features]), "Only features from cls_features are valid"

    # build data from given pointclouds
    x, y = build_data(pointclouds_per_class, n_points, n_samples)
    # convert lists to numpy
    x, y = np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)
    # normalize values
    x[..., :3] = normalize_pc(x[..., :3], 1, 2) # positions
    x[..., 3:6] /= 255                          # colors
    # create input-feature-vectors
    x = combine_features(x, features=features)
    # convert to tensors and copy to device
    x, y = torch.from_numpy(x), torch.from_numpy(y).long()
    # transpose to match shape (batch, features, points)
    x = x.transpose(1, 2)

    return x, y

def build_data_seg(pointclouds, n_points, n_samples, class_bins=None, features=['points', 'color', 'length']):
    """ Create Data for segmentation task """

    # make sure all requested features are available
    assert all([f in seg_features for f in features]), "Only features from seg_features are valid"

    if class_bins is None:
        # standard class-bins keeping all classes as they are 
        class_bins = {c: [c] for c in class2color.keys()}
    # remove points of classes not contained in any bin
    class_ids_of_interest = [list(class2color.keys()).index(n) for bin in class_bins.values() for n in bin]
    pointclouds = {name: [pc[np.isin(pc[:, -1],class_ids_of_interest)] for pc in pcs] for name, pcs in pointclouds.items()}

    # build data from given pointclouds
    x, _ = build_data(pointclouds, n_points, n_samples)
    # convert to numpy array
    x = np.array(x, dtype=np.float32)
    # separate input and class
    x, y, = x[..., :-1], x[..., -1:]
    # normalize values
    x[..., :3] = normalize_pc(x[..., :3], 1, 2) # positions
    x[..., 3:6] /= 255                          # colors
    x[..., -1] /= np.abs(x[..., -1]).max()      # curvatures
    # apply bins to y if bins are given
    y = apply_bins(y, class_bins)
    # create input feature vector
    x = combine_features(x, features=features)
    # convert to tensors
    x, y, = torch.from_numpy(x), torch.from_numpy(y).long()
    # transpose to match shape (batch, features, points)
    x = x.transpose(1, 2)

    return x, y


# *** PREPARE DATA FOR SEGMENTATION ***

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

    # compute curvature values
    curvature = estimate_curvature(original[:, :3])
    stack += (curvature.reshape(-1, 1), )

    # map segmentation to class
    get_class = lambda c: color2class.index(tuple(c))
    classes = np.apply_along_axis(get_class, 1, seg_sematic)
    stack += (classes.reshape(-1, 1), )

    # stack segmentation to array and sace
    combined = np.hstack(stack)
    np.savetxt(save_file, combined)


# *** DATA AUGMENTATION ***

def mirror_pointcloud(original_file, save_file):
    # read file and mirror x-axis
    mirrored = np.loadtxt(original_file).astype(np.float32)
    mirrored[:, 0] *= -1
    # save to file
    np.savetxt(save_file, mirrored)

def rotate_pointcloud(original_file, save_file):
    # create random rotation matrix
    rotMatrix = rotationMatrix(*np.random.uniform(0, np.pi, size=3))
    # read file and rotate pointcloud
    rotated = np.loadtxt(original_file).astype(np.float32)
    rotated[:, :3] = rotated[:, :3] @ rotMatrix.T
    # save to file
    np.savetxt(save_file, rotated)


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

    def augment_data_in_path(path):
        # mirror all files in path
        for fname in tqdm(os.listdir(path)):
            # create full path to files
            orig_file = os.path.join(path, fname)
            save_file = os.path.join(path, '.'.join(fname.split('.')[:-1]) + '_mirrored.' + fname.split('.')[-1])
            # create mirrored data
            mirror_pointcloud(orig_file, save_file)

        # rotate all files in path
        for fname in tqdm(os.listdir(path)):
            # create full path to files
            orig_file = os.path.join(path, fname)
            save_file = os.path.join(path, '.'.join(fname.split('.')[:-1]) + '_rotated.' + fname.split('.')[-1])
            # create rotated data
            rotate_pointcloud(orig_file, save_file)


    # base directories
    dir_bunchs = "H:/Pointclouds/Bunch"
    dir_skeletons = "H:/Pointclouds/Skeleton"

    print("CREATE DATA:\n")
    # create segmentation data
    # create_segmentation_data_from_path(dir_bunchs, use_nearest_color=False)
    create_segmentation_data_from_path(dir_skeletons, use_nearest_color=True)

    print("AUGMENT DATA:\n")
    # augment data
    # augment_data_in_path(os.path.join(dir_bunchs, 'Processed'))
    augment_data_in_path(os.path.join(dir_skeletons, 'Processed'))


