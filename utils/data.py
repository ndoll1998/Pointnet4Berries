# import numpy
import numpy as np
# import torch
import torch
# import NearestNeighbors-Algorithm from sklearn
from sklearn.neighbors import NearestNeighbors
# import utils
from .utils import normalize_pc, voxel_down_sample, estimate_normals

# import others
from random import sample


# *** GENERATE TRAINING / TESTING DATA ***

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

def combine_features(data, features=['points', 'colors', 'length', 'pass_through']):
    # separate data
    points, colors = data[:, :, 0:3], data[:, :, 3:6]
    # normalize
    points, colors = normalize_pc(points, 1, 2), colors / 255    
    # collect features
    feats = []
    # add point features
    if 'points' in features:
        feats.append(points)
    # add color features
    if 'colors' in features:
        feats.append(colors)
    # add length feature
    if 'length' in features:
        feats.append(np.linalg.norm(points, axis=2, keepdims=True))
    # add features given in data
    if 'pass_through' in features:
        feats.append(data[:, :, 6:])
    # stack features
    return np.concatenate(feats, axis=2)

def build_data_cls(pointclouds_per_class, n_points, n_samples, features=['points', 'color', 'length']):
    """ Create Data for classification task """

    # build data from given pointclouds
    x, y = build_data(pointclouds_per_class, n_points, n_samples)
    # convert lists to numpy
    x, y = np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)
    # create input-feature-vectors
    x = combine_features(x, features=features)
    # convert to tensors and copy to device
    x, y = torch.from_numpy(x), torch.from_numpy(y).long()
    # transpose to match shape (batch, features, points)
    x = x.transpose(1, 2)

    return x, y

def build_data_seg(pointclouds, n_points, n_samples, features=['points', 'color', 'length']):
    """ Create Data for segmentation task """

    # build data from given pointclouds
    x, _ = build_data(pointclouds, n_points, n_samples)
    # convert to numpy array
    x = np.array(x, dtype=np.float32)
    # separate input and class
    x, y, = x[..., :-1], x[..., -1:]
    # create input feature vector
    x = combine_features(x, features=features)
    # convert to tensors
    x, y, = torch.from_numpy(x), torch.from_numpy(y).long()
    # transpose to match shape (batch, features, points)
    x = x.transpose(1, 2)

    return x, y


# *** PREPARE DATA FOR SEGMENTATION ***

# map color to class by index
# twig - subtwig - rachis - peduncle - berry - hook
color2class = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (200, 140, 0), (0, 200, 200), (255, 0, 255)]

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

    # get segmentations
    if use_nearest_color:
        # sort out points with colors not matching to any class
        idx = [i for i, x in enumerate(segmentation) if tuple(x[3:6]) in color2class]
        seg_sematic = get_color_from_nearest(original[:, :3], segmentation[idx, :3], segmentation[idx, 3:6])
    else:
        # get color by row
        seg_sematic = segmentation[:, 3:]
    # map segmentation to class
    get_class = lambda c: color2class.index(tuple(c))
    classes = np.apply_along_axis(get_class, 1, seg_sematic)
    stack += (classes.reshape(-1, 1), )

    # stack segmentation to array and sace
    combined = np.hstack(stack)
    np.savetxt(save_file, combined)


# *** DATA AUGMENTATION ***

def mirror_pointcloud(original_file, save_file):
    # read files and mirror x-axis
    mirrored = np.loadtxt(original_file).astype(np.float32)
    mirrored[:, 0] *= -1
    # save to file
    np.savetxt(save_file, mirrored)




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
                save_file = os.path.join(path, "Processed", sub_dir + f"_{index}.xyzrgbc")
                
                # skip if there is no ground thruth for current pointcloud
                if not os.path.exists(gt_file):
                    continue
                
                # create pointcloud
                create_segmentation_pointcloud(orig_file, gt_file, save_file, use_nearest_color=use_nearest_color)

    def augment_data_in_path(path):
        # augment all files in path
        for fname in tqdm(os.listdir(path)):
            # create full path to files
            orig_file = os.path.join(path, fname)
            save_file = os.path.join(path, '.'.join(fname.split('.')[:-1]) + '_mirrored.' + fname.split('.')[-1])
            # create mirrored data
            mirror_pointcloud(orig_file, save_file)


    # base directories
    dir_bunchs = "C:/Users/Niclas/Documents/Pointclouds/Bunch"
    dir_skeletons = "C:/Users/Niclas/Documents/Pointclouds/Skeleton"

    print("CREATE DATA:\n")
    # create segmentation data
    create_segmentation_data_from_path(dir_bunchs, use_nearest_color=False)
    create_segmentation_data_from_path(dir_skeletons, use_nearest_color=True)

    print("AUGMENT DATA:\n")
    # augment data
    augment_data_in_path(os.path.join(dir_bunchs, 'Processed'))
    augment_data_in_path(os.path.join(dir_skeletons, 'Processed'))


