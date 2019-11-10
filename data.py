# import numpy
import numpy as np
# import torch
import torch
# import NearestNeighbors-Algorithm from sklearn
from sklearn.neighbors import NearestNeighbors
# import utils
from utils import normalize_pc, voxel_down_sample, estimate_normals

# import others
from tqdm import tqdm
from random import sample


# *** GENERATE TRAINING / TESTING DATA ***

def build_data_cls(pointclouds_per_class, n_points, n_samples, samples_for_testing=5):
    
    x_train, y_train = [], []
    x_test, y_test = [], []
    for i, pcs in enumerate(pointclouds_per_class.values()):
        # pick a random subset as train-set for current type
        train_idx = sample(range(len(pcs)), len(pcs) - samples_for_testing)
        test_idx = set(range(len(pcs))).difference(train_idx)
        # build train data
        for j in train_idx:
            # create multiple subclouds from one cloud
            for _ in range(n_samples):
                # check if pointcloud consists of enough points
                if pcs[j].shape[0] < n_points:
                    continue
                # get random subset of points
                idx = sample(range(pcs[j].shape[0]), n_points)
                x_train.append(pcs[j][idx, :])
                y_train += [i]
        # build test data
        for j in test_idx:
            # create multiple subclouds from one cloud
            for _ in range(n_samples):
                # check if pointcloud consists of enough points
                if pcs[j].shape[0] < n_points:
                    continue
                # get random subset of points
                idx = sample(range(pcs[j].shape[0]), n_points)
                x_test.append(pcs[j][idx, :])
                y_test += [i]

    # convert lists to numpy
    x_train, x_test = np.array(x_train, dtype=np.float32), np.array(x_test, dtype=np.float32)
    y_train, y_test = np.array(y_train, dtype=np.long), np.array(y_test, dtype=np.long)
    # normalize pointclouds
    x_train[:, :, :3] = normalize_pc(x_train[:, :, :3])
    x_test[:, :, :3] = normalize_pc(x_test[:, :, :3])
    # normalize rgb values
    x_train[:, :, 3:] /= 255
    x_test[:, :, 3:] /= 255
    # convert to tensors and copy to device
    x_train, x_test = torch.from_numpy(x_train), torch.from_numpy(x_test)
    y_train, y_test = torch.from_numpy(y_train), torch.from_numpy(y_test)
    # transpose
    x_train, x_test = x_train.transpose(1, 2), x_test.transpose(1, 2)
    # return data
    return x_train, y_train, x_test, y_test


def get_subsamples(pc, n_points, n_samples):
    # create subsamples of points equally distributed over all classes
    samples = []
    # check if pointcloud consists of enough points
    if pc.shape[0] < n_points:
        return samples
    # separate points by class
    class_idx = [list(np.where(pc[:, -1] == i)[0]) for i in np.unique(pc[:, -1])]
    n_points_per_class = n_points // len(class_idx)
    # create multiple subclouds from one cloud
    for _ in range(n_samples):
        # get random subset of points in each class
        idx = sum([sample(idx_, min(n_points_per_class, len(idx_))) for idx_ in class_idx], [])
        # fill with more random points
        idx.extend(sample(range(pc.shape[0]), n_points - len(idx)))
        # get random subset of points
        samples.append(pc[idx, :])
    # return samples
    return samples

def build_data_seg(pointclouds_per_class, n_points, n_samples, samples_for_testing=5):

    train, test = [], []
    for i, pcs in enumerate(pointclouds_per_class.values()):
        # pick a random subset as train-set for current type
        train_idx = sample(range(len(pcs)), len(pcs) - samples_for_testing)
        test_idx = set(range(len(pcs))).difference(train_idx)
        # build train data
        for j in train_idx:
            train.extend(get_subsamples(pcs[j], n_points, n_samples))
        # build test data
        for j in test_idx:
            test.extend(get_subsamples(pcs[j], n_points, n_samples))

    # convert lists to numpy
    train, test = np.array(train, dtype=np.float32), np.array(test, dtype=np.float32)
    # separate input and class-labels
    x_train, y_train = train[:, :, :-1], train[:, :, -1:]
    x_test, y_test = test[:, :, :-1], test[:, :, -1:]
    # normalize pointclouds
    x_train[:, :, :3] = normalize_pc(x_train[:, :, :3])
    x_test[:, :, :3] = normalize_pc(x_test[:, :, :3])
    # normalize rgb values
    x_train[:, :, 3:6] /= 255
    x_test[:, :, 3:6] /= 255
    # convert to tensors and copy to device
    x_train, x_test = torch.from_numpy(x_train), torch.from_numpy(x_test)
    y_train, y_test = torch.from_numpy(y_train), torch.from_numpy(y_test)
    # transpose
    x_train, x_test = x_train.transpose(1, 2), x_test.transpose(1, 2)
    # return data
    return x_train, y_train, x_test, y_test


# *** PREPARE DATA FOR SEGMENTATION ***

# map color to class by index
color2class = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (200, 140, 0), (0, 200, 200), (255, 0, 255), (255, 255, 255)]

def get_color_from_nearest(query_points, points, colors):
    # build tree
    tree = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(points)
    # get nearest neighbor indices
    idx = tree.kneighbors(query_points, return_distance=False)
    # return colors from indices
    return colors[idx.flatten()]

def create_segmentation_pointcloud(original_file, segmentation_file, save_file, use_nearest_color=False, approximate_normals=False):
    # read files
    original = np.loadtxt(original_file).astype(np.float32)
    segmentation = np.loadtxt(segmentation_file).astype(np.float32)
    # create feature-stack
    stack = (original,)

    # approximate normals
    if approximate_normals:
        stack +=(estimate_normals(original), )

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


# *** GENERATE DATA ***



if __name__ == '__main__':

    # set flags
    COMBINE_DATA = True
    AUGMENT_DATA = True

    # map files
    grapes_files = [
        # Calardis Blanc
        ("C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89/CalardisBlanc/CalardisBlanc_Grape_1E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/CalardisBlanc_Grape_1E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89_seg/CalardisBlanc/CalardisBlanc_Grape_1E.xyzrgbc"),
        ("C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89/CalardisBlanc/CalardisBlanc_Grape_2E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/CalardisBlanc_Grape_2E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89_seg/CalardisBlanc/CalardisBlanc_Grape_2E.xyzrgbc"),
        ("C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89/CalardisBlanc/CalardisBlanc_Grape_3E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/CalardisBlanc_Grape_3E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89_seg/CalardisBlanc/CalardisBlanc_Grape_3E.xyzrgbc"),
        ("C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89/CalardisBlanc/CalardisBlanc_Grape_4E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/CalardisBlanc_Grape_4E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89_seg/CalardisBlanc/CalardisBlanc_Grape_4E.xyzrgbc"),
        # Dornfelder
        ("C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89/Dornfelder/Dornfelder_Grape_1E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/Dornfelder_Grape_1E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89_seg/Dornfelder/Dornfelder_Grape_1E.xyzrgbc"),
        ("C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89/Dornfelder/Dornfelder_Grape_2E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/Dornfelder_Grape_2E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89_seg/Dornfelder/Dornfelder_Grape_2E.xyzrgbc"),
        ("C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89/Dornfelder/Dornfelder_Grape_3E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/Dornfelder_Grape_3E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89_seg/Dornfelder/Dornfelder_Grape_3E.xyzrgbc"),
        ("C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89/Dornfelder/Dornfelder_Grape_4E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/Dornfelder_Grape_4E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89_seg/Dornfelder/Dornfelder_Grape_4E.xyzrgbc"),
        ("C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89/Dornfelder/Dornfelder_Grape_5E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/Dornfelder_Grape_5E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89_seg/Dornfelder/Dornfelder_Grape_5E.xyzrgbc"),
        # Pinot Noir
        ("C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89/PinotNoir/PinotNoir_Grape_1.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/PinotNoir_Grape_1.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89_seg/PinotNoir/PinotNoit_Grape_1.xyzrgbc"),
        ("C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89/PinotNoir/PinotNoir_Grape_2.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/PinotNoir_Grape_2.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89_seg/PinotNoir/PinotNoit_Grape_2.xyzrgbc"),
        ("C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89/PinotNoir/PinotNoir_Grape_3.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/PinotNoir_Grape_3.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89_seg/PinotNoir/PinotNoit_Grape_3.xyzrgbc"),
        ("C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89/PinotNoir/PinotNoir_Grape_4.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/PinotNoir_Grape_4.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89_seg/PinotNoir/PinotNoit_Grape_4.xyzrgbc"),
        ("C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89/PinotNoir/PinotNoir_Grape_5.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/PinotNoir_Grape_5.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89_seg/PinotNoir/PinotNoit_Grape_5.xyzrgbc"),
        # Riesling
        ("C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89/Riesling/Riesling_1.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/Riesling_1.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89_seg/Riesling/Riesling_Grape_1.xyzrgbc"),
        ("C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89/Riesling/Riesling_2.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/Riesling_2.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89_seg/Riesling/Riesling_Grape_2.xyzrgbc"),
        ("C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89/Riesling/Riesling_3.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/Riesling_3.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89_seg/Riesling/Riesling_Grape_3.xyzrgbc"),
        ("C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89/Riesling/Riesling_4.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/Riesling_4.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89_seg/Riesling/Riesling_Grape_4.xyzrgbc"),
        ("C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89/Riesling/Riesling_5.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/Riesling_5.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89_seg/Riesling/Riesling_Grape_5.xyzrgbc"),
    ]

    stem_files = [
        # Calardis Blanc
        ("C:/Users/doll0.SGN/Documents/Grapes/Skeletons/CalardisBlanc/1E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/CalardisBlanc_1E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/Skeletons_seg_normals/CalardisBlanc/1E.xyzrgbc"),
        ("C:/Users/doll0.SGN/Documents/Grapes/Skeletons/CalardisBlanc/2E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/CalardisBlanc_2E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/Skeletons_seg_normals/CalardisBlanc/2E.xyzrgbc"),
        ("C:/Users/doll0.SGN/Documents/Grapes/Skeletons/CalardisBlanc/3E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/CalardisBlanc_3E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/Skeletons_seg_normals/CalardisBlanc/3E.xyzrgbc"),
        ("C:/Users/doll0.SGN/Documents/Grapes/Skeletons/CalardisBlanc/4E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/CalardisBlanc_4E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/Skeletons_seg_normals/CalardisBlanc/4E.xyzrgbc"),
        # Dornfelder
        ("C:/Users/doll0.SGN/Documents/Grapes/Skeletons/Dornfelder/1D.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/Dornfelder_1E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/Skeletons_seg_normals/Dornfelder/1E.xyzrgbc"),
        ("C:/Users/doll0.SGN/Documents/Grapes/Skeletons/Dornfelder/2D.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/Dornfelder_2E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/Skeletons_seg_normals/Dornfelder/2E.xyzrgbc"),
        ("C:/Users/doll0.SGN/Documents/Grapes/Skeletons/Dornfelder/3D.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/Dornfelder_3E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/Skeletons_seg_normals/Dornfelder/3E.xyzrgbc"),
        ("C:/Users/doll0.SGN/Documents/Grapes/Skeletons/Dornfelder/4D.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/Dornfelder_4E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/Skeletons_seg_normals/Dornfelder/4E.xyzrgbc"),
        ("C:/Users/doll0.SGN/Documents/Grapes/Skeletons/Dornfelder/5D.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/Dornfelder_5E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/Skeletons_seg_normals/Dornfelder/5E.xyzrgbc"),
        # Pinot Noir
        ("C:/Users/doll0.SGN/Documents/Grapes/Skeletons/PinotNoir/1.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/PinotNoir_1.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/Skeletons_seg_normals/PinotNoir/1.xyzrgbc"),
        ("C:/Users/doll0.SGN/Documents/Grapes/Skeletons/PinotNoir/2.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/PinotNoir_2.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/Skeletons_seg_normals/PinotNoir/2.xyzrgbc"),
        ("C:/Users/doll0.SGN/Documents/Grapes/Skeletons/PinotNoir/3.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/PinotNoir_3.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/Skeletons_seg_normals/PinotNoir/3.xyzrgbc"),
        ("C:/Users/doll0.SGN/Documents/Grapes/Skeletons/PinotNoir/4.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/PinotNoir_4.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/Skeletons_seg_normals/PinotNoir/4.xyzrgbc"),
        ("C:/Users/doll0.SGN/Documents/Grapes/Skeletons/PinotNoir/5.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/PinotNoir_5.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/Skeletons_seg_normals/PinotNoir/5.xyzrgbc"),
    ]

    # *** COMBINE SEGMENTATION GROUND TRUTH AND COLORS ***

    if COMBINE_DATA:

        print("\n\nCreating Data for Segmentation Task\n")
        # for (orig, seg, save) in tqdm(grapes_files):
        #     # create data - copy colors
        #     create_segmentation_pointcloud(orig, seg, save)

        for (orig, seg, save) in tqdm(stem_files):
            # create data - match color by distance
            create_segmentation_pointcloud(orig, seg, save, use_nearest_color=True, approximate_normals=True)


    # *** AUGMENT DATA ***

    if AUGMENT_DATA:

        print("\n\nAugmenting Data\n")
        # for (_, _, combined) in tqdm(grapes_files):
        #     # create mirrored
        #     mirror_pointcloud(combined, '.'.join(combined.split('.')[:-1]) + '_mirrored.xyzrgbc')

        for (_, _, combined) in tqdm(stem_files):
            # create mirrored
            mirror_pointcloud(combined, '.'.join(combined.split('.')[:-1]) + '_mirrored.xyzrgbc')


    # *** GENERATE DATA ***


    exit()


    import open3d
    from visualize import Visualizer

    points = np.random.uniform(-1, 1, size=(20_000, 3))
    colors = np.random.uniform(0, 1, size=(20_000, 3))

    pc = open3d.geometry.PointCloud()
    pc.points = open3d.utility.Vector3dVector(points)
    pc.colors = open3d.utility.Vector3dVector(colors)

    vis = Visualizer()
    vis.add_by_pointcloud(pc)
    vis.run()
















