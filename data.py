# import numpy
import numpy as np
# import torch
import torch
# import utils
from utils import normalize_pc, voxel_down_sample

# import others
from tqdm import tqdm
from random import sample

""" Generate Training and Testing Data """

def build_data_cls(pointclouds_per_class, n_points, n_samples):
    
    x_train, y_train = [], []
    x_test, y_test = [], []
    for i, pcs in enumerate(pointclouds_per_class.values()):
        # pick a random subset as train-set for current type
        train_idx = sample(range(len(pcs)), len(pcs) - 5)
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

def build_data_seg(pointclouds_per_class, n_points, n_samples):

    train, test = [], []
    for i, pcs in enumerate(pointclouds_per_class.values()):
        # pick a random subset as train-set for current type
        train_idx = sample(range(len(pcs)), len(pcs) - 2)
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
                train.append(pcs[j][idx, :])
        # build test data
        for j in test_idx:
            # create multiple subclouds from one cloud
            for _ in range(n_samples):
                # check if pointcloud consists of enough points
                if pcs[j].shape[0] < n_points:
                    continue
                # get random subset of points
                idx = sample(range(pcs[j].shape[0]), n_points)
                test.append(pcs[j][idx, :])

    # convert lists to numpy
    train, test = np.array(train, dtype=np.float32), np.array(test, dtype=np.float32)
    # separate input and class-labels
    x_train, y_train = train[:, :, :6], train[:, :, 6:]
    x_test, y_test = test[:, :, :6], test[:, :, 6:]
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


""" Perpare Data for Segmentation """

# map color to class by index
color2class = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (200, 140, 0), (0, 200, 200), (255, 0, 255), (255, 255, 255)]

def create_segmentation_pointcloud(original_file, segmentation_file, save_file):
    # read files
    original = np.loadtxt(original_file).astype(np.float32)
    segmentation = np.loadtxt(segmentation_file).astype(np.float32)
    # get segmentations
    seg_sematic = segmentation[:, 3:]
    # map segmentation to class
    get_class = lambda c: color2class.index(tuple(c))
    classes = np.apply_along_axis(get_class, 1, seg_sematic)
    # stack segmentation to array and sace
    combined = np.hstack((original, classes.reshape(-1, 1)))
    np.savetxt(save_file, combined)


""" Data Augmentation """

def mirror_pointcloud(original_file, save_file):
    # read files and mirror x-axis
    mirrored = np.loadtxt(original_file).astype(np.float32)
    mirrored[:, 0] *= -1
    # save to file
    np.savetxt(save_file, mirrored)



if __name__ == '__main__':

    # set flags
    COMBINE_DATA = False
    AUGMENT_DATA = True

    """ combine semantic labels and actual colors in one class """

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
        ("C:/Users/doll0.SGN/Documents/Grapes/Skeletons/CalardisBlanc/1E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/CalardisBlanc_1E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/Skeletons_seg/CalardisBlanc/1E.xyzrgbc"),
        ("C:/Users/doll0.SGN/Documents/Grapes/Skeletons/CalardisBlanc/2E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/CalardisBlanc_2E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/Skeletons_seg/CalardisBlanc/2E.xyzrgbc"),
        ("C:/Users/doll0.SGN/Documents/Grapes/Skeletons/CalardisBlanc/3E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/CalardisBlanc_3E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/Skeletons_seg/CalardisBlanc/3E.xyzrgbc"),
        ("C:/Users/doll0.SGN/Documents/Grapes/Skeletons/CalardisBlanc/4E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/CalardisBlanc_4E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/Skeletons_seg/CalardisBlanc/4E.xyzrgbc"),
        # Dornfelder
        ("C:/Users/doll0.SGN/Documents/Grapes/Skeletons/Dornfelder/1E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/GroundTruth/Dornfelder_1E.xyzrgb", "C:/Users/doll0.SGN/Documents/Grapes/Skeletons_seg/Dornfelder/1E.xyzrgbc"),
    ]

    # combine ground truth and original files

    if COMBINE_DATA:

        print("\n\nCreating Data for Segmentation Task\n")
        for (orig, seg, save) in tqdm(grapes_files):
            # create data
            create_segmentation_pointcloud(orig, seg, save)

        # for (orig, seg, save) in tqdm()

    # augment data

    if AUGMENT_DATA:

        print("\n\nAugmenting Data\n")
        for (_, _, combined) in tqdm(grapes_files):
            # create mirrored
            mirror_pointcloud(combined, '.'.join(combined.split('.')[:-1]) + '_mirrored.xyzrgbc')
