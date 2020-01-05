# import torch
import torch
# import numpy
import numpy as np

# import model
from Pointnet.models import Model_SEG
# import utils
from utils.utils import normalize_pc, align_principle_component
from utils.data import build_data_seg, class2color
from utils.clustering import region_growing
from utils.Visualizer import Visualizer

# import others
import os
import json
from collections import OrderedDict


# *** SET UP ***

# path to example
example_fpath = "C:/Users/Nicla/Google Drive/P4B/Pointclouds/PinotNoir_3.feats"
# path to save folder
fpath = "C:/Users/Nicla/Google Drive/P4B/results/hierarchicalSegmentation_v1"


# *** LOAD CONFIG ***

# open and load json file
with open(os.path.join(fpath, "config.json"), "r") as f:
    config = json.loads(f.read())
# make sure model is trained on segmentation task
assert config['task'] == 'hierarchical_segmentation', "Model must be trained on hierarchical segmentation task"
# get classes predicted by model
hierarchical_classes = [OrderedDict(classes) for classes in config['data']['hierarchy_classes']]
K = len(hierarchical_classes[0])
classes = [key.split(':')[0] for bins in hierarchical_classes for key in bins if 'final' in key]
# get features and feature dimension
features = config['data']['features']
feat_dim = config['data']['feature_dim']


# *** LOAD MODEL ***

# create model
model = Model_SEG(K=K, feat_dim=feat_dim)
# load parameters
model.load_encoder(os.path.join(fpath, "encoder.model"))
model.load_segmentater(os.path.join(fpath, "segmentater.model"))
# evaluate
model.eval()


# *** PASS HIERARCHICAL THROUGH MODEL ***

# remove modifiers from keys, i.e. :final, etc.
make_class_bins = lambda class_bins: OrderedDict({key.split(':')[0]: value for key, value in class_bins.items()})

# load file
pc = np.loadtxt(example_fpath)
# select points of interest
class_ids_of_interest = [list(class2color.keys()).index(n) for bin in hierarchical_classes[0].values() for n in bin]
pc = pc[np.isin(pc[:, -1], class_ids_of_interest)]
# initalize prediction and target arrays
prediction = np.empty(pc.shape[0], dtype=np.int32)
target = np.empty(pc.shape[0], dtype=np.int32)

# lists of masks defining current subpointclouds
masks = [np.ones(pc.shape[0]).astype(np.bool)]
outlier_mask = np.zeros(pc.shape[0]).astype(np.bool)
# loop through all hierarchies
for i, class_bins in enumerate(hierarchical_classes):

    # get global label from label in current hierarchy
    get_label = lambda i: list(class2color.keys()).index(list(class_bins.keys())[int(i)].split(':')[0])

    # get all non-final class indices
    non_final_classes = [j for j, class_name in enumerate(class_bins) if 'final' not in class_name]
    # mask of non final points for each pointcloud
    non_final_masks = []

    with torch.no_grad():

        # run current hierarchy
        for mask in masks:
            # get subpointcloud from mask
            pc_ = pc[mask, :]
            if i > 0:
                # align pointcloud to face down
                pc_[:, :3] = align_principle_component(pc_[:, :3])
            # prepare pointcloud and save target
            x, y = build_data_seg({'-': [pc_]}, -1, 1, features=features)
            target[mask] = y.numpy().flatten()
            # predict classes
            log_probs = model.forward(x).cpu().numpy()
            cur_pred = np.argmax(log_probs, axis=-1).reshape(-1, 1)
            prediction[mask] = np.apply_along_axis(get_label, axis=-1, arr=cur_pred).flatten()
            # create mask and add to list
            mask[mask] = np.isin(cur_pred, non_final_classes).flatten()
            non_final_masks.append(mask)

    # prepare next hierarchy
    if i < len(hierarchical_classes) - 1:
        new_masks = []
        # build pointclouds for next hierarchy
        for non_final in non_final_masks:
            # get normals and curvature values of current pointcloud
            pc_ = pc[non_final, :]
            # and apply region growing on pointcloud
            normals, curvatures = pc_[:, 6:9], pc_[:, 9]
            cluster_mask = region_growing(normalize_pc(pc_[:, :3]), normals, curvatures, min_points=100)
            # build new masks from clusters
            for j in np.unique(cluster_mask):
                mask_ = non_final.copy()
                mask_[non_final] = (cluster_mask==j)
                if j != -1:
                    new_masks.append(mask_)
                else:
                    outlier_mask |= mask_
                    print(mask.sum(), outlier_mask.sum())

        # update masks
        masks = new_masks


# *** VISUALIZE ***

# get actual and predicted colors
get_color = lambda i: list(class2color.values())[int(i)]
actual_colors = np.apply_along_axis(get_color, axis=-1, arr=target.reshape(-1, 1))
pred_colors = np.apply_along_axis(get_color, axis=-1, arr=prediction.reshape(-1, 1))
# get points
points = pc[:, :3]

# color all outliers black
pred_colors[outlier_mask, :] = 255

# create visualizer
vis = Visualizer()
# visualize original pointcloud
vis.add_by_features(points, pc[:, 3:6] / 255, normalize=True)
vis.add_by_features(points, actual_colors / 255, normalize=True)
vis.add_by_features(points, pred_colors / 255, normalize=True)
vis.run()
