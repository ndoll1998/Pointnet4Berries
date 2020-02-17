# import torch
import torch
# import numpy
import numpy as np

# import model
from Pointnet.models import Model_SEG
# import utils
from utils.data import build_data_seg, class2color
from utils.utils import normalize_pc, align_principle_component
from utils.clustering import region_growing
from utils.Visualizer import Visualizer

# import others
import os
import json
from collections import OrderedDict

# *** TASKS ***

class SegmentationVisualizer(Visualizer):

    def __init__(self, config, model_path):
        # initialize visualizer
        Visualizer.__init__(self)
        # save configurations
        self.class_bins = OrderedDict(config['data']['classes'])
        self.classes = list(self.class_bins.keys())
        # get features used by model
        self.features = config['data']['features']
        feature_dim = config['data']['feature_dim']

        # create model
        self.model = Model_SEG(K=len(self.classes), feat_dim=feature_dim)
        # load parameters
        self.model.load_encoder(os.path.join(model_path, 'encoder.model'))
        self.model.load_segmentater(os.path.join(model_path, 'segmentater.model'))
        # evaluate model
        self.model.eval()

    def add_by_file(self, fpath):
        # check if file exists
        assert os.path.isfile(fpath), "File does not exist"

        # load and prepare pointcloud
        pc = np.loadtxt(fpath)
        x, y = build_data_seg({'CB': [pc]}, -1, 1, class_bins=self.class_bins, features=self.features)

        with torch.no_grad():
            # predict classes
            log_probs = self.model.forward(x).cpu().numpy()
            prediction = np.argmax(log_probs, axis=-1).reshape(-1, 1)

        # get actual and predicted colors
        get_color = lambda i: class2color[self.classes[int(i)]]
        actual_colors = np.apply_along_axis(get_color, axis=-1, arr=y.reshape(-1, 1))
        pred_colors = np.apply_along_axis(get_color, axis=-1, arr=prediction)
        # get points
        points = x[0, :3, :].T.cpu().numpy()

        # add pointclouds to visualizer
        self.add_by_features(pc[:, :3], pc[:, 3:6], normalize=True)
        self.add_by_features(points, actual_colors / 255, normalize=False)
        self.add_by_features(points, pred_colors / 255, normalize=False)

        # increase space tp next pointclouds
        self.n += 1


class HierarchivalSegmentationVisualizer(Visualizer):

    def __init__(self, config, model_path):
        # initialize visualizer
        Visualizer.__init__(self)

        # data preparation
        self.align_pointclouds = config['preparation']['align_pointclouds']
        # get classes predicted by model
        self.hierarchical_classes = [OrderedDict(classes) for classes in config['data']['hierarchy_classes']]
        K = len(self.hierarchical_classes[0])
        # get features and feature dimension
        self.features = config['data']['features']
        feat_dim = config['data']['feature_dim']

        # create model
        self.model = Model_SEG(K=K, feat_dim=feat_dim)
        # load parameters
        self.model.load_encoder(os.path.join(model_path, "encoder.model"))
        self.model.load_segmentater(os.path.join(model_path, "segmentater.model"))
        # evaluate
        self.model.eval()

    def add_by_file(self, fpath, normalize=True):
        # check if file exists
        assert os.path.isfile(fpath), "File does not exist"

        # load and prepare pointcloud
        pc = np.loadtxt(fpath)
        # select points of interest
        class_ids_of_interest = [list(class2color.keys()).index(n) for bin in self.hierarchical_classes[0].values() for n in bin]
        pc = pc[np.isin(pc[:, -1], class_ids_of_interest)]
        # initalize prediction and target arrays
        prediction = np.empty(pc.shape[0], dtype=np.int32)
        target = np.empty(pc.shape[0], dtype=np.int32)

        # lists of masks defining current subpointclouds
        masks = [np.ones(pc.shape[0]).astype(np.bool)]
        outlier_mask = np.zeros(pc.shape[0]).astype(np.bool)
        # loop through all hierarchies
        for i, class_bins in enumerate(self.hierarchical_classes):

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
                    if (i > 0) and self.align_pointclouds:
                        # align pointcloud to face down
                        pc_[:, :3] = align_principle_component(pc_[:, :3])
                    # prepare pointcloud and save target
                    x, y = build_data_seg({'CB': [pc_]}, -1, 1, features=self.features)
                    target[mask] = y.numpy().flatten()
                    # predict classes
                    log_probs = self.model.forward(x).cpu().numpy()
                    cur_pred = np.argmax(log_probs, axis=-1).reshape(-1, 1)
                    prediction[mask] = np.apply_along_axis(get_label, axis=-1, arr=cur_pred).flatten()
                    # create mask and add to list
                    mask[mask] = np.isin(cur_pred, non_final_classes).flatten()
                    non_final_masks.append(mask)

            # prepare next hierarchy
            if i < len(self.hierarchical_classes) - 1:
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

                # update masks
                masks = new_masks
        # get actual and predicted colors
        get_color = lambda i: list(class2color.values())[int(i)]
        actual_colors = np.apply_along_axis(get_color, axis=-1, arr=target.reshape(-1, 1))
        pred_colors = np.apply_along_axis(get_color, axis=-1, arr=prediction.reshape(-1, 1))
        # get points
        points = pc[:, :3]
        # color all outliers black
        pred_colors[outlier_mask, :] = 255

        # visualize original pointcloud
        self.add_by_features(points, pc[:, 3:6] / 255, normalize=normalize)
        self.add_by_features(points, actual_colors / 255, normalize=normalize)
        self.add_by_features(points, pred_colors / 255, normalize=normalize)

        # increase space tp next pointclouds
        self.n += 1
    

# map task-names to visualizers
taskVisualizers = {
    "segmentation": SegmentationVisualizer,
    "hierarchical_segmentation": HierarchivalSegmentationVisualizer,
}

if __name__ == '__main__':

    # path to example
    example_fpath = "data/D_3D.feats"
    # path to save folder
    # result_fpath = "results/full/"
    result_fpath = "results/hierarchical/v1"

    # check if path is valid
    assert os.path.exists(result_fpath), "Path does not exist"
    # open and load json file
    with open(os.path.join(result_fpath, "config.json"), "r") as f:
        config = json.loads(f.read())
    # get task
    task = config['task']

    # get visualizer for given task
    vis = taskVisualizers[task](config, result_fpath)
    # add test pointcloud
    vis.add_by_file(example_fpath)
    # show
    vis.run()
