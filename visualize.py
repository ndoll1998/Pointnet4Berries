
# import open3d
import open3d
# import numpy
import numpy as np
# imprt os to check if file exists
import os

# import nomalize pointcloud
from utils import normalize_pc

class Visualizer:

    def __init__(self):
        # create visualizer and window
        self.vis = open3d.visualization.Visualizer()
        self.window = self.vis.create_window()
        # set rendering options
        self.vis.get_render_option().background_color = np.array([0, 0, 0])
        self.vis.get_render_option().point_size = 0.2
        # number of geometries
        self.n = 0

    def add_by_file(self, fpath, normalize=True):
        # check if file exists
        assert os.path.isfile(fpath), "File does not exist"
        # read file
        xyzrgb = np.loadtxt(fpath)
        points = xyzrgb[:, :3]
        colors = xyzrgb[:, 3:6] / 255
        # add pointcloud by features
        return self.add_by_features(points, colors, normalize=normalize)

    def add_by_pointcloud(self, pc, normalize=True):
        # get pointcloud features
        points, colors = np.asarray(pc.points), np.asarray(pc.colors)
        # add pointcloud by features
        return self.add_by_features(points, colors, normalize=normalize)

    def add_by_features(self, points, colors, normalize=True):
        # normalize
        points = normalize_pc(points) if normalize else points
        # create pointcloud and set points and colors
        pc = open3d.geometry.PointCloud()
        pc.points = open3d.utility.Vector3dVector(points + np.array([self.n * 1.5, 0, 0]))
        pc.colors = open3d.utility.Vector3dVector(colors)
        # add pointcloud
        self.vis.add_geometry(pc)
        self.n += 1
        # return pointcloud
        return pc

    def run(self):
        # run and destroy window afterwards
        self.vis.run()
        self.vis.destroy_window()


if __name__ == '__main__':
    # import colors of classes
    from data import color2class
    # import segmentation model
    from models import Model_SEG
    # import torch
    import torch

    # *** SET UP ***
    example_fpath = "C:/Users/doll0.SGN/Documents/Grapes/Skeletons_seg_normals/CalardisBlanc/1E.xyzrgbc"
    encoder_fpath = "I:/Skeletons_normals_equally_distributed/encoder.model"
    segmentater_fpath = "I:/Skeletons_normals_equally_distributed/segmentater.model"

    # load file to numpy array
    pc_raw = np.loadtxt(example_fpath)
    # separate array
    points = pc_raw[:, 0:3]
    colors = pc_raw[:, 3:6] / 255
    classes = pc_raw[:, -1:]
    # create color from classes
    class_colors = np.apply_along_axis(lambda i: color2class[int(i)], 1, classes) / 255
    # create original pointcloud
    pc_original = open3d.geometry.PointCloud()
    pc_original.points = open3d.utility.Vector3dVector(points)
    pc_original.colors = open3d.utility.Vector3dVector(colors)
    # create ground truth poitncloud
    pc_ground_truth = open3d.geometry.PointCloud()
    pc_ground_truth.points = open3d.utility.Vector3dVector(points)
    pc_ground_truth.colors = open3d.utility.Vector3dVector(class_colors)

    # create and load model
    model = Model_SEG(K=len(color2class), feat_dim=6)
    model.load_encoder(encoder_fpath)
    model.load_segmentater(segmentater_fpath)
    model.eval()
    # predict classes of points
    x = torch.from_numpy(pc_raw[:, :-1]).float().t().unsqueeze(0)
    log_probs = model.forward(x)
    predicted = torch.max(log_probs, dim=2)[1]
    predicted = predicted.cpu().numpy()
    # get predicted colors
    predicted_colors = np.apply_along_axis(lambda i: color2class[int(i)], 1, predicted.T) / 255
    # build pointcloud
    pc_predicted = open3d.geometry.PointCloud()
    pc_predicted.points = open3d.utility.Vector3dVector(points)
    pc_predicted.colors = open3d.utility.Vector3dVector(predicted_colors)

    # create visualizer
    vis = Visualizer()
    # add to visualizer
    vis.add_by_pointcloud(pc_original)
    vis.add_by_pointcloud(pc_ground_truth)
    vis.add_by_pointcloud(pc_predicted)
    # run
    vis.run()
