
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
    from data import color2class, build_data_seg
    # import segmentation model
    from models import Model_SEG
    # import torch
    import torch

    # *** SET UP ***
    example_fpath = "C:/Users/Niclas/Documents/Pointclouds/Skeleton/Processed/PinotNoir_1.xyzrgbc"
    encoder_fpath = "results/segmentate/encoder.model"
    segmentater_fpath = "results/segmentate/segmentater.model"

    # *** PARAMETERS ***

    n_points = 10240
    n_samples = 5

    # get pointcloud features
    pc_raw = np.loadtxt(example_fpath)
    points, colors, classes = normalize_pc(pc_raw[:, 0:3]), pc_raw[:, 3:6] / 255, pc_raw[:, -1:]
    # get ground truth colors
    class_colors = np.apply_along_axis(lambda i: color2class[int(i)], 1, classes) / 255

    # create and load model
    model = Model_SEG(K=7, feat_dim=4)
    model.load_encoder(encoder_fpath)
    model.load_segmentater(segmentater_fpath)
    model.eval()

    # predict classes of points
    x, _, _, _ = build_data_seg({'A': [pc_raw]}, n_points, n_samples, 0, features=['points', 'colors', 'length'])  
    log_probs = model.forward(x)
    predicted = torch.max(log_probs, dim=2)[1].view(-1, 1).cpu().numpy()
    # get processed points and predicted colors
    processed_points = x[:, :3, :].transpose(1, 2).view(-1, 3).cpu().numpy()
    predicted_colors = np.apply_along_axis(lambda i: color2class[int(i)], 1, predicted) / 255
    
    # create visualizer
    vis = Visualizer()
    # add to visualizer
    vis.add_by_features(points, colors, normalize=False)
    vis.add_by_features(points, class_colors, normalize=False)
    vis.add_by_features(processed_points, predicted_colors, normalize=False)
    # run
    vis.run()
