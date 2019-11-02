
# import open3d
import open3d
# import numpy
import numpy as np
# import sys to read args
import sys
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

    def add(self, fpath):
        # check if file exists
        assert os.path.isfile(fpath), "File does not exist"
        # read file
        xyzrgb = np.loadtxt(fpath)
        points = normalize_pc(xyzrgb[:, :3]) + np.array([self.n * 1.5, 0, 0])
        colors = xyzrgb[:, 3:] / 255
        # create pointcloud
        pc = open3d.geometry.PointCloud()
        pc.points = open3d.utility.Vector3dVector(points)
        pc.colors = open3d.utility.Vector3dVector(colors)
        # add pointcloud
        self.vis.add_geometry(pc)
        self.n += 1

    def run(self):
        # run and destroy window afterwards
        self.vis.run()
        self.vis.destroy_window()

if __name__ == '__main__':

    # create visualizer
    vis = Visualizer()
    # compare labeling ground trouth to original
    # vis.add("C:/Users/doll0/Documents/Grapes/Skeletons/CalardisBlanc/1E.xyzrgb")
    # vis.add("C:/Users/doll0/Documents/Grapes/GroundTruth/CalardisBlanc_1E.xyzrgb")

    vis.add("C:/Users/doll0/Documents/Grapes/BBCH87_89/PinotNoir/PinotNoir_Grape_2.xyzrgb")
    vis.add("C:/Users/doll0/Documents/Grapes/BBCH87_89/Dornfelder/Dornfelder_Grape_2E.xyzrgb")

    # run
    vis.run()
