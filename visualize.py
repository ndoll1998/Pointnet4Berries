
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

    def add_by_file(self, fpath):
        # check if file exists
        assert os.path.isfile(fpath), "File does not exist"
        # read file
        xyzrgb = np.loadtxt(fpath)
        points = xyzrgb[:, :3]
        colors = xyzrgb[:, 3:6] / 255
        # create pointcloud
        pc = open3d.geometry.PointCloud()
        pc.points = open3d.utility.Vector3dVector(points)
        pc.colors = open3d.utility.Vector3dVector(colors)
        # add pointcloud
        self.add_by_pointcloud(pc)
        # return pointcoud
        return pc

    def add_by_pointcloud(self, pc):
        # create normalized pointcoud
        pc_add = open3d.geometry.PointCloud()
        pc_add.points = open3d.utility.Vector3dVector(normalize_pc(np.asarray(pc.points)))
        pc_add.colors = pc.colors
        # move pointcloud on x axis
        pc_add.points = open3d.utility.Vector3dVector(
            np.asarray(pc_add.points) + np.array([self.n * 1.5, 0, 0])
        )
        # add pointcloud
        self.vis.add_geometry(pc_add)
        self.n += 1
        # return pointcloud
        return pc

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
    # compare original to downsample
    # pc = vis.add_by_file("C:/Users/doll0/Documents/Grapes/Skeletons/CalardisBlanc/1E.xyzrgb")
    # pc_downsample = vis.add_by_pointcloud(open3d.open3d.geometry.voxel_down_sample(pc, voxel_size=1))
    # view ground thruth matched
    vis.add_by_file("C:/Users/doll0/Documents/Grapes/Skeletons_Full/CalardisBlanc_1.xyzrgb")
    # run
    vis.run()
