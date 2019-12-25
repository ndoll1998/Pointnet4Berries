# import open3d
import open3d
# import numpy
import numpy as np
# imprt os to check if file exists
import os

# import nomalize pointcloud
from .utils import normalize_pc

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
        self.add_geometry(pc)
        self.n += 1
        # return pointcloud
        return pc

    def add_geometry(self, geometry):
        # add geometry to visualizer
        self.vis.add_geometry(geometry)

    def run(self, show_coordinate_frame=False):

        # add coordinate frame to view
        if show_coordinate_frame:
            mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(
                size=1, origin=[-1.5, 0, 0])
            self.add_geometry(mesh_frame)

        # run and destroy window afterwards
        self.vis.run()
        self.vis.destroy_window()