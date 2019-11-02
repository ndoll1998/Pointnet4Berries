# import numpy
import numpy as np
# import open3d to easily manipulate pointclouds
import open3d

def normalize_pc(pc, axis=1):

    # transform relative to centroid and resclae
    pc = pc - np.mean(pc, axis=0, keepdims=True)
    pc = pc / np.max(np.linalg.norm(pc, axis=axis, keepdims=True))
    # return 
    return pc

def voxel_down_sample(points, colors, voxel_size):
    # create pointcloud object
    pc = open3d.geometry.PointCloud()
    pc.points = open3d.utility.Vector3dVector(points)
    pc.colors = open3d.utility.Vector3dVector(colors)
    # downsample pointcloud
    pc_downsample = open3d.open3d.geometry.voxel_down_sample(pc, voxel_size=voxel_size)
    # create arrays from downsampled pointcoud
    points_downsamples = np.asarray(pc_downsample.points)
    colors_downsamples = np.asarray(pc_downsample.colors)
    # return
    return points_downsamples, colors_downsamples