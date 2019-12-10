# import pytorch framework
import torch
# import numpy
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
# import open3d to easily manipulate pointclouds
import open3d

# import others
import os
import math
from time import time

""" Helpers """

def rotationMatrix(alpha, betha, gamma) :
    
    # create roation matrix for ratation around x-axis
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(alpha), -math.sin(alpha)],
                    [0, math.sin(alpha),  math.cos(alpha)]])

    # create rotation matrix for rotation around y-axis         
    R_y = np.array([[math.cos(betha), 0, math.sin(betha)],
                    [0, 1, 0],
                    [-math.sin(betha), 0, math.cos(betha)]])
                 
    # create rotation matrix for rotation around z-axis         
    R_z = np.array([[math.cos(gamma), -math.sin(gamma), 0],
                    [math.sin(gamma),  math.cos(gamma), 0],
                    [0, 0, 1]])
    
    # combined rotation matrix
    return R_z @ R_y @ R_x
 

""" Pointcloud Helpers """

def normalize_pc(pc, reduce_axis=0, feature_axis=1):

    # check if there are any points in pointcloud
    if pc.shape[reduce_axis] == 0:
        return pc

    # transform relative to centroid and resclae
    pc = pc - np.mean(pc, axis=reduce_axis, keepdims=True)
    pc = pc / np.max(np.linalg.norm(pc, axis=feature_axis, keepdims=True))
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

def estimate_normals(points):
    # create pointcloud object
    pc = open3d.geometry.PointCloud()
    pc.points = open3d.utility.Vector3dVector(points[:, :3])
    # estimate normals and normalize afterwards
    open3d.open3d.geometry.estimate_normals(pc)
    # pc.estimate_normals()
    pc.normalize_normals()
    # return normals as numpy array
    return np.asarray(pc.normals)


""" Evaluation Helpers """

def compute_fscores(confusion_matrix):

    # count true positives, false positives and false negatives
    tp = np.diag(confusion_matrix)
    fp = confusion_matrix.sum(axis=0) - tp
    fn = confusion_matrix.sum(axis=1) - tp
    
    # compute precision and recall
    p, r = tp / (tp + fp + 1e-10), tp / (tp + fn + 1e-10)
    # compute fscores
    scores = 2 * p * r / (p + r + 1e-10)

    return scores