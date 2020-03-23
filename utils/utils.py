# import pytorch framework
import torch
# import numpy
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
# import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

# import others
import os
import math
import itertools
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

def normalize_pc(points, reduce_axis=0, feature_axis=1):

    # check if there are any points in pointcloud
    if points.shape[reduce_axis] == 0:
        return points

    # transform relative to centroid and resclae
    points = points - np.mean(points, axis=reduce_axis, keepdims=True)
    points = points / np.max(np.linalg.norm(points, axis=feature_axis, keepdims=True))
    # return 
    return points

def interpolate_pc(points, normals, k=2, span=5e-2):
    """ generate k random vectors on each plane spanned by a 
        point and its normal in the pointcloud
    """
    # create random vector in plane spanned by normals
    r = np.random.uniform(-1, 1, size=(points.shape[0] * k, 3))
    n = np.cross(normals.repeat(k, axis=0), r)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    # create new points
    d = np.random.uniform(-span, span, size=n.shape[0])
    inter_points = points.repeat(k, axis=0) + n * d.reshape(-1, 1)
    # return points
    return inter_points

def estimate_curvature_and_normals(points, target_points=None, n_neighbors=750):
    target_points = points if target_points is None else target_points
    # create arrays to store curvatures and normals in
    curvatures = np.empty(target_points.shape[0])
    normals = np.empty_like(target_points)
    # get nearest neighbors
    tree = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', n_jobs=-1).fit(points)
    nearest_idx = tree.kneighbors(target_points, return_distance=False)
    # delete tree to free memory
    del tree

    # estimate curvature of each point
    for i, idx in enumerate(nearest_idx):
        # compute converiance matrix of nearest neighbors
        nearest_points = points[idx, :]
        covariance = np.cov(nearest_points, rowvar=False)
        # estimate eigenvalues of covariance matrix 
        values, vectors = np.linalg.eig(covariance)
        # approximate curvature and normal of current point
        curvatures[i] = values.min() / values.sum()
        normals[i, :] = vectors[np.argmin(values)]

    return curvatures, normals

def align_principle_component(points, b=(0, -1, 0)):
    # principle component anaylsis
    pca = PCA(n_components=1).fit(points)
    a = pca.components_[0:1, :].T
    a /= np.linalg.norm(a)
    # priciple component must show away from origin such that the aligned custer/pointcloud is not upside down
    mean = points.mean(axis=0).reshape(-1, 1)
    d = mean / np.linalg.norm(mean)
    # compute angle between principle component and position of pointcloud and inverse direction if needed
    a *= -1 if (a.T @ d < 0) else 1
    # create rotation matrix to align principle component to b
    c = a + np.asarray(b).reshape(-1, 1)
    R = 2 * (c @ c.T) / (c.T @ c) - np.eye(3)
    # rotate points
    return points @ R.T


""" Voxel Grid Helpers """

def get_points_in_bbox(points, anchorA, anchorB):
    anchorA, anchorB = np.asarray(anchorA), np.asarray(anchorB)
    # translate points to both anchors
    translateA = points - anchorA.reshape(1, -1)
    translateB = points - anchorB.reshape(1, -1)
    # check if points are positive linear combination
    # of the axis spanned by bbox
    maskA = (translateA @ np.diag(anchorB - anchorA)) >= 0
    maskB = (translateB @ np.diag(anchorA - anchorB)) >= 0
    # combined idx
    return np.where(np.logical_and.reduce(maskA, axis=-1) & np.logical_and.reduce(maskB, axis=-1))[0]

def group_points_by_grid(points, voxel_grid_size):
    # get boundings of points
    max_ = np.max(points, axis=0)
    min_ = np.min(points, axis=0)
    # move boundign box anchor to origin
    bbox = max_ - min_
    # compute number of voxels in each dimension
    n_voxels = np.ceil(bbox/voxel_grid_size).astype(np.int32)

    voxels = []
    # loop over all voxels
    for index in np.ndindex(*n_voxels):
        # build anchors of curren voxels
        anchorA = np.asarray(index) * voxel_grid_size + min_
        anchorB = anchorA + voxel_grid_size
        # get points in current voxel
        points_idx = get_points_in_bbox(points, anchorA, anchorB)
        if len(points_idx) > 0:
            voxels.append(points_idx)
    
    return voxels

def get_all_anchors(anchorA, anchorB):
    # get dimension
    dim = len(anchorA)
    # build anchors
    anchor_coord_idx = itertools.product([0, 1], repeat=dim)
    anchors = [tuple((anchorA[i], anchorB[i])[j] for i, j in enumerate(idx)) for idx in anchor_coord_idx]
    return anchors

def group_points_by_octree(points, min_points):

    
    def build_subtree_(points):
        # get bounding box
        bbox_min = np.min(points, axis=0)
        bbox_max = np.max(points, axis=0)
        # get center point - using mean instead of center of bbox
        center = tuple(np.mean(points, axis=0))
        # build all anchor-points
        anchors = get_all_anchors(bbox_min, bbox_max)
        # group points
        return [get_points_in_bbox(points, anchor, center) for anchor in anchors]

    k = 0
    # voxels of first hierarchy
    voxels = [np.array(range(points.shape[0]))]
    # loop over all voxels
    while k < len(voxels):
        voxel = voxels.pop(k)
        # check trivial case
        if len(voxel) > min_points:
            # build subtree
            voxels += [voxel[idx] for idx in build_subtree_(points[voxel])]
        else:
            # add voxel back in and go on with next voxel
            voxels.insert(k, voxel)
            k += 1

    return voxels

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