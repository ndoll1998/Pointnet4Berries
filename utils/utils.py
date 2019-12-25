# import pytorch framework
import torch
# import numpy
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
# import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
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

def normalize_pc(points, reduce_axis=0, feature_axis=1):

    # check if there are any points in pointcloud
    if points.shape[reduce_axis] == 0:
        return points

    # transform relative to centroid and resclae
    points = points - np.mean(points, axis=reduce_axis, keepdims=True)
    points = points / np.max(np.linalg.norm(points, axis=feature_axis, keepdims=True))
    # return 
    return points

def estimate_principle_components(points, k=1):
    # create pca
    pca = PCA(n_components=k).fit(points)
    # return principle axis
    return pca.components_
    

def estimate_curvature_and_normals(points, n_neighbors=750):
    # create array to store curvatures in
    curvatures = np.zeros(points.shape[0])
    normals = np.zeros_like(points)
    # get nearest neighbors
    tree = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree', n_jobs=-1).fit(points)
    _, nearest_idx = tree.kneighbors(points)
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