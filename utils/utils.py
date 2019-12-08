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


""" Train Helpers """

def train_model(model, x_train, y_train, optim, epochs, batch_size, update_interval, device='cpu', callback=lambda:None):
    # set model mode
    model.train()
    # train model
    start_time = time()
    train_size = x_train.shape[0]
    for e in range(1, 1+epochs):
        # shuffle data
        shuffle_idx = np.arange(train_size)
        np.random.shuffle(shuffle_idx)
        x_train, y_train = x_train[shuffle_idx], y_train[shuffle_idx]
        # reset
        loss = 0
        running_loss = 0
        batch_start_time = time()
        # batch-loop
        for b in range(train_size//batch_size):
            # get batch
            x_batch = x_train[b * batch_size : (b+1) * batch_size].to(device).float()
            y_batch = y_train[b * batch_size : (b+1) * batch_size].to(device).long()
            # pass through model
            class_log_probs = model.forward(x_batch)
            # compute loss
            loss += model.loss(class_log_probs, y_batch)
            running_loss += loss.item()
            # optimization
            if (b > 0 and b % update_interval == 0) or (b == train_size//batch_size - 1):
                # update model parameters
                optim.zero_grad()
                loss.backward()
                optim.step()
                # log
                print("Epoch: {0} -\tBatch: {1} -\tLoss: {2:.04f} -\tBatch Time: {3:.02f} -\tTotal Time: {4:.02f}".format(e, b, loss.item(), time() - batch_start_time, time() - start_time))
                # reset for next iteration
                batch_start_time = time()
                loss = 0
            # remove - free gpu memory
            del x_batch, y_batch
        
        # callback
        callback(0 if e == epochs else e, running_loss)


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