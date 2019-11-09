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

def estimate_normals(points):
    # create pointcloud object
    pc = open3d.geometry.PointCloud()
    pc.points = open3d.utility.Vector3dVector(points[:, :3])
    # estimate normals and normalize afterwards
    # open3d.open3d.geometry.estimate_normals(pc)
    pc.estimate_normals()
    pc.normalize_normals()
    # return normals as numpy array
    return np.asarray(pc.normals)


""" Train Helpers """

def train_model(model, x_train, y_train, optim, epochs, batch_size, update_interval, save_interval, save_path, device='cpu'):
    # set model mode
    model.train()
    # train model
    losses = []
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
                print("Epoch: {0} -\tBatch: {1} -\tLoss: {2:.04f} -\tBatch Time: {3:.02f} -\tTotal Time: {4:.02f}".format(e, b, loss.item(), time() - batch_start_time, time() - start_time))
                # update model parameters
                optim.zero_grad()
                loss.backward()
                optim.step()
                # reset for next iteration
                batch_start_time = time()
                loss = 0
            # remove - free gpu memory
            del x_batch, y_batch
        
        # add loss to losses
        losses.append(running_loss)
        # save
        if e % save_interval == 0:
            # save model after enough number of epochs
            model.save(save_path)
            # create loss graph
            fig, ax = plt.subplots(1, 1)
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss")
            # plot losses and save figure
            ax.plot(losses)
            try:
                fig.savefig(os.path.join(save_path, "lossgraph.pdf"), format='pdf')
            except Exception as e:
                print("[EXCEPTION] during save of plot:", e)
            # close figure to free memory
            plt.close()

    # save final encoder and classifier
    model.save(save_path)
    # return losses
    return losses


""" Evaluation Helpers """

def create_confusion_matrix(model, x_test, y_test, K, batch_size, device='cpu'):
    # confusion matrix
    confusion = np.zeros((K, K))
    # set eval
    model.eval()
    # no gradients
    with torch.no_grad():
        test_size = x_test.shape[0]
        for b in range(math.floor(test_size/batch_size)):
            # get batch
            x_batch = x_test[b * batch_size : (b+1) * batch_size].to(device).float()
            y_batch = y_test[b * batch_size : (b+1) * batch_size].numpy().astype(int)
            # pass through model
            class_probs = model.forward(x_batch)
            # get predicted classes
            predicted = torch.max(class_probs.reshape(-1, K), dim=-1)[1].cpu().numpy()
            # evaluate
            for actual, pred in zip(y_batch.flatten(), predicted):
                confusion[actual, pred] += 1
            # remove
            del x_batch, y_batch

    # return confusion table
    return confusion
    
def visualize_confusion_table(confusion, classes, normalize=True, cmap=plt.cm.Blues):

    if normalize:
        # normalize if asked for - handle division by zero
        confusion = confusion.astype('float') / np.maximum(1, confusion.sum(axis=1)[:, np.newaxis])
    else:
        # convert to integers
        confusion = confusion.astype(int)
    # create and configure axes
    fig, ax = plt.subplots()
    ax.set(xticks=np.arange(confusion.shape[1]),
           yticks=np.arange(confusion.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')
    # create confusion matrix and colorbar
    im = ax.imshow(confusion, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = confusion.max() / 2.
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(j, i, format(confusion[i, j], fmt),
                    ha="center", va="center",
                    color="white" if confusion[i, j] > thresh else "black")
    # set tight layout and return axes
    fig.tight_layout()
    return fig, ax
