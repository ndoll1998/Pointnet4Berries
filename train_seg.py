
# import numpy
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
# import torch
import torch
import torch.nn as nn
import torch.optim as optimizer

# import Pointnet
from Pointnet.PointnetPP import PointnetPP_Encoder, PointnetPP_Segmentation
# import utils
from utils import normalize_pc

# other imports
import os
import math
from tqdm import tqdm
from time import time
from random import sample, shuffle

# numpy percision for printing
np.set_printoptions(precision=3)

# map classes to colors by index
class_colors = [
    (1, 0, 0),
    (0, 1, 0),
    (1, 1, 0),
    (0, 0, 1),
    (1, 0, 1),
    (0, 1, 1),
    (1, 1, 1)
]


# *** PARAMS ***

# cuda device
device = 'cuda:0'
# model name
model_name = "BBCH87_89_seg"
# number of classes
K = len(class_colors)
# path to files
fpath = "C:/Users/doll0/Documents/Grapes/Skeletons_Full/"
# number of points to use in training
n_points = 12_000
# number of samples per pointcloud
n_samples = 4
# save path
save_fpath = "C:/Users/doll0/Documents/results/"


# training parameters
epochs = 50
batch_size = 2
# update parameters after n batches
update_interval = 1
# save model after every n-th epoch
save_interval = 2


# *** READ DATA ***

pointclouds = []
# open files
for directory in tqdm(os.listdir(fpath)):
    # create entry
    full_dir = os.path.join(fpath, directory)
    pointclouds.append(np.loadtxt(full_dir, dtype=np.float32))

# *** GENERATE TRAINING AND TESTING DATA ***

x_train, x_test = [], []

# pick a random subset as train-set for current type
train_idx = sample(range(len(pointclouds)), len(pointclouds) - 5)
test_idx = set(range(len(pointclouds))).difference(train_idx)
# build train data
for j in train_idx:
    # create multiple subclouds from one cloud
    for _ in range(n_samples):
        # check if pointcloud consists of enough points
        if pointclouds[j].shape[0] < n_points:
            continue
        # get random subset of points
        idx = sample(range(pointclouds[j].shape[0]), n_points)
        x_train.append(pointclouds[j][idx, :])
# build test data
for j in test_idx:
    # create multiple subclouds from one cloud
    for _ in range(n_samples):
        # check if pointcloud consists of enough points
        if pointclouds[j].shape[0] < n_points:
            continue
        # get random subset of points
        idx = sample(range(pointclouds[j].shape[0]), n_points)
        x_test.append(pointclouds[j][idx, :])

# convert lists to numpy
x_train, x_test = np.array(x_train, dtype=np.float32), np.array(x_test, dtype=np.float32)
# normalize pointclouds
x_train[:, :, :3] = normalize_pc(x_train[:, :, :3])
x_test[:, :, :3] = normalize_pc(x_test[:, :, :3])
# normalize rgb values
x_train[:, :, 3:6] /= 255
x_test[:, :, 3:6] /= 255
# separate classes from input
x_train, y_train = x_train[:, :, :6], (x_train[:, :, 6:] != 0).astype(int)
x_test, y_test = x_test[:, :, :6], (x_test[:, :, 6:] != 0).astype(int)
# get class from bit-representation
get_class = lambda bits: class_colors.index(tuple(bits))
y_train = np.apply_along_axis(get_class, -1, y_train)
y_test = np.apply_along_axis(get_class, -1, y_test)
# create pytorch tensors from numpy arrays
x_train, x_test = torch.from_numpy(x_train), torch.from_numpy(x_test)
y_train, y_test = torch.from_numpy(y_train), torch.from_numpy(y_test)
# transpose to match expected shape (batch, feats, points)
x_train, x_test = x_train.transpose(1, 2), x_test.transpose(1, 2)

# *** CREATE MODEL, OPTIMIZER AND LOSS ***

# create encoder and classifier
encoder = PointnetPP_Encoder(pos_dim=3, feat_dim=3).to(device)
classifier = PointnetPP_Segmentation(k=K, feat_dim=3).to(device)
# create creterion
creterion = nn.NLLLoss()
# create optimizer
optim = optimizer.Adam(list(encoder.parameters()) + list(classifier.parameters()))
losses = []


# *** TRAIN MODEL ***

start_time = time()
train_size = x_train.shape[0]
for e in range(1, 1+epochs):
    # reset
    loss = 0
    running_loss = 0
    batch_start_time = time()
    # shuffle data
    shuffle_idx = np.arange(train_size)
    np.random.shuffle(shuffle_idx)
    x_train, y_train = x_train[shuffle_idx], y_train[shuffle_idx]
    # batch-loop
    for b in range(train_size//batch_size):
        # get batch
        x_batch = x_train[b * batch_size : (b+1) * batch_size].to(device).float()
        y_batch = y_train[b * batch_size : (b+1) * batch_size].to(device).long()
        pos, feats = x_batch[:, :3, :], x_batch[:, 3:, :]
        # pass through model
        layer_outs = encoder.forward(pos, feats)
        class_log_probs = classifier(layer_outs)
        # transpose to match (batch, points, feats) and flatten afterwards
        class_log_probs = class_log_probs.transpose(1, 2).reshape(-1, K)
        # compute loss of every point
        loss += creterion(class_log_probs, y_batch.flatten())
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
        torch.save(encoder.state_dict(), "{0}-encoder.model".format(os.path.join(save_fpath, model_name)))
        torch.save(classifier.state_dict(), "{0}-classifier.model".format(os.path.join(save_fpath, model_name)))
        # create loss graph
        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        # plot losses and save figure
        ax.plot(losses)
        try:
            fig.savefig("{0}-lossgraph.pdf".format(os.path.join(save_fpath, model_name)), format='pdf')
        except Exception as e:
            print("[EXCEPTION] during save of plot:", e)
        # close figure to free memory
        plt.close()


# *** TEST MODEL ***

# confusion matrix
confusion = np.zeros((K, K))
# set eval
encoder.eval()
classifier.eval()
# no gradients
with torch.no_grad():
    test_size = x_test.shape[0]
    for b in range(math.floor(test_size/batch_size)):
        # get batch
        x_batch = x_test[b * batch_size : (b+1) * batch_size].to(device).float()
        y_batch = y_test[b * batch_size : (b+1) * batch_size].numpy()
        pos, feats = x_batch[:, :3, :], x_batch[:, 3:, :]
        # pass through model
        layer_outs = encoder.forward(pos, feats)
        class_probs = classifier(layer_outs)
        # transpose to match (batch, points, feats) and flatten afterwards
        class_probs = class_probs.transpose(1, 2).reshape(-1, K)
        # get predicted classes
        predicted = torch.max(class_probs, dim=-1)[1].cpu().numpy()
        # evaluate
        for actual, pred in zip(y_batch.flatten(), predicted):
            confusion[actual, pred] += 1
        # remove
        del x_batch, y_batch

# save confusion matrix
print(confusion)
np.savetxt(open("{0}-confusion.txt".format(os.path.join(save_fpath, model_name)), "w+"), confusion)


# *** SAVE ***

# save final encoder and classifier
torch.save(encoder.state_dict(), "{0}-encoder.model".format(os.path.join(save_fpath, model_name)))
torch.save(classifier.state_dict(), "{0}-classifier.model".format(os.path.join(save_fpath, model_name)))
