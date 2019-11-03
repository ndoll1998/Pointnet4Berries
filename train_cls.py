
# import numpy
import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
# import torch
import torch
import torch.nn as nn
import torch.optim as optimizer

# import Pointnet
from Pointnet.PointnetPP import PointnetPP_Encoder, PointnetPP_Classification
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

# *** PARAMS ***

# cuda device
device = 'cuda:0'
# model name
model_name = "BBCH87_89_v2"
# number of classes
K = 4
# path to files
fpath = "C:/Users/doll0/Documents/Grapes/BBCH87_89/"
# number of points to use in training
n_points = 20_000
# number of samples per pointcloud
n_samples = 10
# save path
save_fpath = "C:/Users/doll0/Documents/results/"


# training parameters
epochs = 1000
batch_size = 6
# update parameters after n batches
update_interval = 1
# save model after every n-th epoch
save_interval = 2


# *** READ DATA ***

pointclouds = {}
# open files
for directory in tqdm(os.listdir(fpath)):
    # create entry
    full_dir = os.path.join(fpath, directory)
    # pointclouds[directory] = np.random.uniform(-1, 1, size=(len(os.listdir(full_dir)), 2 * n_points, 6)) * 100
    pointclouds[directory] = [np.loadtxt(os.path.join(full_dir, fname), dtype=np.float32) for fname in os.listdir(full_dir)]

# *** GENERATE TRAINING AND TESTING DATA ***

x_train, y_train = [], []
x_test, y_test = [], []
for i, pcs in enumerate(pointclouds.values()):
    # pick a random subset as train-set for current type
    train_idx = sample(range(len(pcs)), len(pcs) - 5)
    test_idx = set(range(len(pcs))).difference(train_idx)
    # build train data
    for j in train_idx:
        # create multiple subclouds from one cloud
        for _ in range(n_samples):
            # check if pointcloud consists of enough points
            if pcs[j].shape[0] < n_points:
                continue
            # get random subset of points
            idx = sample(range(pcs[j].shape[0]), n_points)
            x_train.append(pcs[j][idx, :])
            y_train += [i]
    # build test data
    for j in test_idx:
        # create multiple subclouds from one cloud
        for _ in range(n_samples):
            # check if pointcloud consists of enough points
            if pcs[j].shape[0] < n_points:
                continue
            # get random subset of points
            idx = sample(range(pcs[j].shape[0]), n_points)
            x_test.append(pcs[j][idx, :])
            y_test += [i]

# convert lists to numpy
x_train, x_test = np.array(x_train, dtype=np.float32), np.array(x_test, dtype=np.float32)
y_train, y_test = np.array(y_train, dtype=np.long), np.array(y_test, dtype=np.long)
# normalize pointclouds
x_train[:, :, :3] = normalize_pc(x_train[:, :, :3])
x_test[:, :, :3] = normalize_pc(x_test[:, :, :3])
# normalize rgb values
x_train[:, :, 3:] /= 255
x_test[:, :, 3:] /= 255
# convert to tensors and copy to device
x_train, x_test = torch.from_numpy(x_train), torch.from_numpy(x_test)
y_train, y_test = torch.from_numpy(y_train), torch.from_numpy(y_test)
# transpose
x_train, x_test = x_train.transpose(1, 2), x_test.transpose(1, 2)


# *** CREATE MODEL, OPTIMIZER AND LOSS ***

# create encoder and classifier
encoder = PointnetPP_Encoder(pos_dim=3, feat_dim=3).to(device)
classifier = PointnetPP_Classification(k=K).to(device)
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
        class_log_probs = classifier(layer_outs[-1][1])
        # compute loss
        loss += creterion(class_log_probs, y_batch)
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
        class_probs = classifier(layer_outs[-1][1])
        # get predicted classes
        predicted = torch.max(class_probs, dim=-1)[1].cpu().numpy()
        # evaluate
        for actual, pred in zip(y_batch, predicted):
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
