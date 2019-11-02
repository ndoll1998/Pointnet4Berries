# import pytorch framework
import torch
# import numpy
import numpy as np

# import random
from random import sample

# import Pointnet
from Pointnet.PointnetPP import PointnetPP_Encoder, PointnetPP_Classification
# import utils
from utils import normalize_pc

# import others
import os
from tqdm import tqdm

# cuda device
device = 'cpu'
# number of classes
K = 4
# number of points per pointcloud
n_points = 25_000
# number of samples per pointcloud
n_samples = 3
# batch-size
batch_size = 5
# evaluation data path
fpath = "C:/Users/doll0/Documents/Grapes/BBCH87_89"
# encoder and classifier state-dicts
encoder_state_dict_file = "C:/Users/doll0/Documents/results/BBCH87_89_v2-encoder.model"
classifier_state_dict_file = "C:/Users/doll0/Documents/results/BBCH87_89_v2-classifier.model"


# *** READ DATA ***

pointclouds = {}
# open files
for directory in tqdm(os.listdir(fpath)):
    # create entry
    full_dir = os.path.join(fpath, directory)
    # pointclouds[directory] = np.random.uniform(-1, 1, size=(len(os.listdir(full_dir)), 2 * n_points, 6)) * 100
    pointclouds[directory] = [np.loadtxt(os.path.join(full_dir, fname), dtype=np.float32) for fname in os.listdir(full_dir)]

# *** GENERATE TRAINING AND TESTING DATA ***

x, y = [], []
for i, pcs in enumerate(pointclouds.values()):
    # build train data
    for j in range(len(pcs)):
        # create multiple subclouds from one cloud
        for _ in range(n_samples):
            # check if pointcloud consists of enough points
            if pcs[j].shape[0] < n_points:
                continue
            # get random subset of points
            idx = sample(range(pcs[j].shape[0]), n_points)
            x.append(pcs[j][idx, :])
            y += [i]

# convert lists to numpy
x, y = np.array(x, dtype=np.float32), np.array(y, dtype=np.long)
# normalize pointclouds
x[:, :, :3] = normalize_pc(x[:, :, :3])
# normalize rgb values
x[:, :, 3:] /= 255
# convert to tensors and copy to device
x, y = torch.from_numpy(x), torch.from_numpy(y)
# transpose
x = x.transpose(1, 2)

# *** CREATE MODEL, OPTIMIZER AND LOSS ***

# create encoder and classifier
encoder = PointnetPP_Encoder(pos_dim=3, feat_dim=3)
classifier = PointnetPP_Classification(k=K)
# load state-dicts
encoder_state_dict = torch.load(encoder_state_dict_file)
encoder.load_state_dict(encoder_state_dict)

classifier_state_dict = torch.load(classifier_state_dict_file)
classifier.load_state_dict(classifier_state_dict)


# *** TEST MODEL ***

# confusion matrix
confusion = np.zeros((K, K))
# set eval
encoder.eval()
classifier.eval()
# no gradients
with torch.no_grad():
    test_size = x.shape[0]
    for b in range(test_size//batch_size):
        # get batch
        x_batch = x[b * batch_size : (b+1) * batch_size].to(device).float()
        y_batch = y[b * batch_size : (b+1) * batch_size].numpy()
        # pass through model
        pos, feats = x_batch[:, :3, :], x_batch[:, 3:, :]
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