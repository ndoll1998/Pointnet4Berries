# imports
import sys
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import TensorDataset, DataLoader
# import model and utils
from Pointnet.models import Model_SEG
from utils.data import build_data_seg, class2color, seg_file_features
from utils.utils import compute_fscores, normalize_pc, align_principle_component, interpolate_pc
from utils.clustering import region_growing
from utils.augmentation import Augmenter, augment_rotate_pointcloud
from utils.torchBoard import TorchBoard, ConfusionMatrix
# import others
import os
import json
from time import time
from tqdm import tqdm
from math import ceil
from random import sample
from collections import OrderedDict

# *** SET UP ***

# cude device to use
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# hierarchical mapping
hierarchical_class_bins = [
    # First hierarchy
    OrderedDict({
        'twig:temp': ['twig', 'subtwig', 'berry'],
        'rachis:final': ['rachis', 'peduncle']
    }),
    # Second hierarchy - all classes in last hierarchy must be final
    OrderedDict({
        'subtwig:final': ['subtwig', 'berry'],
        'twig:final': ['twig']
    })
]; K = len(hierarchical_class_bins[0])
# augmentations
augmentations = [
    Augmenter(augment_rotate_pointcloud, feats=seg_file_features, apply_count=10, rot_axis='xyz')
]
# used features
features = ['x', 'y', 'z', 'r', 'g', 'b', 'length-xy', 'curvature']
feature_dim = len(features) - 3
# data preparation
align_pointclouds = False
interpolate_pointclouds = True
# number of points and samples
n_points = 35_000
n_samples = 10
# number of poinclouds per class for testing
n_test_pcs = 1
# initial checkpoint
encoder_init_checkpoint = None
segmentater_init_checkpoint = None
# training parameters
epochs = 40
batch_size = 4
# optimizer parameters
lr = 5e-4
weight_decay = 1e-2
# path to files
fpath = "data/"
# save path
save_path = "results/hierarchical"
os.makedirs(save_path, exist_ok=True)


# *** LOAD DATA ***

pointclouds = {}
# open files
for fname in tqdm(os.listdir(fpath)):
    # get name of pointcloud
    class_name, name = fname.replace('.xyzrgbc', '').split('_')[:2]
    # check for entry in pointclouds
    if class_name not in pointclouds:
        pointclouds[class_name] = {}
    if name not in pointclouds[class_name]:
        pointclouds[class_name][name] = []
    # create full path to file
    full_path = os.path.join(fpath, fname)
    # read pointcloud
    pointclouds[class_name][name].append(np.loadtxt(full_path, dtype=np.float32))

# separate pointclouds into training and testing samples
train_pointclouds, test_pointclouds = {}, {}
for class_name, pcs in pointclouds.items():
    # get random subset to train from
    train_pc_names = sample(pcs.keys(), len(pcs) - n_test_pcs)
    test_pc_names = set(pcs.keys()) - set(train_pc_names)
    # add to dicts
    train_pointclouds[class_name] = sum([pcs[n] for n in train_pc_names], [])
    test_pointclouds[class_name] = sum([pcs[n] for n in test_pc_names], [])


# *** PREPATE TRAINING AND EVALUATION DATA ***

def preprocess(pc):
    if align_pointclouds:
        # align pointcloud
        pc[:, :3] = align_principle_component(pc[:, :3])

    if (pc.shape[0] < n_points) and interpolate_pointclouds:
        # interpolate pointcloud
        k = ceil(n_points/pc.shape[0] - 1)
        points = interpolate_pc(pc[:, :3], pc[:, 6:9], k=k)
        # get nearest neighbors of each point
        tree = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(pc[:, :3])
        neighbor_idx = tree.kneighbors(points, return_distance=False)
        # interpolate features from nearest neighbors
        feats = pc[neighbor_idx, 3:-1].mean(axis=1)
        # get label from source point of each interpolated point
        labels = pc[:, -1:].repeat(k, axis=0)
        # concatenate all
        ipc = np.concatenate((points, feats, labels), axis=1)
        pc = np.concatenate((pc, ipc), axis=0)

    # return pointcloud
    return pc

def build_data(pointclouds):
    # remove modifiers from keys, i.e. :final, etc.
    make_class_bins = lambda class_bins: OrderedDict({key.split(':')[0]: value for key, value in class_bins.items()})

    data = tuple()
    # build first hierarchy data
    class_bins = hierarchical_class_bins[0]
    data += (build_data_seg(pointclouds, n_points, n_samples, make_class_bins(class_bins), features=features, augmentations=augmentations))

    # build data for following hierarchies
    for class_bins in hierarchical_class_bins[1:]:
        class_bin_ids = [list(class2color.keys()).index(n) for bin in class_bins.values() for n in bin]
        # build pointclouds of next hierarchy
        next_pointclouds = {}
        # go through all pointclouds
        for class_name, pcs in tqdm(pointclouds.items()):
            # print(class_name)
            next_pointclouds[class_name] = []
            for i, pc in enumerate(pcs):
                # remove points of classes not contained in any bin
                pc = pc[np.isin(pc[:, -1], class_bin_ids), :]
                points = normalize_pc(pc[:, :3])
                # apply region growing
                curvature, normals = pc[:, 9], pc[:, 6:9]
                cluster_mask = region_growing(points, normals, curvature, min_points=1_500)
                # get cluster ids ignoring outliers
                cluster_idx = np.unique(cluster_mask)
                cluster_idx = cluster_idx[(cluster_idx != -1)]
                # create unnormalized but aligned pointclouds from clusters
                next_pointclouds[class_name].extend([preprocess(pc[cluster_mask==i]) for i in cluster_idx])
        # update pointclouds
        pointclouds = next_pointclouds
        # build data from pointclouds
        data += (build_data_seg(pointclouds, n_points, n_samples//3, make_class_bins(class_bins), features=features, augmentations=augmentations))
    # concatenate data
    return tuple(torch.cat(values, dim=0) for values in zip(*data))

# create testing and training data
train_data = TensorDataset(*build_data(train_pointclouds))
test_data = TensorDataset(*build_data(test_pointclouds))
# create dataloaders
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=1)

# *** CREATE MODEL AND OPTIMIZER ***

# create model
model = Model_SEG(K=K, feat_dim=feature_dim)
model.load_encoder(encoder_init_checkpoint)
model.load_segmentater(segmentater_init_checkpoint)
model.to(device)
# create optimizer
optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


# *** SAVE CONFIGURATION ***

# build config
config = {
    "task": "hierarchical_segmentation",
    "augmentation": [augment.dict() for augment in augmentations],
    "preparation": {
        "align_pointclouds": align_pointclouds,
        "interpolate_pointclouds": interpolate_pointclouds
    },
    "data": {
        "hierarchy_classes": hierarchical_class_bins,
        "features": features,
        "feature_dim": feature_dim,
        "n_points": n_points,
        "n_samples": n_samples, 
        "n_test_pointclouds": n_test_pcs,
        "n_train_samples": len(train_data),
        "n_train_points": dict(zip(hierarchical_class_bins[0].keys(), map(int, np.bincount(train_data[:][-1].flatten().numpy())))),
        "n_test_samples": len(test_data),
        "n_test_points": dict(zip(hierarchical_class_bins[0].keys(), map(int, np.bincount(test_data[:][-1].flatten().numpy())))),
    },
    "training": {
        "epochs": epochs,
        "batch_size": batch_size,
    },
    "optimizer": {
        "learning_rate": lr,
        "weight_decay": weight_decay
    }
}
# save to file
with open(os.path.join(save_path, "config.json"), 'w+') as f:
    json.dump(config, f, indent=2)


# *** TRAINING MODEL ***

# track losses and f-scores
tb = TorchBoard("Train_Loss", "Test_Loss", *hierarchical_class_bins[0].keys())
tb.add_stat(ConfusionMatrix(hierarchical_class_bins[0].keys(), name="Confusion", normalize=True))

best_fscore, start = -1, time()
for epoch in range(epochs):

    # train model
    model.train()
    # reset for epoch
    start_epoch = time()
    running_loss = 0

    # train loop
    for i, (x, y_hat) in enumerate(train_dataloader):
        optim.zero_grad()

        # pass through model
        y = model.forward(x.to(device))
        # compute error
        loss = model.loss(y, y_hat.to(device))
        running_loss += loss.item()
        # update model parameters
        loss.backward()
        optim.step()
        # log
        print("\rEpoch {0}/{1}\t- Batch {2}/{3}\t- Average Loss {4:.02f}\t - Time {5:.04f}s"
            .format(epoch+1, epochs, i+1, len(train_dataloader), running_loss/(i+1), time() - start), end='', flush=True)

    # add to statistic
    tb.Train_Loss += running_loss / len(train_dataloader)

    # eval model
    model.eval()
    # initialize confusion matrix
    confusion_matrix = np.zeros((K, K))
    running_loss = 0

    for x, y_hat in test_dataloader:
        # pass through model and compute error
        y = model.forward(x.to(device))
        running_loss += model.loss(y, y_hat.to(device)).item()
        # update confusion matrix
        for actual, pred in zip(y_hat.flatten().cpu().numpy(), torch.argmax(y.reshape(-1, K), dim=-1).cpu().numpy()):
            confusion_matrix[actual, pred] += 1

    # update board
    tb.Confusion += confusion_matrix
    tb.Test_Loss += running_loss / len(test_dataloader)
    # compute f-scores from confusion matrix
    f_scores = compute_fscores(confusion_matrix)
    for c, f in zip(hierarchical_class_bins[0].keys(), f_scores):
        tb[c] += f
    # save board
    fig = tb.create_fig([[["Train_Loss", "Test_Loss"]], [hierarchical_class_bins[0].keys()], [["Confusion"]]], figsize=(8, 11))
    fig.savefig(os.path.join(save_path, "board.pdf"), format="pdf")
    # save model and best board if fscores improved
    if sum(f_scores) > best_fscore:
        fig.savefig(os.path.join(save_path, "best_board.pdf"), format="pdf")
        model.save(save_path)
        best_fscore = sum(f_scores)
    # end epoch
    print()