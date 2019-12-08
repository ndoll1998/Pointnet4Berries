
# import numpy
import numpy as np
# import torch
import torch
import torch.optim as optimizer
from torch.utils.data import TensorDataset, DataLoader

# import model
from Pointnet.models import Model_SEG
# import utils
from utils.data import build_data_seg
from utils.utils import compute_fscores
from utils.torchBoard import TorchBoard, ConfusionMatrix

# other imports
import os
import json
from time import time
from tqdm import tqdm
from random import sample
from collections import OrderedDict


# *** PARAMS ***

# cuda device
device = 'cpu'

# number of classes
class_bins = OrderedDict({
    'twig': ['twig', 'subtwig', 'berry'], 
    'rachis': ['rachis'],
    'peduncle': ['peduncle'],
    'hook': ['hook'],
    'None': ['None']
})
K = len(class_bins)
# used features
features = ['points', 'colors']
feature_dim = 3
# number of points and samples
n_points = 1024
n_samples = 40
# number of poinclouds per class for testing
n_test_pcs = 1

# initial checkpoint
encoder_init_checkpoint = None
segmentater_init_checkpoint = None
# training parameters
epochs = 20
batch_size = 4

# path to files
fpath = "H:/Pointclouds/Skeleton/Processed"
# save path
save_path = "H:/results/segmentation"
os.makedirs(save_path, exist_ok=True)


# *** READ AND CREATE TRAINING / TESTING DATA ***

print("LOADING POINTCLOUDS...")
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

print("PREPROCESSING POINTCLOUDS...")
# separate pointclouds into training and testing samples
train_pointclouds, test_pointclouds = {}, {}
for class_name, pcs in pointclouds.items():
    # get random subset to train from
    train_pc_names = sample(pcs.keys(), len(pcs) - n_test_pcs)
    test_pc_names = set(pcs.keys()) - set(train_pc_names)
    # add to dicts
    train_pointclouds[class_name] = sum([pcs[n] for n in train_pc_names], [])
    test_pointclouds[class_name] = sum([pcs[n] for n in test_pc_names], [])

# create training and testing datasets
train_data = TensorDataset(*build_data_seg(train_pointclouds, n_points, n_samples, class_bins, features=features))
test_data = TensorDataset(*build_data_seg(test_pointclouds, n_points, n_samples, class_bins, features=features))
# create training and testing dataloaders
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=1000)


# *** CREATE MODEL AND OPTIMIZER ***

# create model
model = Model_SEG(K=K, feat_dim=feature_dim)
model.load_encoder(encoder_init_checkpoint)
model.load_segmentater(segmentater_init_checkpoint)
model.to(device)
# create optimizer
optim = optimizer.Adam(model.parameters())


# *** SAVE PARAMETERS ***

with open(os.path.join(save_path, "config.json"), 'w+') as f:
    config = {
        "task": "segmentation",
        "classes": class_bins,
        "features": features,
        "n_points": n_points,
        "n_samples": n_samples, 
        "n_test_pointclouds": n_test_pcs,
        "epochs": epochs,
        "batch_size": batch_size, 
        "n_train_samples": len(train_data),
        "n_train_points": dict(zip(class_bins.keys(), map(int, np.bincount(train_data[:][-1].flatten().numpy())))),
        "n_test_samples": len(test_data),
        "n_test_points": dict(zip(class_bins.keys(), map(int, np.bincount(test_data[:][-1].flatten().numpy()))))
    }
    json.dump(config, f, indent=2, sort_keys=True)


# *** TRAIN AND TEST MODEL ***

print("TRAINING...")
# track losses and f-scores
tb = TorchBoard("Train_Loss", "Test_Loss", *class_bins.keys())
tb.add_stat(ConfusionMatrix(class_bins.keys(), name="Confusion", normalize=True))

start = time()
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
        print("Epoch {0}/{1} - Batch {2}/{3}\t- Average Loss {4:.02f}\t - Time {5:.04f}s"
            .format(epoch+1, epochs, i+1, len(train_dataloader), running_loss/(i+1), time() - start), end='\r')

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
        for actual, pred in zip(y_hat.flatten(), torch.argmax(y.reshape(-1, K), dim=-1)):
            confusion_matrix[actual, pred] += 1

    # update board
    tb.Confusion += confusion_matrix
    tb.Test_Loss += running_loss / len(test_dataloader)
    # compute f-scores from confusion matrix
    f_scores = compute_fscores(confusion_matrix)
    for c, f in zip(class_bins.keys(), f_scores):
        tb[c] += f

    # save board
    fig = tb.create_fig([[["Train_Loss", "Test_Loss"]], [class_bins.keys()], [["Confusion"]]], figsize=(8, 11))
    fig.savefig(os.path.join(save_path, "board.pdf"), format="pdf")
    # save model
    model.save(save_path, prefix="E{0}-".format(epoch))
    # end epoch
    print()
