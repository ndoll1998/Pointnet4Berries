
# import numpy
import numpy as np
# import torch
import torch
import torch.optim as optimizer
from torch.utils.data import TensorDataset, DataLoader

# import model
from Pointnet.models import Model_CLS
# import utils
from utils.data import build_data_cls
from utils.utils import compute_fscores
from utils.torchBoard import TorchBoard, ConfusionMatrix

# other imports
import os
import json
from tqdm import tqdm
from time import time
from random import sample
import matplotlib.pyplot as plt


# *** PARAMS ***

# cuda device
device = 'cpu'

# number of classes
classes = ['CB', 'D', 'PN', 'R']
K = len(classes)
# used features
features = ['x', 'y', 'z', 'r', 'g', 'b']
feature_dim = len(features) - 3
# number of points and samples
n_points = 1024
n_samples = 100
# number of pointclouds used for testing each class
n_test_pcs = 2

# initial checkpoint
encoder_init_checkpoint = None
classifier_init_checkpoint = None
# training parameters
epochs = 5
batch_size = 5
# optimizer parameters
learning_rate = 5e-4
weight_decay = 1e-2

# path to files
fpath = "H:/Pointclouds/Bunch"
# save path
save_path = "H:/results/classification"
os.makedirs(save_path, exist_ok=True)


# *** CREATE TRAINING / TESTING DATA ***

print("LOADING POINTCLOUDS...")
pointclouds = {}
# open files
for directory in tqdm(os.listdir(fpath)):
    # ignore processed
    if directory == 'Processed':
        continue
    # build full directory
    full_dir = os.path.join(fpath, directory)
    # only read original files in
    pointclouds[directory] = [np.loadtxt(os.path.join(full_dir, fname), dtype=np.float32) for fname in os.listdir(full_dir) if 'OR' in fname]

print("PREPROCESSING POINTCLOUDS...")
# separate pointclouds into training and testing samples
train_pointclouds, test_pointclouds = {}, {}
for directory, pcs in pointclouds.items():
    # get random subset to train from
    train_idx = sample(range(len(pcs)), len(pcs) - n_test_pcs)
    test_idx = set(range(len(pcs))) - set(train_idx)
    # add to dicts
    train_pointclouds[directory] = [pcs[i] for i in train_idx]
    test_pointclouds[directory] = [pcs[i] for i in test_idx]

# create training and testing datasets
train_data = TensorDataset(*build_data_cls(train_pointclouds, n_points, n_samples, features=features))
test_data = TensorDataset(*build_data_cls(test_pointclouds, n_points, n_samples, features=features))
# create training and testing dataloaders
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=100)


# *** CREATE MODEL AND OPTIMIZER ***

# create model
model = Model_CLS(K=K, feat_dim=feature_dim)
model.load_encoder(encoder_init_checkpoint)
model.load_classifier(classifier_init_checkpoint)
model.to(device)
# create optimizer
optim = optimizer.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay=)


# *** SAVE PARAMETERS ***

with open(os.path.join(save_path, "config.json"), 'w+') as f:
    config = {
        "task": "classification",
        "data": {
            "classes": classes,
            "features": features,
            "feature_dim": feature_dim,
            "n_points": n_points,
            "n_samples": n_samples, 
            "n_test_pointclouds": n_test_pcs,
            "n_train_samples": len(train_data),
            "n_train_points": dict(zip(classes, map(int, np.bincount(train_data[:][-1].flatten().numpy())))),
            "n_test_samples": len(test_data),
            "n_test_points": dict(zip(classes, map(int, np.bincount(test_data[:][-1].flatten().numpy())))),
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
        },
        "optimizer": {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        }
    }
    json.dump(config, f, indent=2)


# *** TRAIN AND TEST MODEL ***

print("TRAINING...")
# track losses and f-scores
tb = TorchBoard("Train_Loss", "Test_Loss", *classes)
tb.add_stat(ConfusionMatrix(classes, name="Confusion", normalize=True))

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
        print("Epoch {0}/{1}\t- Batch {2}/{3}\t- Average Loss {4:.02f}\t - Time {5:.04f}s"
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
        for actual, pred in zip(y_hat.flatten().cpu().numpy(), torch.argmax(y.reshape(-1, K), dim=-1).cpu().numpy()):
            confusion_matrix[actual, pred] += 1

    # update board
    tb.Confusion += confusion_matrix
    tb.Test_Loss += running_loss / len(test_dataloader)
    # compute f-scores from confusion matrix
    f_scores = compute_fscores(confusion_matrix)
    for c, f in zip(classes, f_scores):
        tb[c] += f

    # save board
    fig = tb.create_fig([[["Train_Loss", "Test_Loss"]], [classes], [["Confusion"]]], figsize=(8, 11))
    fig.savefig(os.path.join(save_path, "board.pdf"), format="pdf")
    # save model and best board if fscores improved
    if sum(f_scores) > best_fscore:
        fig.savefig(os.path.join(save_path, "best_board.pdf"), format="pdf")
        model.save(save_path)
        best_fscore = sum(f_scores)
    # end epoch
    print()