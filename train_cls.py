
# import numpy
import numpy as np
# import torch
import torch
import torch.optim as optimizer

# import model
from models import Model_CLS
# import data-helpers
from data import build_data_cls
# import utils
from utils import normalize_pc, visualize_confusion_table, create_confusion_matrix, train_model

# other imports
import os
from tqdm import tqdm
from random import sample
from collections import OrderedDict
import matplotlib.pyplot as plt

# numpy percision for printing
np.set_printoptions(precision=3)


# *** PARAMS ***

# cuda device
device = 'cpu'
# number of classes
classes = ['CB', 'D', 'PN', 'R']
K = len(classes)
# path to files
fpath = "C:/Users/Niclas/Documents/Pointclouds/Bunch"
# number of points to use in training
n_points = 1024
# number of samples per pointcloud
n_samples = 500
# number of poinclouds per class for testing
samples_for_testing = 5
# initial checkpoint
encoder_init_checkpoint = None
classifier_init_checkpoint = None
# save path
save_path = "results/classify"
os.makedirs(save_path, exist_ok=True)

# training parameters
epochs = 100
batch_size = 20
# update parameters after n batches
update_interval = 1
# save model after every n-th epoch
save_interval = 1


# *** READ AND CREATE TRAINING / TESTING DATA ***

pointclouds = OrderedDict()
# open files
for directory in tqdm(os.listdir(fpath)):
    # ignore processed
    if directory == 'Processed':
        continue
    # build full directory
    full_dir = os.path.join(fpath, directory)
    # only read original files in
    pointclouds[directory] = [np.loadtxt(os.path.join(full_dir, fname), dtype=np.float32) for fname in os.listdir(full_dir) if 'OR' in fname]

# build data
x_train, y_train, x_test, y_test = build_data_cls(pointclouds, n_points, n_samples, features=['points', 'colors'])


# *** CREATE MODEL AND OPTIMIZER ***

# create model
model = Model_CLS(K=K, feat_dim=x_train.size(1) - 3)
model.load_encoder(encoder_init_checkpoint)
model.load_classifier(classifier_init_checkpoint)
model.to(device)
# create optimizer
optim = optimizer.Adam(model.parameters())


# *** TRAIN AND TEST MODEL ***

# track losses
losses = []
# callback function
def callback(epoch, loss):
    # add loss to losses
    losses.append(loss)
    # save
    if epoch % save_interval == 0:

        # build confusion table
        test_confusion = create_confusion_matrix(model, x_test, y_test, len(classes), batch_size, device=device)
        train_confusion = create_confusion_matrix(model, x_train, y_train, len(classes), batch_size, device=device)
        try:
            # save confusion matrix
            visualize_confusion_table(test_confusion, classes=classes, normalize=True)[0].savefig(os.path.join(save_path, "confusion-test.pdf"), format='pdf')
            visualize_confusion_table(train_confusion, classes=classes, normalize=True)[0].savefig(os.path.join(save_path, "confusion-train.pdf"), format='pdf')
        except Exception as e:
            # handle exception
            print("[EXCEPTION]", e)

        # create loss graph
        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        # plot losses and save figure
        ax.plot(losses)
        try:
            # save loss-graph
            fig.savefig(os.path.join(save_path, "lossgraph.pdf"), format='pdf')
        except Exception as e:
            print("[EXCEPTION] during save of plot:", e)

        # close figure to free memory
        plt.close()
        # save model
        model.save(save_path)

# train
train_model(model, x_train, y_train, optim, epochs, batch_size, update_interval, device=device, callback=callback)
