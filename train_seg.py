
# import numpy
import numpy as np
# import torch
import torch
import torch.optim as optimizer

# import model
from models import Model_SEG
# import data-helpers
from data import build_data_seg
# import utils
from utils import normalize_pc, visualize_confusion_table, create_confusion_matrix, train_model

# other imports
import os
from tqdm import tqdm
from random import sample, shuffle
import matplotlib.pyplot as plt

# numpy percision for printing
np.set_printoptions(precision=3)


# *** PARAMS ***

# cuda device
device = 'cpu'
# number of classes
K = 7
classes = list(range(1, K+1))
# path to files
fpath = "C:/Users/Niclas/Documents/Pointclouds/Skeleton/Processed"
# number of points to use in training
n_points = 1024
# number of samples per pointcloud
n_samples = 3000
# number of poinclouds per class for testing
samples_for_testing = 1
# initial checkpoint
encoder_init_checkpoint = None
segmentater_init_checkpoint = None
# save path
save_path = "results/segmentate"
os.makedirs(save_path, exist_ok=True)

# training parameters
epochs = 100
batch_size = 10
# update parameters after n batches
update_interval = 1
# save model after every n-th epoch
save_interval = 1


# *** READ AND CREATE TRAINING / TESTING DATA ***

pointclouds = {}
# open files
for fname in tqdm(os.listdir(fpath)):
    # get class of pointcloud
    class_name = fname.split('_')[0]
    # check for entry in pointclouds
    if class_name not in pointclouds:
        pointclouds[class_name] = []
    # create full path to file
    full_path = os.path.join(fpath, fname)
    # read pointcloud
    pointclouds[class_name].append(np.loadtxt(full_path, dtype=np.float32))
    
# build data from pointclouds
x_train, y_train, x_test, y_test = build_data_seg(pointclouds, n_points, n_samples, samples_for_testing, features=['points', 'colors', 'length'])


# *** CREATE MODEL AND OPTIMIZER ***

# create model
model = Model_SEG(K=K, feat_dim=x_train.size(1) - 3)
model.load_encoder(encoder_init_checkpoint)
model.load_segmentater(segmentater_init_checkpoint)
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
