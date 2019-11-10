
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

# numpy percision for printing
np.set_printoptions(precision=3)

# *** PARAMS ***

# cuda device
device = 'cuda:0'
# number of classes
classes = ['CB', 'D', 'PN', 'R']
K = len(classes)
# path to files
fpath = "C:/Users/doll0.SGN/Documents/Grapes/BBCH87_89/"
# number of points to use in training
n_points = 20_000
# number of samples per pointcloud
n_samples = 10
# save path
save_path = "C:/Users/doll0.SGN/Documents/results/test"
os.makedirs(save_path, exist_ok=True)

# training parameters
epochs = 500
batch_size = 6
# update parameters after n batches
update_interval = 1
# save model after every n-th epoch
save_interval = 2


# *** READ AND CREATE TRAINING / TESTING DATA ***

pointclouds = OrderedDict()
# open files
for directory in tqdm(os.listdir(fpath)):
    # create entry
    full_dir = os.path.join(fpath, directory)
    pointclouds[directory] = [np.loadtxt(os.path.join(full_dir, fname), dtype=np.float32) for fname in os.listdir(full_dir)]

# build data
x_train, y_train, x_test, y_test = build_data_cls(pointclouds, n_points, n_samples)


# *** CREATE MODEL AND OPTIMIZER ***

# create model
model = Model_CLS(K=K).to(device)
# create optimizer
optim = optimizer.Adam(model.parameters())


# *** TRAIN AND TEST MODEL ***

# train
train_model(model, x_train, y_train, optim, epochs, batch_size, update_interval, save_interval, save_path, device=device)
# build confusion table
confusion = create_confusion_matrix(model, x_test, y_test, K, batch_size, device=device)
# save confusion matrix
visualize_confusion_table(confusion, classes=classes, normalize=True)[0].savefig(os.path.join(save_path, "confusion-norm.pdf"), format='pdf')
visualize_confusion_table(confusion, classes=classes, normalize=False)[0].savefig(os.path.join(save_path, "confusion-total.pdf"), format='pdf')
