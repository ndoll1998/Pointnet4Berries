
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

# numpy percision for printing
np.set_printoptions(precision=3)


# *** PARAMS ***

# cuda device
device = 'cuda:0'
# number of classes
K = 7
# path to files
fpath = "C:/Users/doll0.SGN/Documents/Grapes/Skeletons_seg_normals/"
# number of points to use in training
n_points = 25_000
# number of samples per pointcloud
n_samples = 10
# number of poinclouds per class for testing
samples_for_testing = 1
# initial checkpoint
encoder_init_checkpoint = None
segmentater_init_checkpoint = None
# save path
save_path = "C:/Users/doll0.SGN/Documents/results/Skeletons_normals"
os.makedirs(save_path, exist_ok=True)

# training parameters
epochs = 5000
batch_size = 3
# update parameters after n batches
update_interval = 2
# save model after every n-th epoch
save_interval = 2


# *** READ AND CREATE TRAINING / TESTING DATA ***

pointclouds = {}
# open files
for directory in tqdm(os.listdir(fpath)):
    # create entry
    full_dir = os.path.join(fpath, directory)
    # pointclouds[directory] = np.random.uniform(-1, 1, size=(len(os.listdir(full_dir)), 2 * n_points, 6)) * 100
    pointclouds[directory] = [np.loadtxt(os.path.join(full_dir, fname), dtype=np.float32) for fname in os.listdir(full_dir)]

# build data from pointclouds
x_train, y_train, x_test, y_test = build_data_seg(pointclouds, n_points, n_samples, samples_for_testing)
# get feature-dimension
feat_dim = x_train.size(1) - 3


# *** ANALYSE DATA ***

# print(list([np.sum(y_train.numpy()==i) for i in np.unique(y_train)]))
# exit()

# *** CREATE MODEL AND OPTIMIZER ***

# create model
model = Model_SEG(K=K, feat_dim=feat_dim)
model.load_encoder(encoder_init_checkpoint)
model.load_segmentater(segmentater_init_checkpoint)
model.to(device)
# create optimizer
optim = optimizer.Adam(model.parameters())


# *** TRAIN AND TEST MODEL ***

# train
train_model(model, x_train, y_train, optim, epochs, batch_size, update_interval, save_interval, save_path, device=device)
# build confusion table
confusion = create_confusion_matrix(model, x_test, y_test, K, batch_size, device=device)
# save confusion matrix
visualize_confusion_table(confusion, classes=range(1, K+1), normalize=True)[0].savefig(os.path.join(save_path, "confusion-norm.pdf"), format='pdf')
visualize_confusion_table(confusion, classes=range(1, K+1), normalize=False)[0].savefig(os.path.join(save_path, "confusion-total.pdf"), format='pdf')
