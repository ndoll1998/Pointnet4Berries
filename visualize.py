# import torch
import torch
# import numpy
import numpy as np

# import model
from Pointnet.models import Model_SEG
# import utils
from utils.data import build_data_seg, class2color
from utils.Visualizer import Visualizer

# import others
import os
import json
from collections import OrderedDict


# *** SET UP ***

# path to example
example_fpath = "H:/Pointclouds/Skeleton/Processed/CalardisBlanc_4E.feats"
# path to save folder
fpath = "H:/results/segmentation_v2_curvature_bigger_net"


# *** LOAD CONFIG ***

# open and load json file
with open(os.path.join(fpath, "config.json"), "r") as f:
    config = json.loads(f.read())
# make sure model is trained on segmentation task
assert config['task'] == 'segmentation', "Model must be trained on segmentation task"
# get classes predicted by model
class_bins = OrderedDict(config['data']['classes'])
classes = list(class_bins.keys())
# get features used by model
features = config['data']['features']
feature_dim = config['data']['feature_dim']


# *** LOAD AND PREPARE POINTCLOUD ***

# load file
pc = np.loadtxt(example_fpath)
# prepare data
x, y = build_data_seg({'example': [pc]}, -1, 1, class_bins=class_bins, features=features)

# *** LOAD MODEL ***

# create model
model = Model_SEG(K=len(classes), feat_dim=feature_dim)
# load parameters
model.load_encoder(os.path.join(fpath, 'encoder.model'))
model.load_segmentater(os.path.join(fpath, 'segmentater.model'))
# evaluate model
model.eval()


# *** PREDICT ***

with torch.no_grad():
    # predict classes
    log_probs = model.forward(x).cpu().numpy()
    prediction = np.argmax(log_probs, axis=-1).reshape(-1, 1)

# get actual and predicted colors
get_color = lambda i: class2color[classes[int(i)]]
actual_colors = np.apply_along_axis(get_color, axis=-1, arr=y.reshape(-1, 1))
pred_colors = np.apply_along_axis(get_color, axis=-1, arr=prediction)
# get points
points = x[0, :3, :].T.cpu().numpy()


# *** SHOW ***

# create visualizer
vis = Visualizer()
# visualize original pointcloud
vis.add_by_features(pc[:, :3], pc[:, 3:6] / 255, normalize=True)
vis.add_by_features(points, actual_colors / 255, normalize=False)
vis.add_by_features(points, pred_colors / 255, normalize=False)
vis.run()