# import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# import utils
from .utils import sample_and_group, interpolate
# import Pointnet
from .Pointnet import Pointnet_Encoder, Pointnet_Classification


# *** ABSTRACTION AND INTERPOLATION ***

class SetAbstraction(nn.Module):

    def __init__(self, n_clusters, n_samples, radius, dim=3, shared=(64, 128, 1024)):
        super(SetAbstraction, self).__init__()
        # create shared MLP
        self.convs = nn.ModuleList([nn.Conv2d(in_, out_, 1) for in_, out_ in zip((dim,) + shared[:-1], shared)])
        self.batchnorms = nn.ModuleList([nn.BatchNorm2d(n) for n in shared])
        # sampling - grouping parameters
        self.sample_group_params = {
            'n_clusters':   n_clusters,
            'n_sample':     n_samples,
            'radius':       radius
        }

    def forward(self, pos, feats):
        # sample and group
        centroids, grouped = sample_and_group(pos, feats, **self.sample_group_params)
        # pass through mlp
        x = grouped.transpose(1, 3)
        for conv, bn in zip(self.convs, self.batchnorms):
            x = F.relu(bn(conv(x)))
        # apply max-pooling over clusters
        feats = torch.max(x, dim=2)[0]
        # concatenate with centroids
        return centroids.transpose(1, 2), feats


class FeaturePropagation(nn.Module):

    def __init__(self, dim=3, shared=(512, 256, 128)):
        super(FeaturePropagation, self).__init__()
        # create shared MLP
        self.convs = nn.ModuleList([nn.Conv1d(in_, out_, 1) for in_, out_ in zip((dim,) + shared[:-1], shared)])
        self.batchnorms = nn.ModuleList([nn.BatchNorm1d(n) for n in shared])

    def forward(self, y_points, y_feats, x_points, x_feats):
        # interpolate points
        i_points = interpolate(y_points, x_points, y_feats, x_feats)
        # pass through network
        y = i_points.transpose(1, 2)
        for conv, bn in zip(self.convs, self.batchnorms):
            y = F.relu(bn(conv(y)))
        # return
        return y


# *** POINTNET++ ***

class PointnetPP_Encoder(nn.Module):

    def __init__(self, pos_dim=3, feat_dim=0):
        super(PointnetPP_Encoder, self).__init__()
        # create abstraction layers
        self.setAbstractions =  nn.ModuleList([
            SetAbstraction(1024, 32, 0.1, dim=feat_dim+pos_dim, shared=(32, 32, 64)),
            SetAbstraction(256,  32, 0.2, dim=64      +pos_dim, shared=(64, 64, 128)),
            SetAbstraction(128,  32, 0.4, dim=128     +pos_dim, shared=(128, 128, 256)),
            SetAbstraction(64,   32, 0.8, dim=256     +pos_dim, shared=(256, 256, 512))
        ])
    
    def forward(self, pos, feats):
        layer_output = [(pos, feats)]
        # pass input through all abstraction layers
        for layer in self.setAbstractions:
            layer_output.append(layer(*layer_output[-1]))
        # return each layer output including input
        return layer_output


class PointnetPP_Classification(nn.Module):

    def __init__(self, k, feat_dim=512, shared_A=(64, 64), shared_B=(64, 128, 1024), shape=(512, 256)):
        super(PointnetPP_Classification, self).__init__()
        # create a pointnet encoder and classifier
        self.encoder = Pointnet_Encoder(dim=feat_dim)
        self.decoder = Pointnet_Classification(k=k, g=shared_B[-1], shape=shape)

    def forward(self, feats):
        # pass though pointnet
        return self.decoder(self.encoder(feats)[0])


class PointnetPP_Segmentation(nn.Module):

    def __init__(self, k, feat_dim=0, shared=(128, 128), dropout_rate=0.3):
        super(PointnetPP_Segmentation, self).__init__()
        # create propagation layers
        self.featPropagations = nn.ModuleList([
            FeaturePropagation(dim=512+256,        shared=(256, 256)),
            FeaturePropagation(dim=256+128,        shared=(256, 256)),
            FeaturePropagation(dim=320,            shared=(256, 128)),
            FeaturePropagation(dim=128 + feat_dim, shared=(128, 128))
        ])
        # create classification network
        self.convs = [nn.Conv1d(in_, out_, 1) for in_, out_ in zip((128,) + shared[:-1], shared)]
        self.batchnorms = [nn.BatchNorm1d(n) for n in shared]
        # create classification layer
        self.classify = nn.Conv1d(shared[-1], k, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, layer_outs):
        # get very fist features
        prop_feats = layer_outs[-1][1]
        for i, featProp in enumerate(self.featPropagations):
            prop_feats = featProp(layer_outs[-i-1][0], prop_feats, *layer_outs[-i-2])
        # pass though shared-mlp
        for conv, bn in zip(self.convs, self.batchnorms):
            prop_feats = F.relu(bn(conv(prop_feats)))
        # classify
        prop_feats = self.dropout(prop_feats)
        return F.log_softmax(self.classify(prop_feats), dim=1)


# *** SCRIPT ***

if __name__ == '__main__':
    
    # import numpy
    import numpy as np

    # create random points
    a = np.random.uniform(-1, 1, size=(2, 10, 50))
    a = torch.from_numpy(a).float()
    # separate position from features
    pos_A, feats_A = a[:, :3, :], a[:, 3:, :]
    # set abstraction
    encoder = PointnetPP_Encoder(pos_dim=pos_A.size(1), feat_dim=feats_A.size(1))
    layer_outs = encoder.forward(pos_A, feats_A)
    # classifier
    classifier = PointnetPP_Classification(7)
    class_probs = classifier.forward(layer_outs[-1][1])
    # segmentater
    segementater = PointnetPP_Segmentation(k=3, feat_dim=feats_A.size(1))
    segementater.forward(layer_outs)

         