# import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy
import numpy as np


# *** TRANFORM NETWORK ***

class TNet(nn.Module):

    def __init__(self, dim=3, encoder=(64, 128, 1024), decoder=(512, 256), dropout_rate=0.3):
        super(TNet, self).__init__()
        # save dimension
        self.dim = dim
        # Convolution layers as shared MLP
        self.convs = nn.ModuleList([nn.Conv1d(in_, out_, 1) for in_, out_ in zip((dim,) + encoder[:-1], encoder)])
        self.conv_batchnorms = nn.ModuleList([nn.BatchNorm1d(n) for n in encoder])
        # fully connected layers
        self.decode_linear = nn.ModuleList([nn.Linear(in_, out_) for in_, out_ in zip((encoder[-1],) + decoder[:-1], decoder)])
        self.decode_batchnorm = nn.ModuleList([nn.BatchNorm1d(n) for n in decoder])
        # fully connected to convert to matrix of given dimension
        self.linear = nn.Linear(decoder[-1], dim ** 2)
        # dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        y = x
        # pass each point through shared mlp
        for conv, bn in zip(self.convs, self.conv_batchnorms):
            y = F.relu(bn(conv(y)))
        # apply max-pooling along points
        y = torch.max(y, dim=2)[0]
        # pass through decoder layers
        for linear, bn in zip(self.decode_linear, self.decode_batchnorm):
            y = F.relu(bn(linear(y)))
        # apply dropout
        y = self.dropout(y)
        # convert to translation matrix
        y = self.linear(y).view(-1, self.dim, self.dim)

        # compute panelty
        panelty = torch.norm(torch.eye(self.dim).to(y.device) - y)
        # apply transform matrix to given points
        x = x.transpose(1, 2) @ y

        # return transformed input and panelty
        return x.transpose(1, 2), panelty


# *** POINTNET ***

class Pointnet_Encoder(nn.Module):

    def __init__(self, dim=3, shared_A=(64, 64), shared_B=(64, 128, 1024)):
        super(Pointnet_Encoder, self).__init__()
        # transform Networks
        self.tnet_A = TNet(dim=dim, encoder=(64,), decoder=(32,))
        self.tnet_B = TNet(dim=shared_A[-1], encoder=(64,), decoder=(32,))
        # shared mlps
        self.convs_A = nn.ModuleList([nn.Conv1d(in_, out_, 1) for in_, out_ in zip((dim,) + shared_A[:-1], shared_A)])
        self.convs_B = nn.ModuleList([nn.Conv1d(in_, out_, 1) for in_, out_ in zip((shared_A[-1],) + shared_B[:-1], shared_B)])
        # batchnorm layers
        self.batchnorms_A = nn.ModuleList([nn.BatchNorm1d(n) for n in shared_A])
        self.batchnorms_B = nn.ModuleList([nn.BatchNorm1d(n) for n in shared_B])
        # current panelty
        self.panelty_ = 0

    def forward(self, x):
        out_A = x
        # apply first transformation and shared mlp
        out_A, panelty_A = self.tnet_A.forward(out_A)
        for conv, bn in zip(self.convs_A, self.batchnorms_A):
            out_A = F.relu(bn(conv(out_A)))
        # apply second transformation and shared mlp
        out_B, panelty_B = self.tnet_B(out_A)
        for conv, bn in zip(self.convs_B, self.batchnorms_B):
            out_B = F.relu(bn(conv(out_B)))
        # apply max-pooling along points
        global_feats = torch.max(out_B, dim=2)[0]
        # update panelty
        self.panelty_ += panelty_A + panelty_B
        # return global and local features
        return global_feats, out_A

    def panelty(self):
        # return and reset panelty
        cur_panelty = self.panelty_
        self.panelty_ = 0
        return cur_panelty

class Pointnet_Classification(nn.Module):

    def __init__(self, k, g=1024, shape=(512, 256), dropout_rate=0.3):
        super(Pointnet_Classification, self).__init__()
        # create linear layers
        self.linear = nn.ModuleList([nn.Linear(in_, out_) for in_, out_ in zip((g,) + shape[:-1], shape)])
        self.batchnorms = nn.ModuleList([nn.BatchNorm1d(n) for n in shape])
        # classification layer
        self.classify = nn.Linear(shape[-1], k)
        # dropout-layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, global_local_feats):
        # get features of interest
        x = global_local_feats[0]
        # pass theough linear layers
        for linear, bn in zip(self.linear, self.batchnorms):
            x = torch.sigmoid(bn(linear(x)))
        # apply dropout
        x = self.dropout(x)
        # get class log-probs
        return F.log_softmax(self.classify(x), dim=1)


class Pointnet_Segmentation(nn.Module):

    def __init__(self, k, g=1024, l=64, shared=(512, 256, 128, 128), dropout_rate=0.3):
        super(Pointnet_Segmentation, self).__init__()
        # create linear layer for global features
        self.linear = nn.Linear(g, shared[0], bias=False)
        # create shared mlps
        self.convs = nn.ModuleList([nn.Conv1d(in_, out_, 1) for in_, out_ in zip((l,) + shared[:-1], shared)])
        self.batchnorms = nn.ModuleList([nn.BatchNorm1d(n) for n in shared])
        # create classification layer
        self.classify = nn.Conv1d(shared[-1], k, 1)
        # dropout-layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, global_local_feats):
        # separate features
        global_feats, local_feats = global_local_feats
        # pretend concatenation of global and local features
        y = self.convs[0](local_feats) + self.linear(global_feats).unsqueeze(2)
        y = F.relu(self.batchnorms[0](y))
        # pass through remaining mlp
        for (conv, bn) in zip(self.convs[1:], self.batchnorms[1:]):
            y = F.relu(bn(conv(y)))
        # apply dropout
        y = self.dropout(y)
        # transpose to match (batch, points, feats)
        class_log_probs = F.log_softmax(self.classify(y), dim=1)
        return class_log_probs.transpose(1, 2)

        

# *** SCRIPT ***

if __name__ == '__main__':
    # create random points
    n_examples, features, n_points = 2, 512, 64
    points = np.random.uniform(-1, 1, size=(n_examples, features, n_points))
    points = torch.from_numpy(points).float()
    # encode points
    encoder = Pointnet_Encoder(dim=features)
    global_feats, local_feats = encoder.forward(points)
    # classify
    classifier = Pointnet_Classification(10)
    class_probs = classifier.forward(global_feats)
    # semantic segmentation
    segmentater = Pointnet_Segmentation(10)
    segmentater.forward(global_feats, local_feats)