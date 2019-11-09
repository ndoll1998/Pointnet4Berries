# improt pytorch-framework
import torch
import torch.nn as nn
# import PointnetPP
from Pointnet.PointnetPP import PointnetPP_Encoder, PointnetPP_Classification, PointnetPP_Segmentation
# import others
import os

class Model_CLS(nn.Module):
    """ Pointnet++ for Classification """

    def __init__(self, K):
        super(Model_CLS, self).__init__()
        # encoder and classifier
        self.encoder = PointnetPP_Encoder(pos_dim=3, feat_dim=3)
        self.classifier = PointnetPP_Classification(k=K)
        # creterion
        self.criterion = nn.NLLLoss()

    def forward(self, x):
        # encode and classify
        return self.classifier(self.encoder(x))

    def loss(self, y, y_hat):
        # compute loss
        return self.criterion(y, y_hat)

    def save(self, file_path):
        # save encoder and classifier separatly
        torch.save(self.encoder.state_dict(), os.path.join(file_path, "encoder.model"))
        torch.save(self.classifier.state_dict(), os.path.join(file_path, "classifier.model"))


class Model_SEG(nn.Module):
    """ Pointnet++ for Segmentation """

    def __init__(self, K):
        super(Model_SEG, self).__init__()
        # save number of classes
        self.K = K
        # encoder and segmentater
        self.encoder = PointnetPP_Encoder(pos_dim=3, feat_dim=3)
        self.segmentater = PointnetPP_Segmentation(k=K, feat_dim=3)
        # creterion
        self.criterion = nn.NLLLoss()

    def forward(self, x):
        # encode and classify
        return self.segmentater(self.encoder(x))

    def loss(self, y, y_hat):
        # compute loss
        return self.criterion(y.reshape(-1, self.K), y_hat.flatten())

    def save(self, file_path):
        # save encoder and segmentater separatly
        torch.save(self.encoder.state_dict(), os.path.join(file_path, "encoder.model"))
        torch.save(self.segmentater.state_dict(), os.path.join(file_path, "segmentater.model"))


