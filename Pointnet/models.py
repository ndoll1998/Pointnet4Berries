# improt pytorch-framework
import torch
import torch.nn as nn
# import PointnetPP
from Pointnet.PointnetPP import PointnetPP_Encoder, PointnetPP_Classification, PointnetPP_Segmentation
# import others
import os

# improt pytorch-framework
import torch
import torch.nn as nn
# import PointnetPP
from Pointnet.Pointnet import Pointnet_Encoder, Pointnet_Classification, Pointnet_Segmentation
from Pointnet.PointnetPP import PointnetPP_Encoder, PointnetPP_Classification, PointnetPP_Segmentation
# import others
import os

# *** POINTNET ***

class Model_CLS(nn.Module):
    """ Pointnet for Classification """

    def __init__(self, K, feat_dim=3):
        super(Model_CLS, self).__init__()
        # encoder and classifier
        self.encoder = Pointnet_Encoder(dim=3+feat_dim, shared_A=(32,), shared_B=(64,))
        self.classifier = Pointnet_Classification(k=K, g=64, shape=(128,))
        # creterion
        self.criterion = nn.NLLLoss()

    def forward(self, x):
        # encode and classify
        return self.classifier(self.encoder(x))

    def loss(self, y, y_hat):
        # compute loss
        return self.criterion(y, y_hat)

    def save(self, file_path, prefix=""):
        # save encoder and classifier separatly
        torch.save(self.encoder.state_dict(), os.path.join(file_path, prefix + "encoder.model"))
        torch.save(self.classifier.state_dict(), os.path.join(file_path, prefix + "classifier.model"))

    def load_encoder(self, file_path):
        # check if a file is given
        if file_path is None:
            return
        # load encoder state dict
        self.encoder.load_state_dict(torch.load(file_path))

    def load_classifier(self, file_path):
        # check if a file is given
        if file_path is None:
            return
        # load encoder state dict
        self.classifier.load_state_dict(torch.load(file_path))

class Model_SEG(nn.Module):
    """ Pointnet for Segmentation """

    def __init__(self, K, feat_dim=3):
        super(Model_SEG, self).__init__()
        # save number of classes
        self.K = K
        # encoder and segmentater
        self.encoder = Pointnet_Encoder(dim=3+feat_dim, shared_A=(64,), shared_B=(512,))
        self.segmentater = Pointnet_Segmentation(k=K, g=512, l=64, shared=(256,))
        # creterion
        self.criterion = nn.NLLLoss()

    def forward(self, x):
        # encode and classify
        return self.segmentater(self.encoder(x))

    def loss(self, y, y_hat):
        # compute loss
        return self.criterion(y.reshape(-1, self.K), y_hat.flatten())

    def save(self, file_path, prefix=""):
        # save encoder and segmentater separatly
        torch.save(self.encoder.state_dict(), os.path.join(file_path, prefix + "encoder.model"))
        torch.save(self.segmentater.state_dict(), os.path.join(file_path, prefix + "segmentater.model"))

    def load_encoder(self, file_path):
        # check if a file is given
        if file_path is None:
            return
        # load encoder state dict
        self.encoder.load_state_dict(torch.load(file_path, map_location='cpu'))

    def load_segmentater(self, file_path):
        # check if a file is given
        if file_path is None:
            return
        # load encoder state dict
        self.segmentater.load_state_dict(torch.load(file_path, map_location='cpu'))



# *** POINTNET++ ***

class ModelPP_CLS(nn.Module):
    """ Pointnet++ for Classification """

    def __init__(self, K):
        super(ModelPP_CLS, self).__init__()
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

    def save(self, file_path, prefix=""):
        # save encoder and classifier separatly
        torch.save(self.encoder.state_dict(), os.path.join(file_path, prefix + "encoder++.model"))
        torch.save(self.classifier.state_dict(), os.path.join(file_path, prefix + "classifier++.model"))

    def load_encoder(self, file_path):
        # check if a file is given
        if file_path is None:
            return
        # load encoder state dict
        self.encoder.load_state_dict(torch.load(file_path))

    def load_classifier(self, file_path):
        # check if a file is given
        if file_path is None:
            return
        # load encoder state dict
        self.classifier.load_state_dict(torch.load(file_path))

class ModelPP_SEG(nn.Module):
    """ Pointnet++ for Segmentation """

    def __init__(self, K, feat_dim=3):
        super(ModelPP_SEG, self).__init__()
        # save number of classes
        self.K = K
        # encoder and segmentater
        self.encoder = PointnetPP_Encoder(pos_dim=3, feat_dim=feat_dim)
        self.segmentater = PointnetPP_Segmentation(k=K, feat_dim=feat_dim)
        # creterion
        self.criterion = nn.NLLLoss()

    def forward(self, x):
        # encode and classify
        return self.segmentater(self.encoder(x))

    def loss(self, y, y_hat):
        # compute loss
        return self.criterion(y.reshape(-1, self.K), y_hat.flatten())

    def save(self, file_path, prefix=""):
        # save encoder and segmentater separatly
        torch.save(self.encoder.state_dict(), os.path.join(file_path, prefix + "encoder++.model"))
        torch.save(self.segmentater.state_dict(), os.path.join(file_path, prefix + "segmentater++.model"))

    def load_encoder(self, file_path):
        # check if a file is given
        if file_path is None:
            return
        # load encoder state dict
        self.encoder.load_state_dict(torch.load(file_path, map_location='cpu'))

    def load_segmentater(self, file_path):
        # check if a file is given
        if file_path is None:
            return
        # load encoder state dict
        self.segmentater.load_state_dict(torch.load(file_path, map_location='cpu'))

