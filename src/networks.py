from torch._C import set_flush_denormal
import torchvision
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn.functional as F



class MLP( nn.Module ):  # pretrain the classifier to make income predictions.

    def __init__(self, n_features, n_hidden=256, p_dropout=0.2, num_classes=1, num_layers=4):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.p_dropout = p_dropout

        assert num_layers >= 2, "num_layers must be >= 2"

        self.network = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.network.append( nn.Linear(n_features, n_hidden) )
            else:
                self.network.append( nn.Linear(n_hidden, n_hidden) )
        self.network.append( nn.Linear(n_hidden, num_classes) )

    def forward(self, x):
        for layer in range(self.num_layers+1):
            x = self.network[layer](x)
            if layer != self.num_layers:
                if layer == self.num_layers - 1:
                    h = x
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
        return h, torch.sigmoid(x)




class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ResNet18_Encoder(nn.Module):
    def __init__(self, pretrained, n_hidden=128):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet.fc = Identity()
        # self.resnet.avgpool = Identity()
        # self.resnet.fc = nn.Linear(512, n_hidden)
        # self.fc1 = nn.Linear(1000, n_hidden)
        self.fc = nn.Linear(512, 1)

        # self.layer_norm = nn.LayerNorm(n_hidden, elementwise_affine=False)

    def forward(self, x):
        x = self.resnet(x)

        x = self.fc(x)
        x = torch.sigmoid(x)
        return x


# class LinearModel(nn.Module):
#     def __init__(self):
#         super(LinearModel, self).__init__()
#         # self.fc1 = nn.Linear(512, 512)
#         self.fc2 = nn.Linear(128, 1)
#         self.relu = nn.ReLU()
#         self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))

#     def forward(self, x):
#         # x = self.avg(x).view(-1, 512)
#         # x = self.fc1(x)
#         x = self.relu(x)
#         outputs = self.fc2(x)
#         return torch.sigmoid(outputs)



class ResNet50_Encoder(nn.Module):
    def __init__(self, pretrained, n_hidden=128):
        super().__init__()
        # self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet = torchvision.models.resnet50(pretrained=pretrained)
        # self.resnet.fc = Identity()
        # self.resnet.avgpool = Identity()
        self.resnet.fc = nn.Linear(2048, n_hidden)
        # self.fc1 = nn.Linear(1000, n_hidden)
        self.fc = nn.Linear(n_hidden, 1)

        self.layer_norm = nn.LayerNorm(n_hidden, elementwise_affine=False)

    def forward(self, x):
        x = self.resnet(x)

        # x = self.fc1( x )
        # x = self.relu(x)
        # print( "outputs", outputs.shape )
        # return outputs.view(-1, 512, 8, 8)
        # x = torch.sigmoid(x)
        x = F.relu(x)
        h = self.layer_norm(x)
        h1 = F.normalize(h, p=2)

        x = self.fc(x)
        x = torch.sigmoid(x)
        return h1, x