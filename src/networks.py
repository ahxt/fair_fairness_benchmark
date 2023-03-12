import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F



class MLP( nn.Module ):

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
    def __init__(self, pretrained, n_hidden=512):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet.fc = Identity()
        self.fc = nn.Linear(512, 1)

        # self.layer_norm = nn.LayerNorm(n_hidden, elementwise_affine=False)

    def forward(self, x):
        x = self.resnet(x)
        h = x
        x = self.fc(x)
        x = torch.sigmoid(x)
        return h, x



class ResNet50_Encoder(nn.Module):
    def __init__(self, pretrained, n_hidden=128):
        super().__init__()
        # self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet = torchvision.models.resnet50(pretrained=pretrained)
        self.resnet.fc = Identity()
        self.fc = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = self.resnet(x)
        h = x
        x = self.fc(x)
        x = torch.sigmoid(x)
        return h, x


class ResNet152_Encoder(nn.Module):
    def __init__(self, pretrained, n_hidden=128):
        super().__init__()
        # self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet = torchvision.models.resnet152(pretrained=pretrained)
        self.resnet.fc = Identity()
        self.fc = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = self.resnet(x)
        h = x
        x = self.fc(x)
        x = torch.sigmoid(x)
        return h, x


class vit_encoder(nn.Module):
    def __init__(self, pretrained, n_hidden=128):
        super().__init__()
        # self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.vit = torchvision.models.vit_base_patch16_224(pretrained=pretrained)
        self.vit.fc = Identity()
        self.fc = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = self.vit(x)
        h = x
        x = self.fc(x)
        x = torch.sigmoid(x)
        return h, x

class swin_encoder(nn.Module):
    def __init__(self, pretrained, n_hidden=128):
        super().__init__()
        # self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.swin = torchvision.models.swin_base_patch4_window7_224(pretrained=pretrained)
        self.swin.fc = Identity()
        self.fc = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = self.swin(x)
        h = x
        x = self.fc(x)
        x = torch.sigmoid(x)
        return h, x