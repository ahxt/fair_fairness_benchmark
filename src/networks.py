import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function



class GradReverse(Function):
    """
    borrwed from https://github.com/hanzhaoml/ICLR2020-CFair/blob/master/models.py
    Implement the gradient reversal layer for the convenience of domain adaptation neural network.
    The forward part is the identity function while the backward part is the negative function.
    """

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(x):
    return GradReverse.apply(x)



# class MLP_dep(nn.Module):

#     def __init__(self, n_features, n_hidden=256, p_dropout=0.2, num_classes=1, num_layers=4):
#         super(MLP_dep, self).__init__()
#         self.num_layers = num_layers
#         self.num_classes = num_classes
#         self.n_features = n_features
#         self.n_hidden = n_hidden
#         self.p_dropout = p_dropout

#         assert num_layers >= 2, "num_layers must be >= 2"

#         self.network = nn.ModuleList()
#         for layer in range(num_layers):
#             if layer == 0:
#                 self.network.append( nn.Linear(n_features, n_hidden) )
#             else:
#                 self.network.append( nn.Linear(n_hidden, n_hidden) )
#         self.network.append( nn.Linear(n_hidden, num_classes) )

#     def forward(self, x):
#         for layer in range(self.num_layers+1):
#             x = self.network[layer](x)
#             if layer != self.num_layers:
#                 if layer == self.num_layers - 1:
#                     h = x
#                 x = F.relu(x)
#                 x = F.dropout(x, training=self.training)
#         return h, torch.sigmoid(x)



class MLP(nn.Module):

    def __init__(self, n_features, mlp_layers= [512, 256, 64], p_dropout=0.2, num_classes=1):
        super(MLP, self).__init__()
        self.num_classes = num_classes
        self.mlp_layers = [n_features] + mlp_layers
        self.p_dropout = p_dropout

        self.network = nn.ModuleList( [nn.Linear(i, o) for i, o in zip( self.mlp_layers[:-1], self.mlp_layers[1:])] )
        self.head = nn.Linear(self.mlp_layers[-1], num_classes)

    def forward(self, x):
        for layer in self.network:
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        h = x
        x = self.head(x)
        return h, torch.sigmoid(x)




class AdvDebiasing(nn.Module):
    """
    modified from https://github.com/hanzhaoml/ICLR2020-CFair/blob/master/models.py
    Multi-layer perceptron with adversarial training for fairness.
    """

    def __init__(self, n_features, num_classes=1, hidden_layers=[60], adversary_layers=[50]):
        super(AdvDebiasing, self).__init__()
        self.input_dim = n_features
        self.num_classes = num_classes
        self.num_hidden_layers = len(hidden_layers)
        self.num_neurons = [self.input_dim] + hidden_layers

        # Parameters of hidden, fully-connected layers, feature learning component.
        self.hiddens = nn.ModuleList([nn.Linear(self.num_neurons[i], self.num_neurons[i + 1])for i in range(self.num_hidden_layers)])
        
        # Parameter of the final softmax classification layer.
        self.softmax = nn.Linear(self.num_neurons[-1], self.num_classes)

        # Parameter of the adversary classification layer.
        self.num_adversaries = [self.num_neurons[-1]] + adversary_layers
        self.num_adversaries_layers = len(adversary_layers)
        self.adversaries = nn.ModuleList([nn.Linear(self.num_adversaries[i], self.num_adversaries[i + 1])for i in range(self.num_adversaries_layers)])
        self.sensitive_cls = nn.Linear(self.num_adversaries[-1], 1)

    def forward(self, inputs):
        h_relu = inputs
        for hidden in self.hiddens:
            h_relu = F.relu(hidden(h_relu))

        # Classification probability.
        logprobs = torch.sigmoid(self.softmax(h_relu))

        # Adversary classification component.
        h_relu = grad_reverse(h_relu)
        for adversary in self.adversaries:
            h_relu = F.relu(adversary(h_relu))

        cls = torch.sigmoid(self.sensitive_cls(h_relu))
        return logprobs, cls

    def inference(self, inputs):
        h_relu = inputs
        for hidden in self.hiddens:
            h_relu = F.relu(hidden(h_relu))

        # Classification probability.
        logprobs = torch.sigmoid(self.softmax(h_relu))
        return None, logprobs






class CFairNet(nn.Module):
    """
    modified from https://github.com/hanzhaoml/ICLR2020-CFair/blob/master/models.py
    Multi-layer perceptron with adversarial training for conditional fairness.
    """
    #  n_features, num_classes=1, hidden_layers=[60], adversary_layers=[50]
    def __init__(self, configs, n_features, num_classes=1, hidden_layers=[60], adversary_layers=[50]):
        super(CFairNet, self).__init__()
        self.input_dim = n_features
        self.num_classes = num_classes
        self.num_hidden_layers = hidden_layers
        self.num_neurons = [self.input_dim] + hidden_layers

        # Parameters of hidden, fully-connected layers, feature learning component.
        self.hiddens = nn.ModuleList([nn.Linear(self.num_neurons[i], self.num_neurons[i + 1]) for i in range(self.num_hidden_layers)])

        # Parameter of the final softmax classification layer.
        self.softmax = nn.Linear(self.num_neurons[-1], num_classes)

        # Parameter of the conditional adversary classification layer.
        self.num_adversaries = [self.num_neurons[-1]] + adversary_layers
        self.num_adversaries_layers = len(adversary_layers)

        # Conditional adversaries for sensitive attribute classification, one separate adversarial classifier for
        # one class label.
        self.adversaries = nn.ModuleList([nn.ModuleList([nn.Linear(self.num_adversaries[i], self.num_adversaries[i + 1])
                                                        for i in range(self.num_adversaries_layers)])
                                        for _ in range(self.num_classes)])
        self.sensitive_cls = nn.ModuleList([nn.Linear(self.num_adversaries[-1], 1) for _ in range(self.num_classes)])

    def forward(self, inputs, labels):
        h_relu = inputs
        for hidden in self.hiddens:
            h_relu = F.relu(hidden(h_relu))
        # Classification probabilities.
        logprobs = F.log_softmax(self.softmax(h_relu), dim=1)
        # Adversary classification component.
        c_losses = []
        h_relu = grad_reverse(h_relu)
        for j in range(self.num_classes):
            idx = labels == j
            c_h_relu = h_relu[idx]
            for hidden in self.adversaries[j]:
                c_h_relu = F.relu(hidden(c_h_relu))
            c_cls = F.log_softmax(self.sensitive_cls[j](c_h_relu), dim=1)
            c_losses.append(c_cls)
        return logprobs, c_losses

    def inference(self, inputs):
        h_relu = inputs
        for hidden in self.hiddens:
            h_relu = F.relu(hidden(h_relu))
        # Classification probabilities.
        logprobs = F.log_softmax(self.softmax(h_relu), dim=1)
        return logprobs







class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x



import torch
from torch import nn
import torch.nn.functional as F


class RMSELoss(nn.Module):

    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss




class LAFTR(nn.Module):
    '''
    Mofified from https://github.com/louisabraham/feaml/blob/master/laftr.py
    '''

    def __init__(self,encoder,decoder,adversary,classifier=None,rec_loss=None,adv_loss=None,classif_loss=None,A_x=0,A_y=1,A_z=50):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.adversary = adversary
        self.classifier = classifier
        if rec_loss is None:
            rec_loss = RMSELoss()
        self.rec_loss = rec_loss
        if adv_loss is None:
            # adv_loss = nn.BCEWithLogitsLoss()
            adv_loss = nn.BCELoss()
        self.adv_loss = adv_loss
        if classif_loss is None:
            # classif_loss = nn.BCEWithLogitsLoss()
            classif_loss = nn.BCELoss()
        self.classif_loss = classif_loss
        self.A_x = A_x
        self.A_y = A_y
        self.A_z = A_z

    def forward(self, x, is_protected):
        if len(is_protected.shape) == 1:
            is_protected = is_protected[:, None]

        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)

        if self.classifier is not None:
            _, classif_pred = self.classifier(encoded)
        else:
            classif_pred = None
        encoded = grad_reverse(encoded)
        _, adv_pred = self.adversary(encoded)

        return encoded, decoded, classif_pred, adv_pred
    # return encoded, decoded, classif_pred, adv_pred

    def loss(self, x, y, is_protected):
        if len(is_protected.shape) == 1:
            is_protected = is_protected[:, None]
        if len(y.shape) == 1:
            y = y[:, None]

        encoded, decoded, classif_pred, adv_pred = self.forward(x, is_protected)

        # reconstruction error
        L_x = self.rec_loss(x, decoded)

        # prediction error
        L_y = (
            self.classif_loss(classif_pred, y)
            if classif_pred is not None
            else torch.tensor(0.0)
        )

        # adversarial loss
        L_z = self.adv_loss(adv_pred, is_protected)

        # total loss
        return self.A_x * L_x + self.A_y * L_y + self.A_z * L_z, L_x, L_y, L_z










class resnet18_encoder(nn.Module):
    def __init__(self, pretrained, n_hidden=512):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet.fc = Identity()
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.resnet(x)
        h = x
        x = self.fc(x)
        x = torch.sigmoid(x)
        return h, x

class resnet34_encoder(nn.Module):
    def __init__(self, pretrained, n_hidden=512):
        super().__init__()
        self.resnet = torchvision.models.resnet34(pretrained=pretrained)
        self.resnet.fc = Identity()
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.resnet(x)
        h = x
        x = self.fc(x)
        x = torch.sigmoid(x)
        return h, x


class resnet50_encoder(nn.Module):
    def __init__(self, pretrained, n_hidden=2048):
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=pretrained)
        self.resnet.fc = Identity()
        self.fc = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = self.resnet(x)
        h = x
        x = self.fc(x)
        x = torch.sigmoid(x)
        return h, x

class resnet101_encoder(nn.Module):
    def __init__(self, pretrained, n_hidden=2048):
        super().__init__()
        self.resnet = torchvision.models.resnet101(pretrained=pretrained)
        self.resnet.fc = Identity()
        self.fc = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = self.resnet(x)
        h = x
        x = self.fc(x)
        x = torch.sigmoid(x)
        return h, x


class resnet152_encoder(nn.Module):
    def __init__(self, pretrained, n_hidden=2048):
        super().__init__()
        self.resnet = torchvision.models.resnet152(pretrained=pretrained)
        self.resnet.fc = Identity()
        self.fc = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = self.resnet(x)
        h = x
        x = self.fc(x)
        x = torch.sigmoid(x)
        return h, x



class resnext50_32x4d_encoder(nn.Module):
    def __init__(self, pretrained, n_hidden=2048):
        super().__init__()
        self.resnet = torchvision.models.resnext50_32x4d(pretrained=pretrained)
        self.resnet.fc = Identity()
        self.fc = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = self.resnet(x)
        h = x
        x = self.fc(x)
        x = torch.sigmoid(x)
        return h, x


class resnext101_32x8d_encoder(nn.Module):
    def __init__(self, pretrained, n_hidden=2048):
        super().__init__()
        self.resnet = torchvision.models.resnext101_32x8d(pretrained=pretrained)
        self.resnet.fc = Identity()
        self.fc = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = self.resnet(x)
        h = x
        x = self.fc(x)
        x = torch.sigmoid(x)
        return h, x


class resnext101_64x4d_encoder(nn.Module):
    def __init__(self, pretrained, n_hidden=2048):
        super().__init__()
        self.resnet = torchvision.models.resnext101_64x4d(pretrained=pretrained)
        self.resnet.fc = Identity()
        self.fc = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = self.resnet(x)
        h = x
        x = self.fc(x)
        x = torch.sigmoid(x)
        return h, x



class wide_resnet50_2_encoder(nn.Module):
    def __init__(self, pretrained, n_hidden=2048):
        super().__init__()
        self.resnet = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        self.resnet.fc = Identity()
        self.fc = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = self.resnet(x)
        h = x
        x = self.fc(x)
        x = torch.sigmoid(x)
        return h, x


class wide_resnet101_2_encoder(nn.Module):
    def __init__(self, pretrained, n_hidden=2048):
        super().__init__()
        self.resnet = torchvision.models.wide_resnet101_2(pretrained=pretrained)
        self.resnet.fc = Identity()
        self.fc = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = self.resnet(x)
        h = x
        x = self.fc(x)
        x = torch.sigmoid(x)
        return h, x



class swin_t_encoder(nn.Module):
    def __init__(self, pretrained, n_hidden=1000):
        super().__init__()
        self.swin = torchvision.models.swin_t(pretrained=pretrained)
        self.swin.fc = Identity()
        self.fc = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = self.swin(x)
        h = x
        x = self.fc(x)
        x = torch.sigmoid(x)
        return h, x