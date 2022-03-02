#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 12:30:27 2022

@author: etienne
"""

import torch.nn.functional as F
import torch as th
from torch import nn
from itertools import combinations
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import silhouette_score as ss
import warnings
warnings.filterwarnings("ignore")

cuda_device = 0
# device = th.device("cuda:%d" % cuda_device if th.cuda.is_available() else "cpu")
device = 'cpu'
dtype = th.float64


        



def get_sim_forces(f1, f2):
    f1 = f1.reshape(1,-1)
    f2 = f2.reshape(1,-1)
    return th.cdist(f1,f2)
    # return .5*(F.cosine_similarity(f1, f2)+1).squeeze()

def get_node2node_forces(embedding, path_for):
    mat = th.zeros(embedding.shape[0], embedding.shape[0]).to(device, dtype)
    for i,j in combinations(range(embedding.shape[0]),2):
        f1 = embedding[i,:]
        f2 = embedding[j,:]
        mat[i,j] = get_sim_forces(f1, f2)
        mat[j,i] = get_sim_forces(f1, f2)
    return mat.mul(path_for)
            
def get_silhouette(force,fr):
    tmp = force.cpu().detach().numpy()
    labels = np.zeros(len(fr),int)
    for c in list(fr):
        vec = fr[c].values
        for i in np.where(vec==1): labels[i] = int(c.replace('C',''))
    return Variable(th.tensor(ss(tmp, metric='precomputed', labels=labels)).reshape(1,1), requires_grad=True) 

def get_silhouette2(features,fr):
    tmp = features.cpu().detach().numpy()
    labels = np.zeros(len(fr),int)
    for c in list(fr):
        vec = fr[c].values
        for i in np.where(vec==1): labels[i] = int(c.replace('C',''))
    return Variable(th.tensor(ss(tmp, labels=labels)).reshape(1,1), requires_grad=True) 

# def scale_matrice(matrice):
#     for i in range(matrice.shape[1]):
#         mini = th.min(matrice[:, i])
#         maxi = th.max(matrice[:, i])
#         matrice[:, i] = (matrice[:, i] - mini)/(maxi-mini)
#         # tmp_mat[:, i] = std * (maxi - mini) + mini
#     return matrice


class Loss(nn.Module):


    def __init__(self,weight):
        super().__init__()
        # self.criterion = nn.MSELoss()
        self.criterion = nn.L1Loss()
        self.weight = weight
        # self.weight = th.tensor([.15, .85]).to(device, dtype)

    def forward(self, sil_pred, sil_true, y_pred, y_true):
        # return self.weight @ th.tensor([self.discrim_loss(y_pred, y_true), self.encoder_loss]).to(device, dtype)
        return self.weight[0] * self.criterion(y_pred, y_true) + self.weight[1]* self.criterion(sil_pred, sil_true)

          

def controversialencoderloss(
        embedding,
        membership
        ):
    
    def in_sim():
        som = 0
        for c in list(membership):
            nodes = membership[membership[c] == 1].index
            for i, j in combinations(nodes, 2):
                f1 = embedding[i, :]
                f2 = embedding[j, :]
                som += get_sim_forces(f1, f2)
        return som
    
    def out_sim():
        som = 0
        groups = list(membership)
        for i in range(len(groups)):
            nodes1 = membership[membership[groups[i]] == 1].index
            for n1 in nodes1:
                f1 = embedding[n1, :]
                som_list = []
                for j in range(len(groups)):
                    if i != j:
                        nodes2 = membership[membership[groups[j]] == 1].index
                        tmp_som = 0
                        for n2 in nodes2:
                            f2 = embedding[n2,:]
                            tmp_som += get_sim_forces(f1, f2)
                        som_list.append(tmp_som)
                som += min(som_list)
        return som
    
    pred_silhouette = (in_sim() - out_sim())/th.max(th.tensor([in_sim(), out_sim()]))
    # print(pred_silhouette)
    # expected_silhouette = Variable(th.tensor(1).to(device, dtype), requires_grad=False)
    # encodloss = F.mse_loss(pred_silhouette, expected_silhouette)
   
    
    return th.tensor(1).to(device, dtype) - pred_silhouette

def controversialdiscriminatorloss(
        predicted_prob,
        expected_proba
        ):
    discrimloss = F.mse_loss(Variable(predicted_prob, requires_grad=True), expected_proba)
    
    return discrimloss




class ControGraphConvLayer(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        p2p,
        bias=True
    ):
        super(ControGraphConvLayer, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.bias = bias
        self.p2p = p2p
        self.layer = nn.Linear(in_feats, out_feats, bias=bias).to(device, dtype)
        self.reset_parameters()
        
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.layer.weight, gain=gain).clamp(0,1) 
    
            
    def forward(
        self, 
        F_,   # node2node exerted matrix forces
        X    # node features matrix
    ):
        input_ = F_.to(device, dtype)@X.to(device, dtype)
        new_features = self.layer(input_)
        new_force = get_node2node_forces(new_features, self.p2p)
        return new_force, new_features
    

class controversialnet2(nn.Module):
    def __init__(
        self, 
        in_feats1, 
        hidden_feats1,
        hidden_feats2,
        bias = True
    ):
        
        def reset_net_parameters(net):
            gain = nn.init.calculate_gain('relu')
            for i in range(len(net)):
                try:
                    net[i].weight = (nn.init.xavier_uniform_(net[i].weight, gain=gain).sign()+1)/2
                except:
                    pass
            # for name, layer in net.named_modules():
            # # for i in range(len(net)):
            #     # nn.init.xavier_uniform_(net['encod layer '+str(i+1)].weight, gain=gain).clamp(0,1) 
            #     print(layer.name)
            #     nn.init.xavier_uniform_(layer.name, gain=gain).clamp(0,1) 
        # def reset_discriminator_parameters(net):
        #     gain = nn.init.calculate_gain('relu')
        #     for i in range(len(net)):
        #         nn.init.xavier_uniform_(net['layer '+str(i+1)].weight, gain=gain).clamp(0,1) 
            
        super(controversialnet2, self).__init__()
        self.in_feats1 = in_feats1
        self.hidden_feats1 = hidden_feats1
        self.hidden_feats2 = hidden_feats2
        self.bias = bias
        
        # self.path_forces = path_forces
        self.encoder_layers = nn.Sequential()
        self.discriminator_layers = nn.Sequential()
        
        
        
        for i in range(len(hidden_feats1)):
            if i == 0:
                self.encoder_layers.add_module('encod layer '+str(i+1),nn.Linear(in_feats1, 
                                                                    hidden_feats1[0], 
                                                                    bias=bias
                                                                   ).to(device, dtype)
                                              )
                self.encoder_layers.add_module('encod activation '+str(i+1), nn.Tanh().to(device, dtype))
                self.encoder_layers.add_module('dropping '+str(i+1), nn.Dropout(.25))
            elif 0 < i < len(hidden_feats1)-1:
                self.encoder_layers.add_module('encod layer '+str(i+1), nn.Linear(hidden_feats1[i-1], 
                                                                    hidden_feats1[i], 
                                                                    bias=bias).to(device, dtype)
                                              )
                self.encoder_layers.add_module('encod activation '+str(i+1), nn.Tanh().to(device, dtype))
                self.encoder_layers.add_module('dropping '+str(i+1), nn.Dropout(.25))
            else:
                self.encoder_layers.add_module('encod layer '+str(i+1), nn.Linear(hidden_feats1[i-1], 
                                                                    hidden_feats1[i], 
                                                                    bias=bias).to(device, dtype)
                                              )
                
        for i in range(len(hidden_feats2)):
            if i == 0:
                layer = nn.Linear(hidden_feats1[-1], hidden_feats2[i], bias=bias).to(device, dtype)
                self.discriminator_layers.add_module('layer '+str(i+1), layer)
                self.discriminator_layers.add_module('activation '+str(i+1), nn.ReLU().to(device, dtype))
                self.discriminator_layers.add_module('dropping '+str(i+1), nn.Dropout(.25))
            elif 0 < i < len(hidden_feats2)-1:
                layer = nn.Linear(hidden_feats2[i-1], hidden_feats2[i], bias=bias).to(device, dtype)
                self.discriminator_layers.add_module('layer '+str(i+1), layer)
                self.discriminator_layers.add_module('activation '+str(i+1), nn.ReLU().to(device, dtype))
                self.discriminator_layers.add_module('dropping '+str(i+1), nn.Dropout(.5))
            else:
                layer = nn.Linear(hidden_feats2[i-1], hidden_feats2[i], bias=False).to(device, dtype)
                self.discriminator_layers.add_module('layer '+str(i+1), layer)
                self.discriminator_layers.add_module('out '+str(i+1), nn.Sigmoid().to(device, dtype))
                
        reset_net_parameters(self.encoder_layers)
        reset_net_parameters(self.discriminator_layers)
        # reset_discriminator_parameters(self.discriminator_layers)
        
        
            
    
    
    def forward(
        self, 
        node_forces   # node features matrix
    ):
        new_features = self.encoder_layers(node_forces.to(device, dtype))
        
        # print(Fg.shape)
        output = self.discriminator_layers(new_features)
        
        return new_features, output

class controversialnet(nn.Module):
    def __init__(
        self, 
        in_feats1, 
        in_feats2,
        hidden_feats1,
        hidden_feats2,
        path_forces,
        memberships,
        members,
        bias = True
    ):
        super(controversialnet, self).__init__()
        self.in_feats1 = in_feats1
        self.in_feats2 = in_feats2
        self.hidden_feats1 = hidden_feats1
        self.hidden_feats2 = hidden_feats2
        self.path_forces = path_forces
        self.memberships = memberships
        self.members = members
        self.bias = bias
        # self.path_forces = path_forces
        self.encoder_layers = nn.Sequential()
        self.discriminator_layers = nn.Sequential()
        
        for i in range(len(hidden_feats1)):
            if i == 0:
                self.encoder_layers.add_module('encod layer '+str(i+1),nn.Linear(in_feats1, 
                                                                    hidden_feats1[0], 
                                                                    bias=bias
                                                                   ).to(device, dtype)
                                              )
                self.encoder_layers.add_module('encod activation '+str(i+1), nn.Tanh().to(device, dtype))
                self.encoder_layers.add_module('dropping '+str(i+1), nn.Dropout(.25))
            elif 0 < i < len(hidden_feats1)-1:
                self.encoder_layers.add_module('encod layer '+str(i+1), nn.Linear(hidden_feats1[i-1], 
                                                                    hidden_feats1[i], 
                                                                    bias=bias).to(device, dtype)
                                              )
                self.encoder_layers.add_module('encod activation '+str(i+1), nn.Tanh().to(device, dtype))
                self.encoder_layers.add_module('dropping '+str(i+1), nn.Dropout(.25))
            else:
                self.encoder_layers.add_module('encod layer '+str(i+1), nn.Linear(hidden_feats1[i-1], 
                                                                    hidden_feats1[i], 
                                                                    bias=bias).to(device, dtype)
                                              )
                
        for i in range(len(hidden_feats2)):
            if i == 0:
                layer = nn.Linear(in_feats2, hidden_feats2[i], bias=bias).to(device, dtype)
                self.discriminator_layers.add_module('layer '+str(i+1), layer)
                self.discriminator_layers.add_module('activation '+str(i+1), nn.ReLU().to(device, dtype))
                self.discriminator_layers.add_module('dropping '+str(i+1), nn.Dropout(.25))
            elif 0 < i < len(hidden_feats2)-1:
                layer = nn.Linear(hidden_feats2[i-1], hidden_feats2[i], bias=bias).to(device, dtype)
                self.discriminator_layers.add_module('layer '+str(i+1), layer)
                self.discriminator_layers.add_module('activation '+str(i+1), nn.ReLU().to(device, dtype))
                self.discriminator_layers.add_module('dropping '+str(i+1), nn.Dropout(.5))
            else:
                layer = nn.Linear(hidden_feats2[i-1], hidden_feats2[i], bias=False).to(device, dtype)
                self.discriminator_layers.add_module('layer '+str(i+1), layer)
                self.discriminator_layers.add_module('out '+str(i+1), nn.Softmax().to(device, dtype))
            
    
    
    def forward(
        self, 
        F_,   # node2node exerted matrix forces
        X    # node features matrix
    ):
        input_ = F_.to(device, dtype)@X.to(device, dtype)
        new_features = self.encoder_layers(input_)
        
        new_force = get_node2node_forces(new_features,self.path_forces)
        si = get_silhouette2(new_features, self.memberships).to(device, dtype)
        Fg = new_force @ self.members
        # print(Fg.shape)
        output = self.discriminator_layers(Fg.reshape(1,-1))
        
        return new_features, new_force, output, si
        
class ControEncodNet(nn.Module):
    
    def __init__(
        self, 
        in_feats, 
        hidden_feats,
        path_forces,
        memberships,
        members,
        bias = True
    ):
        super(ControEncodNet, self).__init__()
        self.in_feats = in_feats
        self.bias = bias
        self.hidden_feats = hidden_feats
        self.path_forces = path_forces
        self.memberships = memberships
        self.members = members
        # self.controverse_layers = nn.ModuleList()
        self.controverse_layers = nn.Sequential()

        
        for i in range(len(hidden_feats)):
            if i == 0:
                self.controverse_layers.add_module('layer '+str(i+1), nn.Linear(in_feats, 
                                                                    hidden_feats[0], 
                                                                    bias=bias
                                                                   ).to(device, dtype)
                                              )
                self.controverse_layers.add_module('activation '+str(i+1), nn.Tanh().to(device, dtype))
                self.controverse_layers.add_module('dropping '+str(i+1), nn.Dropout(.25))
            elif 0 < i < len(hidden_feats)-1:
                self.controverse_layers.add_module('layer '+str(i+1), nn.Linear(hidden_feats[i-1], 
                                                                    hidden_feats[i], 
                                                                    bias=bias).to(device, dtype)
                                              )
                self.controverse_layers.add_module('activation '+str(i+1), nn.Tanh().to(device, dtype))
                self.controverse_layers.add_module('dropping '+str(i+1), nn.Dropout(.5))
            else:
                self.controverse_layers.add_module('activation '+str(i+1), nn.Linear(hidden_feats[i-1], 
                                                                    hidden_feats[i], 
                                                                    bias=bias).to(device, dtype)
                                              )
                self.controverse_layers.add_module('output '+str(i+1), nn.Sigmoid().to(device, dtype))
    
    
    
    def forward(self, F_, X):
        input_ = F_.to(device, dtype)@X.to(device, dtype)
        new_features = self.controverse_layers(input_)
        new_force = get_node2node_forces(new_features,self.path_forces)
        si = get_silhouette2(new_features, self.memberships)
        return new_features, si,new_force.to(device, dtype)@self.members.to(device, dtype)

class ControDiscriNet(nn.Module):
    def __init__(
            self,
            in_feats,
            hidden_feats,
            bias = True
            ):
        
        super(ControDiscriNet, self).__init__()
        self.in_feats = in_feats
        self.hidden_feats = hidden_feats
        self.bias = bias 
        # self.discriminant_layers = nn.ModuleList()
        self.discriminant_layers = nn.Sequential()
        
        layer = None
        for i in range(len(hidden_feats)):
            if i == 0:
                layer = nn.Linear(in_feats, hidden_feats[i], bias=bias).to(device, dtype)
                self.discriminant_layers.add_module('layer '+str(i+1), layer)
                self.discriminant_layers.add_module('activation '+str(i+1), nn.ReLU().to(device, dtype))
                self.discriminant_layers.add_module('dropping '+str(i+1), nn.Dropout(.25))
            elif 0 < i < len(hidden_feats)-1:
                layer = nn.Linear(hidden_feats[i-1], hidden_feats[i], bias=bias).to(device, dtype)
                self.discriminant_layers.add_module('layer '+str(i+1), layer)
                self.discriminant_layers.add_module('activation '+str(i+1), nn.ReLU().to(device, dtype))
                self.discriminant_layers.add_module('dropping '+str(i+1), nn.Dropout(.5))
            else:
                layer = nn.Linear(hidden_feats[i-1], hidden_feats[i], bias=False).to(device, dtype)
                self.discriminant_layers.add_module('layer '+str(i+1), layer)
            
            # self.discriminant_layers.append(layer)
            
    def forward(
            self, 
            force
            ):
        
        output = self.discriminant_layers(force.reshape(1,-1))
                
        return output
    
    
    
    
    
import torch.nn as nn
from torch.autograd import Function


class WeightBinarizerFunction(Function):
    @staticmethod
    def forward(ctx, x):
        # ctx is a context object that can be used to stash information
        # for backward computation
        return x.sign() * x.abs().mean()

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        return grad_output


class WeightBinarizer(nn.Module):
    def forward(self, input):
        return WeightBinarizerFunction.apply(input)