#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 16:35:26 2022

@author: etienne
"""


from loading import import_data
from geometrical_properties import graph_properties as gp
from controversial_net import ControEncodNet, ControDiscriNet, ControGraphConvLayer, controversialnet, controversialnet2
from sklearn.metrics import silhouette_score as ss
from embedding_visualisation import plot_features
from torch.autograd import Variable
import networkx as nx
import torch as th
from torch import nn
import torch.nn.functional as F
import random
from poutyne import Model
from itertools import combinations
import numpy as np


cuda_device = 0
device = th.device("cuda:%d" % cuda_device if th.cuda.is_available() else "cpu")
# device = 'cpu'
dtype = th.float64

def scale_matrice(matrice):
    for i in range(matrice.shape[1]):
        mini = th.min(matrice[:, i])
        maxi = th.max(matrice[:, i])
        matrice[:, i] = (matrice[:, i] - mini)/(maxi-mini)
        # tmp_mat[:, i] = std * (maxi - mini) + mini
    return matrice

def get_sim_forces(f1, f2):
    f1 = f1.reshape(1,-1)
    f2 = f2.reshape(1,-1)
    return .5*(F.cosine_similarity(f1, f2)+1).squeeze()

def get_sim_forces_euc(f1, f2):
    f1 = f1.reshape(1,-1)
    f2 = f2.reshape(1,-1)
    return th.cdist(f1, f2)

def get_new_forces(path_attraction, new_features, members):
    matrix = th.zeros(new_features.shape[0], new_features.shape[0]).to(device, dtype)
    for i in range(new_features.shape[0]-1):
        f1 = new_features[i, :]
        for j in range(i+1, new_features.shape[0]):
            f2 = new_features[j, :]
            matrix[i, j] = get_sim_forces(f1, f2)
            matrix[j, i] = get_sim_forces(f2, f1)
    node2node_force = matrix.mul(path_attraction)
    
    return scale_matrice(th.matmul(node2node_force, members))

def grid_search_hyperparameters(
            space_layer_search1, 
            space_layer_search2, 
            lr_space,
            trainer,
            size_vec,
            n2n_path,
            memberships,
            membs,
            graph_,
            n_epoch,
            expected_sil,
            expected_pro,
            bias = False
            ):
        dico = {}
        cpt = 1
        for i, hidden_layers1 in enumerate(space_layer_search1):
            dico['encoder_layer '+str(i+1)] = {}
            
            for j, hidden_layers2 in enumerate(space_layer_search2):
                dico[' model '+str(cpt)] = {}
                dico[' model '+str(cpt)]['Encoder_layers '+str(i+1)] = hidden_layers1
                dico[' model '+str(cpt)]['Discriminator_layers '+str(j+1)] = hidden_layers2
                dico[' model '+str(cpt)]['Training'] = {}
                model = controversialnet(
                    size_vec, 
                    membs.shape[0]*membs.shape[1],
                    hidden_layers1,
                    hidden_layers2,
                    n2n_path,
                    memberships,
                    membs,
                    bias = bias
                    )
                model.apply(init_weights)
                print(' model '+str(cpt)+':\n', model)
                
                for lr in lr_space:
                    dico[' model '+str(cpt)]['Training']['lr'] = lr
                    optimizer = th.optim.SGD(model.parameters(), lr=lr)
                    print('training model '+str(cpt)+' with a learning rate = '+str(lr))
                    dico[' model '+str(cpt)]['Training']['losses'] = trainer(n2n_force, 
                                   graph_.ndata['features'], 
                                   model, 
                                   optimizer,
                                   expected_sil, 
                                   expected_pro,
                                   n_epoch)
                cpt += 1
        return dico

def init_weights(m):
    if isinstance(m, nn.Linear):
        th.nn.init.xavier_uniform(m.weight)
        try:
            m.bias.data.fill_(0.01)
        except: pass
        
def get_batches(M, fr, P, F_, X, size_batch):
    sub_M = [M[i:i+size_batch, ] for i in range(0, len(fr)-size_batch, size_batch)]
    sub_fr = [fr.loc[i:i+size_batch-1] for i in range(0, len(fr)-size_batch, size_batch)]
    sub_P = [P[i:i+size_batch, i:i+size_batch] for i in range(0, len(fr)-size_batch, size_batch)]
    sub_F = [F_[i:i+size_batch, i:i+size_batch] for i in range(0, len(fr)-size_batch, size_batch)]
    sub_X = [X[i:i+size_batch, ] for i in range(0, len(fr)-size_batch, size_batch)]
    return sub_M, sub_fr, sub_P, sub_F, sub_X
    
def pytorch_train2(F_, X, pytorch_network, optimizer, sils, pros, nepoch, criterion):
    early_stopping = False
    pytorch_network.train(True)
    # th.autograd.set_detect_anomaly(True)
    entier = int(nepoch*.01)
    # print(entier)
    with th.enable_grad():
        error = []
        sil_story = []
        for e in range(nepoch):
            for i in range(len(sils)):
                optimizer.zero_grad()
                new_features, new_force, output, si = pytorch_network(F_,X)
                sil_story.append(si)
                if si.item()>.5: early_stopping = True
                # criterion = Loss(weight = [.99, .01])
                # print(output)
                si, output = Variable(si, requires_grad=True), Variable(output, requires_grad=True)
                # print(i, len(sils), len(pros))
                encod_error = criterion(si, sils[i].to(device, dtype),
                                        output, pros[i].to(device, dtype))
                # if len(error)>2: 
                #     if si < min(sil_story):
                #         encod_error = error[sil_story.index(min(sil_story))] - error[sil_story.index(min(sil_story))]*.001
                    # if encod_error > min(error):
                        # encod_error = min(error) - min(error)*.001
                encod_error.backward(retain_graph=False)
                optimizer.step()
                error.append(encod_error)
                
                if early_stopping: 
                    break
            # if e % entier == 0:
            print('In epoch {}| contro loss {:.3f}, silhouette {:.3f}'.format(e, encod_error.item(),si.item()))
            
            if early_stopping: 
                break
                print('Early stopping with an silhouette score of ', si.item())
                break
                
                
                # if si.item() > .25: break
            
            # error.append(loss_sum/len(batch_M))
    return error

def pytorch_train(F_, X, P, M, fr, pytorch_network, optimizer,nepoch,criterion, size_batch):
    batch_M, batch_fr, batch_P, batch_F, batch_X = get_batches(M, fr, P, F_, X, size_batch)
    for m,f,p,ff,x in zip(batch_M, batch_fr, batch_P, batch_F, batch_X):
        print(len(m), len(f), len(p), len(ff), len(x))
    pytorch_network.train(True)
    with th.enable_grad():
        error = []
        for e in range(nepoch):
            loss_sum = 0
            for i in range(len(batch_M)):
                M, fr, P,F_,X = batch_M[i], batch_fr[i], batch_P[i], batch_F[i], batch_X[i]
                optimizer.zero_grad()
                new_features, new_force, output,si = pytorch_network(M, fr, P,F_,X)
                # si = Variable(th.tensor(get_silhouette(new_force, fr)).to(device, dtype), requires_grad=True)
                # print(si)
                discrim_error = criterion(output, th.ones(1,1).to(device, dtype))
                encod_error = criterion(si, th.ones(1,1).to(device, dtype))
                # loss = controversialencoderloss(new_features,fr)
                discrim_error.backward()
                encod_error.backward()
                optimizer.step()
                loss_sum += discrim_error + encod_error
            if e % 2 == 0:
                print('In epoch {}| encod loss {:.3f}, discr loss {:.3f}, silhou {:.3f}'.format(e, encod_error.item(), discrim_error.item(),si.item()))
                
                # if si.item() > .25: break
            
            error.append(loss_sum/len(batch_M))
    return loss_sum

def get_silhouette(force,fr):
    tmp = force.cpu().detach().numpy()
    labels = np.zeros(len(fr),int)
    for c in list(fr):
        vec = fr[c].values
        for i in np.where(vec==1): labels[i] = int(c.replace('C',''))
    return ss(tmp, metric='precomputed', labels=labels)

def get_silhouette2(features,fr):
    tmp = features.cpu().detach().numpy()
    labels = np.zeros(len(fr),int)
    for c in list(fr):
        vec = fr[c].values
        for i in np.where(vec==1): labels[i] = int(c.replace('C',''))
    return ss(tmp, metric='cosine',labels=labels)
  


def train_neural_net(F_, X, path_attraction, memberships, members, model_encod, model_discrim, true_samples, criterion1,criterion2, epochs=50, lr=0.01):

#     optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_encod = th.optim.SGD(model_encod.parameters(), lr=lr, momentum=0.19)
    optimizer_discrim = th.optim.SGD(model_discrim.parameters(), lr=lr, momentum=0.19)
    
    errorD = []
    errorE = []
    model_discrim.train(True)
    early_stopping = False
    with th.enable_grad():
        for e in range(epochs):
            for i, sample in enumerate(true_samples):
                
                optimizer_discrim.zero_grad()
                
                # Train the model to know real forces
                true_proba = model_discrim(sample)
                error_true = criterion1(true_proba.to(device, dtype), th.ones(1,1).to(device, dtype))
                error_true.backward()
                
                # calculate fake forces 
                
                fake_n2n_force = scale_matrice(th.randn(sample.shape[0], sample.shape[0]))
                Y,si,fake_force = model_encod(fake_n2n_force, X)
                # print('  ', get_silhouette2(Y, memberships))
                # fake_force = get_new_forces(path_attraction, Y, members)
                fake_proba = model_discrim(fake_force.detach())
                error_false = criterion1(fake_proba, th.zeros(1,1).to(device, dtype))
                
                # error_false = controversialdiscriminatorloss(fake_proba, th.zeros(1,1).to(device, dtype))
                error_false.backward()
                
                error_disc = error_true + error_false
                # print(error_true, error_false, error_disc)
                optimizer_discrim.step()
                
                optimizer_encod.zero_grad()
                
                Y2,si2,fake_force2 = model_encod(F_, X)
                error_sil = criterion2(Y2, fr)
                error_sil.backward()
                optimizer_encod.step()
                
                final_loss = .75*error_sil + .25*error_disc
                entier = int(epochs*.05)
                if e % 5 == 0:
                    # print(e, error_true.item())
                    print('In epoch {}| loss true {:.3f}, loss fake {:.3f}, whole disc {:.3f}, error_sil {:.3f}'.format(e, error_true.item(), error_false.item(), error_disc.item(),error_sil.item()))
                    
                # errorD.append(error_disc)
                # errorE.append(error_encod)
        
            
    # return errorD, errorE