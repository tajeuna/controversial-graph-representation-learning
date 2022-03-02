#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 04:38:36 2022

@author: etienne
"""


from loading import import_data
from geometrical_properties import graph_properties as gp
from controversial_net import * #ControEncodNet, ControDiscriNet, ControGraphConvLayer, controversialnet, controversialnet2
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

class CNN2(nn.Module):
        def __init__(self,              
                     in_feats,
                     h_feats,
                     p2p,
                     memb,
                     bias=True,
                     dropout=.3
                     ):
            
            """
    
            :param in_feats: <int> Dimensionality of node input features
            :param h_feats: <list> Dimensionality of hidden layers
            :param num_classes: <int> Number of output classes
            :param dropout: <float> The amount of dropout for all but the last
                layer. It should be a value in [0.0, 1.0]
            """
            super(CNN2, self).__init__()
    
            self.dropout = dropout
            self.cnn_layers = nn.ModuleList()
            self.memb = memb
    
            self.cnn_layers.append(ControGraphConvLayer(in_feats, h_feats[0],p2p,bias))
    
            for i in range(1, len(h_feats)):
                self.cnn_layers.append(ControGraphConvLayer(h_feats[i-1], h_feats[i],p2p,bias))
    
            # self.cnn_layers.append(nn.Linear(h_feats[-1], num_classes).to(device, dtype))
    
        def forward(self, F_, X):
    
            forc,h = F_, X
            for i, layer in enumerate(self.cnn_layers):
                # if i < len(self.cnn_layers)-1:
                forc,h = layer(forc, h)
                h = F.dropout(F.relu(h), p=self.dropout)
                forc = F.dropout(F.relu(forc), p=self.dropout)
            
                    
    
            return forc, F.softmax(forc@self.memb)
        
def train2(g, model,F_, fr, members, epochs=200, lr=0.01, weight_decay=0.0005):
    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    losses = {"train": [], "silhouette": []}
    # accs = {"train": [], "test": []}
    
    best_test_acc = 0

    features = g.ndata['features']
    labels = g.ndata['labels']
    train_mask = g.ndata['train_mask']
    test_mask = g.ndata['test_mask']
    for e in range(epochs):
        model.train()
        # Forward
        forc1,forcg = model(F_, features)
       
        loss = F.binary_cross_entropy_with_logits(forcg, members)
        losses["train"].append(loss.item())
        losses["silhouette"].append(get_silhouette(forc1,fr))
        sil = get_silhouette(forc1,fr)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # model.eval()
        # # Compute accuracy on training/validation/test
        # logits = model(F_, features)

        # # Compute prediction
        # pred = logits.argmax(1)
        
        # train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        # test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # accs["train"].append(train_acc)
        # accs["test"].append(test_acc)
        
        # # Save the best validation accuracy and the corresponding test accuracy.
        # if best_test_acc < test_acc:
        #     best_test_acc = test_acc
        
        
        if e % 10 == 0:
            print('In epoch {}, loss: {:.3f}, silh: {:.3f}'.format(e, loss, sil.item()))
            
    return losses
    
    
class CNN(nn.Module):
        def __init__(self,              
                     in_feats,
                     h_feats,
                     p2p,
                     num_classes,
                     bias=True,
                     dropout=0
                     ):
            
            """
    
            :param in_feats: <int> Dimensionality of node input features
            :param h_feats: <list> Dimensionality of hidden layers
            :param num_classes: <int> Number of output classes
            :param dropout: <float> The amount of dropout for all but the last
                layer. It should be a value in [0.0, 1.0]
            """
            super(CNN, self).__init__()
    
            self.dropout = dropout
            self.cnn_layers = nn.ModuleList()
    
            self.cnn_layers.append(ControGraphConvLayer(in_feats, h_feats[0],p2p,bias))
    
            for i in range(1, len(h_feats)):
                self.cnn_layers.append(ControGraphConvLayer(h_feats[i-1], h_feats[i],p2p,bias))
    
            self.cnn_layers.append(nn.Linear(h_feats[-1], num_classes).to(device, dtype))
    
        def forward(self, F_, X):
    
            forc,h = F_, X
            for i, layer in enumerate(self.cnn_layers):
                if i < len(self.cnn_layers)-1:
                    forc,h = layer(forc, h)
                    h = F.dropout(F.relu(h), p=self.dropout)
                    forc = F.dropout(F.relu(forc), p=self.dropout)
                else:
                    h = layer(h)
                    
    
            return F.sigmoid(h)


def train(g, model,F_, members, epochs=200, lr=0.01, weight_decay=0.0005):
    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    losses = {"train": [], "test": []}
    accs = {"train": [], "test": []}
    
    best_test_acc = 0

    features = g.ndata['features']
    labels = g.ndata['labels']
    train_mask = g.ndata['train_mask']
    test_mask = g.ndata['test_mask']
    for e in range(epochs):
        model.train()
        # Forward
        logits = model(F_, features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that we only need the loss over the nodes in the training set for
        # updating the model parameters but we compute it for the validation and test nodes
        # for reporting.
        loss = F.binary_cross_entropy_with_logits(logits[train_mask], members[train_mask])
        losses["train"].append(loss.item())
        losses["test"].append(F.binary_cross_entropy_with_logits(logits[test_mask], members[test_mask]).item())
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        # Compute accuracy on training/validation/test
        logits = model(F_, features)

        # Compute prediction
        pred = logits.argmax(1)
        
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        accs["train"].append(train_acc)
        accs["test"].append(test_acc)
        
        # Save the best validation accuracy and the corresponding test accuracy.
        if best_test_acc < test_acc:
            best_test_acc = test_acc
        
        
        if e % 10 == 0:
            print('In epoch {}, loss: {:.3f}, test acc: {:.3f} (best {:.3f})'.format(
                e, loss, test_acc, best_test_acc))
            
    return losses, accs

# cuda_device = 0
# device = th.device("cuda:%d" % cuda_device if th.cuda.is_available() else "cpu")
device = 'cpu'
dtype = th.float64


    


# gml_path_file = '/home/etienne/CGRL/Data/football/football.gml'

# gml_path_file = '/home/etienne/CGRL/Data/lesmis/lesmis.gml'

gml_path_file = '/home/etienne/CGRL/Data/dolphins/dolphins.gml'

# gml_path_file = '/home/etienne/CGRL/Data/as-22july06/as-22july06.gml'

# gml_path_file = '/home/etienne/CGRL/Data/adjnoun/adjnoun.gml'

###############################################################################
############### omics data ####################################################
###############################################################################

path_to_metadata = '/home/etienne/Bio-info/metadata.csv'
path_to_metabolomic = '/home/etienne/Bio-info/metabolomics.csv'
path_to_proteomic = '/home/etienne/Bio-info/proteomics.csv'
path_to_proteomics_cyt = '/home/etienne/Bio-info/proteomics_cyt.csv'
edge_type = 'metabolomics'
attribute_type = 'proteomics'
distance = 'SAX-Matching'

###############################################################################
###############################################################################

generate_features = True
size_feature_vec = 10
size_batch = 75
generate_weights = True


# manualSeed = 7270
manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)

method = 'Finch'


if __name__ == '__main__':
    
    '''
        Import data to be used ...
    '''
    dgl_g, dgl_node = import_data(
        gml_path_file = gml_path_file,
        generate_features = generate_features,
        size_feature_vec = size_feature_vec,
        generate_weights = generate_weights,
        manualSeed = manualSeed
        ).gml_data_to_network()
    
    # dgl_g, dgl_node = import_data(generate_features=generate_features).nice_graph()
    # dgl_g, dgl_node = import_data(generate_features=generate_features).karate_club()
    # print(dgl_node)
    # dgl_g, dgl_node = import_data(
    #    gml_path_file = None, 
    #    generate_features = generate_features,
    #    size_feature_vec = size_feature_vec,
    #    generate_weights = generate_weights,
    #    manualSeed = manualSeed
    #    ).karate_club()
    
    
    
    # dgl_g,nx_graph = import_data(
    #         manualSeed=manualSeed,
    #         components = 6
    #     ).covid_long_data(
    #         path_to_metadata=path_to_metadata,
    #         path_to_metabolomic=path_to_metabolomic,
    #         path_to_proteomic=path_to_proteomic,
    #         path_to_proteomics_cyt=path_to_proteomics_cyt,
    #         threshold=None, 
    #         edge_type=edge_type,
    #         attribute_type=attribute_type, 
    #         distance=distance
    # )
    
    
    foot_properties = gp(
        dgl_graph = dgl_g, 
        method=method, 
        manualSeed=manualSeed
        )
    
    print(' Searching for nodes shortest paths...')
    path_scores = foot_properties.node2node_path_attraction()
    
    print(' Calculate the pairwise node similarities...')
    sim_scores = foot_properties.node2node_similarity_attraction()
    
    print(' Calculate the pairwise node force attraction')
    n2n_force = foot_properties.node2node_force_attraction2(sim_scores, path_scores)  
    
    print(' Getting node memberships')
    fr, nx_graph, members = foot_properties.label_nodes_graph()
    print(nx.info(nx_graph))
    
    print('# substructures based on '+method+' clustering ...'+str(len(list(fr))))
    g2n_force = foot_properties.group2node_force_attraction(members, n2n_force)
    
    visualize = plot_features(nxg=nx_graph).plot_graph()
    samples = foot_properties.cross_validation_sample_graphs(10, 80, fr)
 
    # in_feats,
    #                  h_feats,
    #                  p2p,
    #                  num_classes,
    #                  bias=True,
    #                  dropout=0
    
    # model = CNN(in_feats=dgl_g.ndata['features'].shape[1], h_feats=[32, 2],p2p=path_scores, num_classes=2)

    
    # visualize = plot_features(embedding = g2n_force, nxg=g, view='TSNE', manualSeed=manualSeed).plot()
    
    adj = th.zeros(dgl_g.num_nodes(), dgl_g.num_nodes())
    for i,j in zip(dgl_g.edges()[0], dgl_g.edges()[1]): 
        adj[i.item(),j.item()] = 1
        adj[i,i] = 1
    adj = adj.to(device, dtype)
    norm = th.diag(adj.sum(axis=0)**(-.5))
    force_net = controversialnet2(184, [50, 20], [len(list(fr))])
    # force_net.train(True)
    optimizer = th.optim.SGD(force_net.parameters(), lr=.01, momentum=.3)
    error = []
    expected_forces = members.mul(th.rand_like(members))
    batch_size = int(dgl_g.num_nodes()/10)
    print(batch_size)
    touse = norm@(n2n_force.to(device, dtype) @ dgl_g.ndata['features'].to(device,dtype))
    
    with th.enable_grad():
        
        for e in range(100):
            force_net.train(True)
            embedding_ = th.zeros(dgl_g.num_nodes(), 20)
            for n in range(0,dgl_g.num_nodes(),batch_size):
                optimizer.zero_grad()
                new_features, output = force_net(touse[n:n+batch_size,:])
                cum_loss = F.binary_cross_entropy_with_logits(output, expected_forces[n:n+batch_size, :])
                # criterion = Loss(weight = [.99, .01])
                
                cum_loss.backward(retain_graph=False)
                optimizer.step()
                embedding_[n:n+batch_size,:] = new_features.detach()
                error.append(cum_loss)
            sil_loss = controversialencoderloss(Variable(embedding_.to(device,dtype), requires_grad=True), fr)
            sil_loss.backward()
            if e % 5 == 0: 
                print('Epoch {}, Loss: {:.3f}, silo: {:.3f}'.format( e, error[e].item(), sil_loss.item()))
                
   
    
    h,_ = force_net(touse)
    
    visualize = plot_features(embedding = h, nxg=g, view='TSNE', manualSeed=manualSeed).plot()
        
        
        
    # expected_proba, expected_silhouette = [], []
    
    # for val in th.rand(15):
    #     if val < .5:
    #         expected_silhouette.append(val + .5)
    #     else: expected_silhouette.append(val)
    #     if val+.3 < .7:
    #         expected_proba.append(val+.3)
    #     else: expected_proba.append(val)
    
  
    # space_layer_search1 = [[size_feature_vec*2, 
    #                         size_feature_vec, 
    #                         int(size_feature_vec/2),
    #                         int(size_feature_vec/3),
    #                         int(np.sqrt(size_feature_vec))], 
    #                        [size_feature_vec, 
    #                         int(size_feature_vec/2),
    #                         int(size_feature_vec/3),
    #                         int(np.sqrt(size_feature_vec))], 
    #                        [int(size_feature_vec/2),
    #                         int(size_feature_vec/3),
    #                         int(np.sqrt(size_feature_vec))], 
    #                        [int(size_feature_vec/3),
    #                         int(np.sqrt(size_feature_vec))], 
    #                        [int(np.sqrt(size_feature_vec))]
    #                        ]
    # space_layer_search2 = [[members.shape[0]*members.shape[1]*2, 
    #                         members.shape[0]*members.shape[1], 
    #                         int(members.shape[0]*members.shape[1]/2),
    #                         int(members.shape[0]*members.shape[1]/3),
    #                         int(np.sqrt(members.shape[0]*members.shape[1])),
    #                         1], 
    #                        [members.shape[0]*members.shape[1], 
    #                         int(members.shape[0]*members.shape[1]/2),
    #                         int(members.shape[0]*members.shape[1]/3),
    #                         int(np.sqrt(members.shape[0]*members.shape[1])),
    #                         1], 
    #                        [int(members.shape[0]*members.shape[1]/2),
    #                         int(members.shape[0]*members.shape[1]/3),
    #                         int(np.sqrt(members.shape[0]*members.shape[1])),
    #                         1], 
    #                        [int(members.shape[0]*members.shape[1]/3),
    #                         int(np.sqrt(members.shape[0]*members.shape[1])),
    #                         1], 
    #                        [int(np.sqrt(members.shape[0]*members.shape[1])),
    #                         1]
    #                        ]
    # lr_space = [.005, .002, .001, .05, .02, .01]
    # trainer = pytorch_train2
    # size_vec = size_feature_vec
    # n2n_path = path_scores
    # memberships = fr
    # membs = members
    # graph_ = dgl_g
    # n_epoch = 5
    # sils = expected_silhouette
    # pros = expected_proba
    
    # dico = grid_search_hyperparameters(
    #         space_layer_search1, 
    #         space_layer_search2, 
    #         lr_space,
    #         trainer,
    #         size_vec,
    #         n2n_path,
    #         memberships,
    #         membs,
    #         graph_,
    #         n_epoch,
    #         sils,
    #         pros,
    #         bias = False
    #         )

    # contro_model_lay = ControEncodNet(size_feature_vec,[10, 2],path_scores,fr,members).to(device, dtype)
    # contro_model_lay = ControGraphConvLayer(size_feature_vec, 2)
    # contro_model_lay.apply(init_weights)
    # discri_model_lay = ControDiscriNet(members.shape[0]*members.shape[1], [5, 1]).to(device, dtype)
    
    # print(contro_model_lay)
    # print(discri_model_lay)
    # controversial_model = ControEncodNet(
    #     size_feature_vec, 
    #     [8, 5, 2],
    #     path_scores,
    #     fr,
    #     members,
    #     bias = True
    #     )
    
    # controversial_model = controversialnet(
    #     dgl_g.ndata['features'].shape[1],
    #     members.shape[0]*members.shape[1],
    #     [8, 2],
    #     [10, 1],
    #     path_scores,
    #     fr,
    #     members,
    #     bias = True
    #     )
    # controversial_model.apply(init_weights)
    # print(controversial_model)
    # new_features, new_force, output, si = controversial_model(n2n_force,dgl_g.ndata['features'])
    # visualize = plot_features(embedding = new_features, nxg=g, view='TSNE', manualSeed=manualSeed).plot()
    
    # optimizer = th.optim.SGD(controversial_model.parameters(), lr=.01, momentum=.3)
    # criterion = Loss(weight=[.99, .01])
    # criterion = nn.MSELoss().to(device, dtype)
    
            
    
    # losses = pytorch_train2(n2n_force, dgl_g.ndata['features'], controversial_model, optimizer,expected_silhouette, expected_proba,100)
    # new_features, new_force, output, si = controversial_model(n2n_force,dgl_g.ndata['features'])
    # losses = pytorch_train(n2n_force, g_foot.ndata['features'], path_scores, members, fr,
    #                 controversial_model, optimizer,20,criterion,size_batch)
    # new_features, new_force, output, si = controversial_model(members, fr, path_scores,n2n_force,g_foot.ndata['features'])
    # new_features, si, new_force = controversial_model(n2n_force,g_foot.ndata['features'])

    
    # controversial_model.apply(init_weights)
    # # controversial_training(n2n_force, g_foot.ndata['features'], path_scores, fr, members, contro_model_lay, discri_model_lay, epochs=10, lr=0.1)
    # true_samples = [scale_matrice(th.randn_like(g2n_force)).mul(members).to(device,dtype) for i in range(20)]
    # criterion = nn.BCEWithLogitsLoss().to(device, dtype)
    # train_neural_net(n2n_force, g_foot.ndata['features'], path_scores, fr, members, 
    #                           contro_model_lay, discri_model_lay, true_samples, criterion, controversialencoderloss, epochs=10, lr=0.002)
    # errorD, errorE = train_neural_net(g_foot.ndata['features'], path_scores, fr, members, 
    #                           contro_model_lay, discri_model_lay, true_samples, epochs=50, lr=0.002)
    
    # embed = scale_matrice(new_features)
    
    # new_forces = get_new_forces(path_scores, embed, members)
    
    
    
    # checking = discri_model_lay(new_forces)
    
    # loss = ControLoss(embed, new_forces, fr)
    
    # silhouet = loss.encodloss()
    
    # visualize = plot_features(embedding = h, nxg=g, view='PCA', manualSeed=manualSeed, graph=nx_graph).plot_graph()
