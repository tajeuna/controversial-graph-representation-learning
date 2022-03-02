#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 04:55:18 2022

@author: etienne
"""

import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch as th
import igraph as ig
import networkx as nx
import seaborn as sns
from itertools import combinations
from scipy.stats import wasserstein_distance as wd
from sklearn.mixture import GaussianMixture as GM
import math
from finch import FINCH
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")



cuda_device = 0
# device = th.device("cuda:%d" % cuda_device if th.cuda.is_available() else "cpu")
device = 'cpu'
dtype = th.float64

class graph_properties:
    
    def __init__(
            self, 
            dgl_graph, 
            method = 'Markov',
            manualSeed = None,
            ):
        
        self.dgl_graph = dgl_graph
        self.method = method
        self.manualSeed = manualSeed
        
    def cast_feature_to_normal_distribution(self, feature):
        X = feature.reshape(len(feature), 1)
        model = GM(n_components=1, random_state=self.manualSeed).fit(X)
        loc = model.means_.squeeze().item()
        scale = model.covariances_.squeeze().item()
            
        return norm.pdf(X, loc=loc, scale=scale)
    
    def scale_matrice(self, matrice):
        tmp_mat = th.zeros_like(matrice).to(device, dtype)
        for i in range(matrice.shape[1]):
            mini = th.min(matrice[:, i])
            maxi = th.max(matrice[:, i])
            tmp_mat[:, i] = (matrice[:, i] - mini)/(maxi-mini)
            # tmp_mat[:, i] = std * (maxi - mini) + mini
        return tmp_mat
    
    def add_label_to_graph(self,frame):
        tmp_g = self.dgl_graph.clone()
        tmp_g.ndata['labels'] = th.zeros(tmp_g.num_nodes())
        for h in list(frame):
            vec = frame[h].values.tolist()
            for n in range(tmp_g.num_nodes()):
                if vec[n] == 1:
                    tmp_g.ndata['labels'][n] = int(h.replace('C',''))
        return tmp_g
    
    def cross_validation_sample_graphs(self, cv, percentage, frame):
        samples = []
        
        for k in range(cv):
            sample = self.add_label_to_graph(frame)
            sample.ndata['train_mask'] = th.tensor([False for l in range(len(frame))])
            sample.ndata['test_mask'] = th.tensor([False for l in range(len(frame))])
            for h in list(frame):
                subframe = frame[frame[h]==1]
                train_size = int(len(subframe)*percentage*.01)
                train_samp = subframe.sample(n=train_size)
                test_samp = subframe.loc[[ind for ind in subframe.index if ind not in train_samp.index], :]
                for ind in train_samp.index:
                    sample.ndata['train_mask'][ind] = th.tensor(True)
                for ind in test_samp.index:
                    sample.ndata['test_mask'][ind] = th.tensor(True)
                
            samples.append(sample)
        return samples
            
                
        
        
        
        
    def dgl_to_networkx(self):
        # print(self.dgl_graph)
        g_nx = nx.Graph()
        nodes = list(self.dgl_graph.nodes().cpu().detach().numpy())
        edges = [(edge1.item(), edge2.item()) for edge1, edge2 in zip(self.dgl_graph.edges()[0], self.dgl_graph.edges()[1])]
        features = self.dgl_graph.ndata['features'].cpu().detach().numpy()
        weights = self.dgl_graph.edata['weight'].cpu().detach().numpy()

        g_nx.add_nodes_from(nodes)
        g_nx.add_edges_from(edges)

        for i, n in enumerate(nodes):
            g_nx.nodes[n]['features'] = features[i, :]

        for i, (n1, n2) in enumerate(g_nx.edges()):
            g_nx[n1][n2]['weight'] = weights[i]
        return g_nx
    
    def label_nodes_graph(self):
        g1 = self.dgl_to_networkx()
        list_nodes = list(self.dgl_graph.nodes().cpu().detach().numpy())
        g2 = ig.Graph()
        g2.add_vertices(self.dgl_graph.num_nodes())
        edges = [(edge1.item(), edge2.item()) 
                 for edge1, edge2 in 
                 zip(self.dgl_graph.edges()[0], self.dgl_graph.edges()[1])
                 ]
        g2.add_edges(edges)
        labels = {}
        cpt = 1
        if self.method == 'Infomax':
            for group in g2.community_infomap():
                for j in group:
                    n = list_nodes[j]
                    labels[n] = 'C' + str(cpt)
                cpt += 1
        if self.method == 'Markov':
            for group in g2.community_walktrap().as_clustering():
            # for group in g2.community_walktrap(weights=self.dgl_graph.edata['weight'].tolist()).as_clustering():
                for j in group:
                    n = list_nodes[j]
                    labels[n] = 'C' + str(cpt)
                cpt += 1
        if self.method == 'Label propagation':
            for group in g2.community_label_propagation():
                for j in group:
                    n = list_nodes[j]
                    labels[n] = 'C' + str(cpt)
                cpt += 1
                
        if self.method == 'Finch':
            split_data = FINCH(self.dgl_graph.ndata['features'].cpu().detach().numpy())[0][:,-1]
            for i in range(len(split_data)):
                labels[i] = 'C'+str(split_data[i]+1)
            
        labs = list(set(list(labels.values())))
        for n, c in labels.items():
            g1.nodes[n]['class'] = c
            g1.nodes[n]['color'] = sns.color_palette("hls", len(labs))[labs.index(c)]
            
        frame = {}
        frame['nodes'] = list_nodes
        
        for el in labs: 
            frame[el] = np.zeros(len(list_nodes))
            for n in list_nodes:
                if labels[n] == el:
                    frame[el][n] = 1
        frame = pd.DataFrame(frame)
        frame.set_index('nodes', inplace=True)
        return frame, g1, th.tensor(frame.values).to(device, dtype)
    
    
    
    def node2node_path_attraction(self):
        def get_path_node_forces(nxg, n1, n2):
            som1 = th.tensor(0).to(device, dtype)
            som2 = th.tensor(0).to(device, dtype)
            neigh1 = list(nxg.neighbors(n1))
            neigh2 = list(nxg.neighbors(n2))
            for n in neigh1: som1 += th.tensor(nxg[n1][n]['weight']).to(device, dtype)
            for n in neigh2: som2 += th.tensor(nxg[n2][n]['weight']).to(device, dtype)
            num = som1
            den = th.max(som1, som2)
            return num/den
        
        matrix = th.zeros(self.dgl_graph.num_nodes(), self.dgl_graph.num_nodes()).to(device, dtype)
        nxg = self.dgl_to_networkx()
        # for i, j in combinations(range(self.dgl_graph.num_nodes()), 2):
        for i in range(self.dgl_graph.num_nodes()-1):
            for j in range(i+1, self.dgl_graph.num_nodes()):
                short_path = nx.shortest_path(nxg, source=i, target=j)
                attraction = 1
                for n1, n2 in zip(short_path[0:len(short_path)-1], short_path[1:len(short_path)]):
                    attraction *= get_path_node_forces(nxg, n1, n2)
                matrix[i, j] = attraction
                matrix[j, i] = attraction
        return matrix
    
    def node2node_similarity_attraction(self):
        def get_sim_forces(nxg, n1, n2):
            f1 = th.tensor(nxg.nodes[n1]['features']).to(device, dtype)
            f1 = f1.reshape(1,-1)
            f2 = th.tensor(nxg.nodes[n2]['features']).to(device, dtype)
            f2 = f2.reshape(1,-1)
            # return th.abs(F.cosine_similarity(f1, f2)).squeeze()
            return .5*(F.cosine_similarity(f1, f2)+1).squeeze()
        
        matrix = th.zeros(self.dgl_graph.num_nodes(), self.dgl_graph.num_nodes()).to(device, dtype)
        nxg = self.dgl_to_networkx()
        for i in range(self.dgl_graph.num_nodes()-1):
            for j in range(i+1, self.dgl_graph.num_nodes()):
                matrix[i, j] = get_sim_forces(nxg, i, j)
                matrix[j, i] = get_sim_forces(nxg, j, i)
        return matrix
        
        
    def node2node_force_attraction2(self, simforce, pathforce):
        return simforce.mul(pathforce)
    
    def node2node_force_attraction(self):


        def get_neighbor_node_forces(nxg, n1, n2):
            f1 = th.tensor(nxg.nodes[n1]['features']).to(device, dtype)
            f1 = f1.reshape(1,-1)
            f2 = th.tensor(nxg.nodes[n2]['features']).to(device, dtype)
            f2 = f2.reshape(1,-1)
            som1 = th.tensor(0).to(device, dtype)
            som2 = th.tensor(0).to(device, dtype)
            neigh1 = list(nxg.neighbors(n1))
            neigh2 = list(nxg.neighbors(n2))
            for n in neigh1: som1 += th.tensor(nxg[n1][n]['weight']).to(device, dtype)
            for n in neigh2: som2 += th.tensor(nxg[n2][n]['weight']).to(device, dtype)
            sim = th.abs(F.cosine_similarity(f1, f2)).squeeze()
            num = sim * som1
            den = th.max(som1, som2)
            return num/den

        matrix = th.zeros(self.dgl_graph.num_nodes(), self.dgl_graph.num_nodes()).to(device, dtype)
        nxg = self.dgl_to_networkx()
        for i in range(self.dgl_graph.num_nodes()-1):
            for j in range(i+1, self.dgl_graph.num_nodes()):
                short_path = nx.shortest_path(nxg, source=i, target=j)
                attraction = 1
                for n1, n2 in zip(short_path[0:len(short_path)-1], short_path[1:len(short_path)]):
                    attraction *= get_neighbor_node_forces(nxg, n1, n2)/len(short_path)
                matrix[i, j] = attraction
    
        return matrix
    
    
    
    def group2node_force_attraction(self, members, matrix_forces):
        return self.scale_matrice(th.matmul(matrix_forces, members))
    
    
    
   