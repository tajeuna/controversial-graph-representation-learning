#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 04:01:31 2022

@author: etienne
"""

import pandas as pd
import numpy as np
import torch as th
import dgl
import networkx as nx
import igraph as ig
from tools import DataLoader
from itertools import combinations
from scipy.stats import wasserstein_distance as wd
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture as GM
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings("ignore")
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

color_names = [name for name, color in colors.items()]





class import_data:
    
    def __init__(
            self, 
            path_file=None, 
            gml_path_file=None, 
            url=None, 
            generate_features=False, 
            size_feature_vec=50, 
            generate_weights = True,
            manualSeed = None,
            components = 10
            ):
        
        self.path_file = path_file
        self.generate_features = generate_features
        self.size_feature_vec = size_feature_vec
        self.url = url
        self.generate_weights = generate_weights
        self.gml_path_file = gml_path_file
        self.manualSeed = manualSeed 
        self.components = components
        self.intervals = {}
        min_val =0
        cpt = 1
        while min_val < 1.:
            self.intervals['g'+str(cpt)] = [min_val, min_val+(1./components)]
            min_val += 1./components
            cpt += 1
        print(self.intervals)
        
        
    
        
        
    def cast_frame_to_normal_distribution(self, frame):
        tmp_frame = {}
        for dim in list(frame):
            X = frame[dim].values.reshape(len(frame), 1)
            model = GM(n_components=1, random_state=self.manualSeed).fit(X)
            loc = model.means_.squeeze().item()
            scale = model.covariances_.squeeze().item()
            tmp_frame[dim] = norm.pdf(frame[dim].values, loc=loc, scale=scale)
            
        return pd.DataFrame(tmp_frame, index=frame.index)
    
    def cast_frame_to_categories(self, frame):
        def categorize_series(series):
            sequence = []
            for val in series:
                placed = False
                for k, interval in self.intervals.items():
                    if min(interval) <= val < max(interval):
                            sequence.append(k)
                            placed = True
                if placed == False:
                    sequence.append('g'+str(self.components))
            return sequence
        
        header = list(frame)
        
        frame_label = frame.copy()
        for s in header: 
            frame_label[s] = categorize_series(frame[s])
        
        
        return frame_label
    
    def get_graph(self, frame, weights, threshold=None):
        headers = list(frame)
        thresh = max(list(weights.values()))-10
        g, g2 = None, None
        final_g, final_g2 = None, None
        edges = []
        w_attr = []
        if threshold is None:
            print('Size of series '+str(max(list(weights.values())))+' testing with threshold:')
            while True:
                print(thresh, end=',')
                g = nx.Graph()
                g2 = ig.Graph()
                g.add_nodes_from(list(frame))
                g2.add_vertices(np.arange(frame.shape[1]))
                
                for n1, n2 in weights.keys():
                    i, j = headers.index(n1), headers.index(n2)
                    w = weights[(n1, n2)]
                    if  w > thresh:
                        g.add_edge(n1, n2)
                        g[n1][n2]['weight'] = w 
                        g2.add_edges([(i,j)])
                        w_attr.append(w)
                
                if nx.is_connected(g):
                    final_g, final_g2 = g, g2
                    thresh += 1
                else:
                    if final_g is None:
                        thresh -= 1
                        edges = []
                        w_attr = []
                    else:
                        break
            
        else:
            final_g = nx.Graph()
            final_g2 = ig.Graph()
            final_g.add_nodes_from(list(frame))
            final_g2.add_vertices(np.arange(frame.shape[1]))
            
            for n1, n2 in weights.keys():
                i, j = headers.index(n1), headers.index(n2)
                w = weights[(n1, n2)]
                if  w > threshold:
                    final_g.add_edge(n1, n2)
                    final_g[n1][n2]['weight'] = w 
                    final_g2.add_edges([(i,j)])
                    w_attr.append(w)
            
            if nx.is_connected(final_g) == False:
                print('Warning ...\n \t The built graph is disconnected \n \t Try with a smaller threshold')
                    
        # clusters = []
        # if self.clustering_method == 'Markov':
        #     for cl in final_g2.community_walktrap(weights=w_attr).as_clustering():
        #         clusters.append([list(frame)[el] for el in cl])
        # if self.clustering_method == 'Infomax':
        #     for cl in final_g2.community_infomap(w_attr):
        #         clusters.append([list(frame)[el] for el in cl])
                    
            
        #     # print('graph build with '+str(thresh)+' threshold')
        #     print(nx.info(final_g))
        
        #     for i in range(len(clusters)):
        #         cluster = clusters[i]
        #         for n in cluster:
        #             final_g.nodes[n]['class'] = 'C'+str(i+1)
        #             final_g.nodes[n]['color'] = color_names[i]
        #             final_g.nodes[n]['features'] = self.original.loc[frame.index, [n]].values.flatten()
            
        return final_g, final_g2
        
    
    def nice_graph(self):
        source = th.tensor([0,0,1,1,2,2,3,4,4,4,5,5,6,6,7])
        destination = th.tensor([1,3,2,3,3,5,4,5,6,8,6,7,7,8,8])
        g = dgl.graph((source, destination))
        if self.generate_features:
            g.ndata['features'] = th.rand(g.num_nodes(),
                                           self.size_feature_vec)
        if self.generate_weights:
            g.edata['weight'] = th.tensor([.75,.89,.68,.94,.77,.15,.05,.68,.72,.71,.83,.65,.94,.89,.93])
        else:
            g.edata['weight'] = th.ones(len(source))
        return g, g.nodes()
    
    def gml_data_to_network(self):
        g = nx.read_gml(self.gml_path_file)
        fr = {}
        fr['source'] = []
        fr['destination'] = []
        for u, v in g.edges():
            fr['source'].append(u)
            fr['destination'].append(v)
        fr = pd.DataFrame(fr)
        nodes = list(set(fr.source).union(set(fr.destination)))
        fr.source = [nodes.index(val) for val in fr.source]
        fr.destination = [nodes.index(val) for val in fr.destination]
        g = dgl.graph((th.tensor(fr.source.values), 
                       th.tensor(fr.destination.values))
                     )
        if self.generate_features:
            np.random.seed(self.manualSeed)
            feat = MinMaxScaler().fit_transform(np.random.randn(g.num_nodes(), 
                                                                self.size_feature_vec))
            g.ndata['features'] = th.tensor(feat)
        if self.generate_weights:
            g.edata['weight'] = th.rand(g.num_edges())
        return g, nodes
        
    def karate_club(self):
        source, destination = [], []
        for u, v in nx.karate_club_graph().edges():
            source.append(u)
            destination.append(v)
        source = th.tensor(source)
        destination = th.tensor(destination)
        g = dgl.graph((source, destination))
        if self.generate_features:
            g.ndata['features'] = th.rand(g.num_nodes(),
                                           self.size_feature_vec)
        if self.generate_weights:
            g.edata['weight'] = th.rand(g.num_edges())
        else:
            g.edata['weight'] = th.ones(g.num_edges())
        return g, g.nodes()
    
    def covid_long_data(self, 
                        path_to_metadata=None, 
                        path_to_metabolomic=None,  
                        path_to_proteomic=None, 
                        path_to_proteomics_cyt=None, 
                        threshold = None,
                        edge_type=None,
                        attribute_type=None,
                        distance=None,
                        data_location=None,
                        axis = None
                       ):
        def scale_frame(frame):
            tmpfr = pd.DataFrame(MinMaxScaler().fit_transform(frame.values))
            tmpfr.columns = frame.columns
            tmpfr.index = frame.index
            return tmpfr
        final_g, final_g2, g, nodes = None, None, None, None
        loader = DataLoader(path_to_metadata=path_to_metadata, 
                            path_to_metabolomics=path_to_metabolomic, 
                            path_to_proteomics=path_to_proteomic, 
                            path_to_proteomics_cyt=path_to_proteomics_cyt)
        
        metadata = loader.get_metadata(all_symptoms=True)
        proteomics = loader.get_proteomic(metadata=metadata, dropping=True)
        proteomic_cyt = loader.get_cytokine(metadata=metadata, dropping=True)
        metabolomics = loader.get_metabolomic(metadata=metadata, dropping=False)
        metabolomics = metabolomics.T.loc[proteomics.index, :]
        metabolomics.dropna(axis=1, inplace=True)
        
        for h in list(metabolomics): 
            try:
                metabolomics[h] = metabolomics[h].values.astype(float)
            except:
                del metabolomics[h]
        
        sympto = metadata[metadata.Symptomatic == 1]
        non_sympto = metadata[metadata.Symptomatic == 0]
        sympto.set_index('ID', inplace=True)
        non_sympto.set_index('ID', inplace=True)
        
        weights = {}
        metadata.set_index('ID', inplace=True)
        
        metacols = ['Fever', 'cough', 'difficulty breathing', 'chest pain', 'fatigue', 
         'myalgia', 'sore throat ', 'headache', 'dizziness', 'anosmia', 'agueusia', 'nasal congestion', 'diarrhea', 
         'nausea', 'vomitting', 'abdominal pain', 'Loss of appetite', 'other']
        
        if edge_type == 'residual':
            fra = metadata.loc[:, metacols]
            if distance == 'wasserstein':
                if axis == 0:
                    fra = self.cast_frame_to_normal_distribution(fra)
                    for n1, n2 in combinations(metadata.index, 2):
                        v1 = fra.loc[n1, :].values.astype(float)
                        v2 = fra.loc[n2, :].values.astype(float)
                        weights[(n1, n2)] = wd(v1, v2) 
                if axis == 1:
                    fra = self.cast_frame_to_normal_distribution(fra.T).T
                    for n1, n2 in combinations(metadata.index, 2):
                        v1 = fra.loc[n1, :].values.astype(float)
                        v2 = fra.loc[n2, :].values.astype(float)
                        weights[(n1, n2)] = wd(v1, v2) 
                
            else:
                for n1, n2 in combinations(metadata.index, 2):
                    v1 = fra.loc[n1, :].values.astype(float).reshape(1, len(metacols))
                    v2 = fra.loc[n2, :].values.astype(float).reshape(1, len(metacols))
                    weights[(n1, n2)] = cdist(v1, v2, metric=distance).flatten()[0]
                
                
        elif edge_type == 'metabolomics':
            if distance == 'wasserstein':
                if axis == 0:
                    fra = self.cast_frame_to_normal_distribution(metabolomics)
                    for n1, n2 in combinations(metadata.index, 2):
                        v1 = fra.loc[n1, :].values.astype(float)
                        v2 = fra.loc[n2, :].values.astype(float)
                        weights[(n1, n2)] = wd(v1, v2) 
                if axis == 1:
                    fra = self.cast_frame_to_normal_distribution(fra.T).T
                    for n1, n2 in combinations(metadata.index, 2):
                        v1 = fra.loc[n1, :].values.astype(float)
                        v2 = fra.loc[n2, :].values.astype(float)
                        weights[(n1, n2)] = wd(v1, v2) 
            elif distance == 'SAX-Matching':
                fra = scale_frame(metabolomics)
                fra = self.cast_frame_to_categories(fra)
                for n1, n2 in combinations(metadata.index, 2):
                    vec1 = fra.loc[n1,:].values.tolist()
                    vec2 = fra.loc[n2,:].values.tolist()
                    weights[(n1, n2)] = len([i for i,j in zip(vec1, vec2) if i==j])
                print(fra.T)
                final_g, final_g2 = self.get_graph(fra.T, weights)
            else:
                fra = scale_frame(metabolomics)
                for n1, n2 in combinations(metadata.index, 2):
                    v1 = fra.loc[n1, :].values.astype(float).reshape(1, len(list(metabolomics)))
                    v2 = fra.loc[n2, :].values.astype(float).reshape(1, len(list(metabolomics)))
                    weights[(n1, n2)] = cdist(v1, v2, metric=distance).flatten()[0]
                
        elif edge_type == 'proteomics':
            if distance == 'wasserstein':
                if axis == 0:
                    fra = self.cast_frame_to_normal_distribution(proteomics)
                    for n1, n2 in combinations(metadata.index, 2):
                        v1 = fra.loc[n1, :].values.astype(float)
                        v2 = fra.loc[n2, :].values.astype(float)
                        weights[(n1, n2)] = wd(v1, v2) 
                if axis == 1:
                    fra = self.cast_frame_to_normal_distribution(proteomics.T).T
                    for n1, n2 in combinations(metadata.index, 2):
                        v1 = fra.loc[n1, :].values.astype(float)
                        v2 = fra.loc[n2, :].values.astype(float)
                        weights[(n1, n2)] = wd(v1, v2) 
            elif distance == 'SAX-Matching':
                fra = scale_frame(proteomics)
                fra = self.cast_frame_to_categories(fra)
                for n1, n2 in combinations(metadata.index, 2):
                    vec1 = fra.loc[n1,:].values.tolist()
                    vec2 = fra.loc[n2,:].values.tolist()
                    weights[(n1, n2)] = len([i for i,j in zip(vec1, vec2) if i==j])
                final_g, final_g2 = self.get_graph(fra.T, weights)
            else:
                fra = scale_frame(proteomics)
                for n1, n2 in combinations(metadata.index, 2):
                    v1 = fra.loc[n1, :].values.astype(float).reshape(1, len(list(proteomics)))
                    v2 = fra.loc[n2, :].values.astype(float).reshape(1, len(list(proteomics)))
                    weights[(n1, n2)] = cdist(v1, v2, metric=distance).flatten()[0]
                
        elif edge_type == 'proteomics_cyt':
            if distance == 'wasserstein':
                if axis == 0:
                    fra = self.cast_frame_to_normal_distribution(proteomic_cyt)
                    for n1, n2 in combinations(metadata.index, 2):
                        v1 = fra.loc[n1, :].values.astype(float)
                        v2 = fra.loc[n2, :].values.astype(float)
                        weights[(n1, n2)] = wd(v1, v2) 
                if axis == 1:
                    fra = self.cast_frame_to_normal_distribution(proteomic_cyt.T).T
                    for n1, n2 in combinations(metadata.index, 2):
                        v1 = fra.loc[n1, :].values.astype(float)
                        v2 = fra.loc[n2, :].values.astype(float)
                        weights[(n1, n2)] = wd(v1, v2) 
            elif distance == 'SAX-Matching':
                fra = scale_frame(proteomic_cyt)
                fra = self.cast_frame_to_categories(fra)
                for n1, n2 in combinations(metadata.index, 2):
                    vec1 = fra.loc[n1,:].values.tolist()
                    vec2 = fra.loc[n2,:].values.tolist()
                    weights[(n1, n2)] = len([i for i,j in zip(vec1, vec2) if i==j])
                final_g, final_g2 = self.get_graph(fra.T, weights)
            else:
                fra = scale_frame(proteomics)
                for n1, n2 in combinations(metadata.index, 2):
                    v1 = fra.loc[n1, :].values.astype(float).reshape(1, len(list(proteomic_cyt)))
                    v2 = fra.loc[n2, :].values.astype(float).reshape(1, len(list(proteomic_cyt)))
                    weights[(n1, n2)] = cdist(v1, v2, metric=distance).flatten()[0]
        
        if final_g2 is not None:
            source, destination = [], []
            nodes = list(final_g.nodes())
            for u, v in final_g.edges():
                nodes = list(final_g.nodes())
                source.append(nodes.index(u))
                destination.append(nodes.index(v))
            source = th.tensor(source)
            destination = th.tensor(destination)
            g = dgl.graph((source, destination))
            edge_features = nx.get_edge_attributes(final_g, 'weight')
            # g.ndata['features'] = th.tensor([v for k,v in )
            g.edata['weight'] = th.tensor([edge_features[e] for e in final_g.edges()])
        else:
            fr = {}
            fr['source'] = []
            fr['target'] = []
            fr['weight'] = []
            kap = 0
            if threshold is not None:
                kap += threshold
            else:
                kap += np.median(list(weights.values()))
            print('Threshold '+str(kap))
            for k in weights:
                if weights[k] < kap:
                    fr['source'].append(k[0])
                    fr['target'].append(k[1])
                    fr['weight'].append(weights[k])
                    
            fr = pd.DataFrame(fr)
            nodes = list(set(fr.source).union(set(fr.target)))
            fr.source = [nodes.index(val) for val in fr.source]
            fr.target = [nodes.index(val) for val in fr.target]
            g = dgl.graph((th.tensor(fr.source.values), 
                           th.tensor(fr.target.values)))
            
            g.edata['weight'] = th.tensor(fr.weight)
            
        if attribute_type == 'metabolomics':
            g.ndata['features'] = th.tensor(metabolomics.loc[nodes,:].values)
        elif attribute_type == 'proteomics':
            g.ndata['features'] = th.tensor(proteomics.loc[nodes,:].values)
        elif attribute_type == 'proteomic_cyt':
            g.ndata['features'] = th.tensor(proteomic_cyt.loc[nodes,:].values)
        
        return g, final_g