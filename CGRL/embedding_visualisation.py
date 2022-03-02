#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 14:04:31 2022

@author: etienne
"""

# import seaborn as sns
# import dash
# from dash import dcc
# from dash import html
# import chart_studio.plotly as py
# import plotly.graph_objects as go
# from plotly.offline import iplot
# from plotly.offline import init_notebook_mode, iplot
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# import numpy as np
import networkx as nx


class plot_features:
    
    def __init__(
            self,
            embedding=None, 
            nxg=None,
            view = None,
            nbre_component = 2,
            manualSeed = None
            ):
        self.mat = None
        if embedding is not None:
            self.mat = embedding.cpu().detach().numpy()
        self.nxg = nxg
        self.view = view
        self.manualSeed = manualSeed
        self.nbre_component = nbre_component
        
    def plot(self):
        
        model_view = None
        if self.view is not None:
            if self.view == 'PCA':
                model_view = PCA(n_components=self.nbre_component)
            elif self.view == 'TSNE':
                model_view = TSNE(n_components=self.nbre_component, random_state=self.manualSeed)
            
            data = model_view.fit_transform(self.mat)
                
            fig = plt.figure(dpi=175, figsize=(3,3))
            # axe0 = fig.add_subplot(1, 3, 1)
            axe1 = fig.add_subplot(111)
            for n in self.nxg.nodes():
                axe1.plot(data[n, 0], data[n, 1], '*', color=self.nxg.nodes[n]['color'])
                
            axe1.set_title(self.view+' projection')
            plt.show()
        else:
            fig = plt.figure(dpi=175, figsize=(3,3))
            # axe0 = fig.add_subplot(1, 3, 1)
            axe1 = fig.add_subplot(111)
            for n in self.nxg.nodes():
                axe1.plot(self.mat[n, 0], self.mat[n, 1], '*', color=self.nxg.nodes[n]['color'])
                
            axe1.set_title(' projection')
            plt.show()
            
    def plot_graph(self):
        nx.draw(self.nxg, pos=nx.kamada_kawai_layout(self.nxg))
                