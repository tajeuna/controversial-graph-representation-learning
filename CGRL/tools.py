import pandas as pd
from tqdm import tqdm
import warnings
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import metrics
from finch import FINCH
from scipy.spatial.distance import cosine


class DataLoader:
    
    def __init__(self, 
                 path_to_metadata=None, 
                 path_to_metabolomics=None, 
                 path_to_proteomics=None, 
                 path_to_proteomics_cyt=None):
        
        self.path_to_metadata = path_to_metadata
        self.path_to_metabolomics = path_to_metabolomics
        self.path_to_proteomics = path_to_proteomics
        self.path_to_proteomics_cyt = path_to_proteomics_cyt
        
        
    def get_metadata(self, all_symptoms=False):
        
        meta1 = pd.read_csv(self.path_to_metadata, index_col=0)
        meta = None
        if all_symptoms == False:
            meta = meta1.loc[:, ['ID', 'Time since infection', 'Age']]
        else:
            meta = meta1.loc[:, ['ID', 'Time since infection', 'Age', 'Fever', 'cough', 'difficulty breathing', 
                                 'chest pain', 'fatigue', 'myalgia', 'sore throat ', 'headache', 'dizziness', 
                                 'anosmia', 'agueusia', 'nasal congestion', 'diarrhea', 'nausea', 'vomitting', 
                                 'abdominal pain', 'Loss of appetite', 'other']]
        meta['Symptomatic'] = [1 if v == 'S' else 0 for v in meta1['Unnamed: 3']]
        return meta
    
    
    def get_metabolomic(self, metadata, dropping=False):
        metabolomics = pd.read_csv(self.path_to_metabolomics, skiprows=1, index_col=0)
        heads = [val for val in list(metabolomics) if len(val.split('_')[-2].split('-')) == 2]
        metabolomics = metabolomics.loc[:, heads]
        new_heads = [val.split('_')[-2] for val in list(metabolomics)]
        metabolomics.columns = new_heads
        if dropping == True:
            for h in list(metabolomics):
                metabolomics[h] = pd.to_numeric(metabolomics[h].values, errors='coerce')
            metabolomics.dropna(axis=1, inplace=True)
        return metabolomics

    
    def get_proteomic(self, metadata, dropping=False):
        proteomics = pd.read_csv(self.path_to_proteomics, skiprows=4, skipfooter=4, engine='python')
        proteomics = proteomics.drop([0,1])
        proteomics = proteomics.drop(columns = list(proteomics)[-4:])
        proteomics.set_index('Uniprot ID', inplace=True)
        proteomics = proteomics.loc[metadata.ID, :]
        if dropping == True:
            for h in list(proteomics):
                proteomics[h] = pd.to_numeric(proteomics[h].values, errors='coerce')
            proteomics.dropna(axis=1, inplace=True)
        return proteomics
    
    
    def get_cytokine(self, metadata, dropping=False):
        cytokines = pd.read_csv(self.path_to_proteomics_cyt, skiprows=4, skipfooter=14, engine='python')
        cytokines = cytokines.drop([0, 1, 2])
        cytokines = cytokines.drop(columns = list(cytokines)[-2:])
        cytokines.set_index('Uniprot ID', inplace=True)
        cytokines = cytokines.loc[metadata.ID, :]
        if dropping == True:
            for h in list(cytokines):
                cytokines[h] = pd.to_numeric(cytokines[h].values, errors='coerce')
            cytokines.dropna(axis=1, inplace=True)
        return cytokines
    
    
    def merge_data(self, 
                   cytokine=None, 
                   metabolomic=None, 
                   proteomic=None, 
                   metadata=None, 
                   dropping=None):
        
        new_fr = metadata.copy()
        new_fr.set_index('ID', inplace=True)
        fr = pd.concat([new_fr, metabolomic, proteomic, cytokine], axis=1)
        if dropping == True:
            for h in list(fr):
                fr[h] = pd.to_numeric(fr[h].values, errors='coerce')
            fr = fr.dropna(axis=1, inplace=True)
        return fr




    
    
class PreProcess:
    
    def __init__(self, fr):
        
        self.fr = fr
        
        
    def transform_data(self, method='PCA', nbre_components=3):
        heads = [h for h in list(self.fr) if h not in ['Time since infection', 'Symptomatic']]
        X = self.fr.loc[:, heads].values
        X = MinMaxScaler((0,1)).fit_transform(X)
        X2 = None
        if method == 'PCA':
            X2 = PCA(n_components=nbre_components).fit_transform(X)
        elif method == 'tsne':
            X2 = TSNE(n_components=nbre_components).fit_transform(X)
        new_fr = pd.DataFrame(X2, columns=['AX'+str(i+1) for i in range(nbre_components)], index=self.fr.index)
        try:
            new_fr['Time since infection'] = self.fr['Time since infection']
            new_fr['Symptomatic'] = self.fr['Symptomatic']
        except: pass

        return new_fr
    

    

