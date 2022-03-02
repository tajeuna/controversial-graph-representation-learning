#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 12:24:19 2022

@author: etienne
"""


import pandas as pd
import numpy as np
import torch as th
from sklearn.metrics import accuracy_score
from sklearn import svm, tree
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import pyscm
import warnings
warnings.filterwarnings("ignore")


device = 'cuda:0'
# device = 'cpu'
dtype = th.float



class evaluation_on_classical_models:

    def __init__(
            self, 
            embedding = None, 
            percentage = None, 
            nbre_samples = None, 
            response = None,
            manualSeed = None
            ):

        self.embedding = embedding ## un array/matrice de type n x m
        self.data = pd.DataFrame(embedding) ## charge la matrice dans un dataframe
        self.data['y'] = response ## réponse à prédire
        self.percentage = percentage ## pourcentage de données d'entraînement, doit toujours être une valeur entière de préférence comprise entre [50, 100]
        self.nbre_samples = nbre_samples ## nombre de validation croisée à effectuée, valeur entière de préférence supérieure à 1. Plus elle sera grande, plus le coût de calcul sera aussi.
        self.response = response ## labels à prédire, pour certaines des méthodes ci-dessous, (exemple model scm), plus de deux labels ne seront pas prédictibles par le modèle.
        self.manualSeed = manualSeed ## valeur entière pour définir la position du générateur de nombres. Elle permet de reproduire les résultats.

    def random_train_test_frame_choice(self):
        trainfr, testfr = [], []
        for label in np.unique(self.response):
            subfr = self.data[self.data.y == label]
            trainsub = subfr.sample(n=int(len(subfr) * self.percentage * .01), random_state=self.manualSeed)
            testsub = subfr.loc[[ind for ind in subfr.index if ind not in trainsub.index], :]
            trainfr.append(trainsub)
            testfr.append(testsub)
        return pd.concat(trainfr), pd.concat(testfr)

    def get_samples(self):
        trains, tests = [], []
        for i in range(self.nbre_samples):
            train, test = self.random_train_test_frame_choice()
            trains.append(train)
            tests.append(test)
        return trains, tests
    
    def cross_valid_scm_model(self, train_sets=None, test_sets=None):
        score = 0
        if train_sets is None:
            train_sets, test_sets = self.get_samples()
        for i in range(self.nbre_samples):
            train, test = train_sets[i], test_sets[i]
            X_train, y_train = train.loc[:, list(train)[0:len(list(train) ) -1]].values, train.loc[:, [list(train)[-1]]]
            X_test, y_test = test.loc[:, list(test)[0:len(list(test) ) -1]].values, test.loc[:, [list(test)[-1]]]
            model1 = pyscm.SetCoveringMachineClassifier()
            model1.fit(X_train, np.ravel(y_train))
            predicted1 = model1.predict(X_test)
            score += accuracy_score(predicted1, y_test)
        return score /self.nbre_samples

    def cross_valid_svm_model(self, train_sets=None, test_sets=None):
        score = 0
        if train_sets is None:
            train_sets, test_sets = self.get_samples()
        for i in range(self.nbre_samples):
            train, test = train_sets[i], test_sets[i]
            X_train, y_train = train.loc[:, list(train)[0:len(list(train) ) -1]].values, train.loc[:, [list(train)[-1]]]
            X_test, y_test = test.loc[:, list(test)[0:len(list(test) ) -1]].values, test.loc[:, [list(test)[-1]]]
            model1 = svm.SVC()
            model1.fit(X_train, np.ravel(y_train))
            predicted1 = model1.predict(X_test)
            score += accuracy_score(predicted1, y_test)
        return score /self.nbre_samples
    
    def cross_valid_svm_linear_model(self, train_sets=None, test_sets=None):
        score = 0
        if train_sets is None:
            train_sets, test_sets = self.get_samples()
        for i in range(self.nbre_samples):
            train, test = train_sets[i], test_sets[i]
            X_train, y_train = train.loc[:, list(train)[0:len(list(train) ) -1]].values, train.loc[:, [list(train)[-1]]]
            X_test, y_test = test.loc[:, list(test)[0:len(list(test) ) -1]].values, test.loc[:, [list(test)[-1]]]
            model1 = svm.SVC(kernel='linear')
            model1.fit(X_train, np.ravel(y_train))
            predicted1 = model1.predict(X_test)
            score += accuracy_score(predicted1, y_test)
        return score /self.nbre_samples
    
    def cross_valid_svm_sigmoid_model(self, train_sets=None, test_sets=None):
        score = 0
        if train_sets is None:
            train_sets, test_sets = self.get_samples()
        for i in range(self.nbre_samples):
            train, test = train_sets[i], test_sets[i]
            X_train, y_train = train.loc[:, list(train)[0:len(list(train) ) -1]].values, train.loc[:, [list(train)[-1]]]
            X_test, y_test = test.loc[:, list(test)[0:len(list(test) ) -1]].values, test.loc[:, [list(test)[-1]]]
            model1 = svm.SVC(kernel='sigmoid')
            model1.fit(X_train, np.ravel(y_train))
            predicted1 = model1.predict(X_test)
            score += accuracy_score(predicted1, y_test)
        return score /self.nbre_samples

    def cross_valid_mlp_model(self, train_sets=None, test_sets=None):
        score = 0
        if train_sets is None:
            train_sets, test_sets = self.get_samples()
        for i in range(self.nbre_samples):
            train, test = train_sets[i], test_sets[i]
            X_train, y_train = train.loc[:, list(train)[0:len(list(train) ) -1]].values, train.loc[:, [list(train)[-1]]]
            X_test, y_test = test.loc[:, list(test)[0:len(list(test) ) -1]].values, test.loc[:, [list(test)[-1]]]
            model = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes = (5, 2), random_state = 1)
            model.fit(X_train, np.ravel(y_train))
            predicted1 = model.predict(X_test)
            score += accuracy_score(predicted1, y_test)
        return score /self.nbre_samples

    def cross_valid_sgdc_model(self, train_sets=None, test_sets=None):
        score = 0
        if train_sets is None:
            train_sets, test_sets = self.get_samples()
        for i in range(self.nbre_samples):
            train, test = train_sets[i], test_sets[i]
            X_train, y_train = train.loc[:, list(train)[0:len(list(train) ) -1]].values, train.loc[:, [list(train)[-1]]]
            X_test, y_test = test.loc[:, list(test)[0:len(list(test) ) -1]].values, test.loc[:, [list(test)[-1]]]
            model = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
            model.fit(X_train, np.ravel(y_train))
            predicted1 = model.predict(X_test)
            score += accuracy_score(predicted1, y_test)
        return score /self.nbre_samples

    def cross_valid_dtree_model(self, train_sets=None, test_sets=None):
        score = 0
        if train_sets is None:
            train_sets, test_sets = self.get_samples()
        for i in range(self.nbre_samples):
            train, test = train_sets[i], test_sets[i]
            X_train, y_train = train.loc[:, list(train)[0:len(list(train) ) -1]].values, train.loc[:, [list(train)[-1]]]
            X_test, y_test = test.loc[:, list(test)[0:len(list(test) ) -1]].values, test.loc[:, [list(test)[-1]]]
            model = tree.DecisionTreeClassifier()
            model.fit(X_train, np.ravel(y_train))
            predicted1 = model.predict(X_test)
            score += accuracy_score(predicted1, y_test)
        return score /self.nbre_samples

    def cross_valid_rforest_model(self, train_sets=None, test_sets=None):
        score = 0
        if train_sets is None:
            train_sets, test_sets = self.get_samples()
        for i in range(self.nbre_samples):
            train, test = train_sets[i], test_sets[i]
            X_train, y_train = train.loc[:, list(train)[0:len(list(train) ) -1]].values, train.loc[:, [list(train)[-1]]]
            X_test, y_test = test.loc[:, list(test)[0:len(list(test) ) -1]].values, test.loc[:, [list(test)[-1]]]
            model = RandomForestClassifier(n_estimators=10)
            model.fit(X_train, np.ravel(y_train))
            predicted1 = model.predict(X_test)
            score += accuracy_score(predicted1, y_test)
        return score /self.nbre_samples

    def cross_valid_boosting_model(self, train_sets=None, test_sets=None):
        score = 0
        if train_sets is None:
            train_sets, test_sets = self.get_samples()
        for i in range(self.nbre_samples):
            train, test = train_sets[i], test_sets[i]
            X_train, y_train = train.loc[:, list(train)[0:len(list(train) ) -1]].values, train.loc[:, [list(train)[-1]]]
            X_test, y_test = test.loc[:, list(test)[0:len(list(test) ) -1]].values, test.loc[:, [list(test)[-1]]]
            model = AdaBoostClassifier(n_estimators=100)
            model.fit(X_train, np.ravel(y_train))
            predicted1 = model.predict(X_test)
            score += accuracy_score(predicted1, y_test)
        return score /self.nbre_samples