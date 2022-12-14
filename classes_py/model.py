# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 10:19:43 2022

@author: mato
"""


import pandas as pd
import matplotlib.pyplot as plt
import math
import reverse_geocoder as rg
import csv

import pickle

# Machine learning - Preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler, StandardScaler,MinMaxScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer

# Machine learning - Modèle selection
from sklearn.model_selection import train_test_split, GridSearchCV

# Machine learning - Métriques d'erreur
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, accuracy_score, ConfusionMatrixDisplay, f1_score, fbeta_score,classification_report
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

class Model():
    
    def __init__(self, dtf):
        self.dtf = dtf     
    
        @staticmethod
        def mlpRegressor(dft):
            #Cible code interface 
            dft.dtypes
    
            # Define features data (X)
            X = dft[['rho','F0_IDW']]
    
            # Initialise the encoder
            scmm = MinMaxScaler()
    
            # Apply encoder on features data
    
            X = scmm.fit_transform(X)
    
            # Define target data (y)
            y = dft[['Profondeur']]
            
            # Split data into 2 parts : train & test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            regr = MLPRegressor(hidden_layer_sizes=(20),random_state=1,learning_rate="constant", activation='relu', solver='lbfgs', max_iter=2000,  early_stopping=True)
    
            model_reg=regr.fit(X_train, y_train)
    
            y_predr = model_reg.predict(X_test)
    
            MSE= metrics.mean_squared_error(y_test, y_predr)
    
            RMSE = math.sqrt(MSE)
            return model_reg, MSE, RMSE
    
        def mymodel(self):
            self.dft = self.mlpRegressor(self.dft)