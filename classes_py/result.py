# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 10:49:19 2022

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

# class Result():
    
#     def __init__(self, dtf):
#         self.dtf = dtf
        
#         @staticmethod
#         def process_df(df):
            
#             # Define features data (X)
#             Xessaie = df[['rho','F0_IDW']]
            
#             # Define target data (y)
#             yessaie = df[['Profondeur']]
            
#             # Initialise the encoder
#             scmm = MinMaxScaler()
            
#             # Apply encoder on target data
#             Xessaie = scmm.fit_transform(Xessaie)
#             return Xessaie

#         def loadandresult(file_pickle,df):
#             with open(file_pickle , 'rb') as f:
#                 lr = pickle.load(f)
#                 df['reg_pred'] = lr.predict(self.process_df(file_pickle,self.df))
                