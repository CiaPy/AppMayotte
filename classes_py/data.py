# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 10:06:48 2022

@author: mato
"""

import pandas as pd


def get_datapointés(df):
      my_listfeaturespointes = df.columns.values.tolist()
      return my_listfeaturespointes
        
@staticmethod
def group_by_source(df):
           df = df.drop(['z','x_epsg4471','y_epsg4471'], axis=1).groupby("code_interface").sum().T
           df["source"] = df.sum(axis=1)
           return df
   
def get_datahv(df):
    my_listfeatureshv = df
    return my_listfeatureshv
    
    
@staticmethod
def count_hv(df):
    return len(df.index)
       
