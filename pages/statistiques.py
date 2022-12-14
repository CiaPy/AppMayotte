# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 09:32:55 2022

@author: mato
"""

import pandas as pd
import base64

import dash
from dash import html, dcc, callback, Input, Output
from dash.dependencies import State
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
import dash_labs as dl
from dash import dash_table

import seaborn as sns
import matplotlib.pyplot as plt
import psycopg2

#mydata
'''
df_pointés = pd.read_csv(r"D:\Documents\mato\OneDrive - BRGM\Bureau\Aleasismique_Mayotte\Data_processed\RESULTATS\processing_pointesinterfaces_v1.csv", sep=";")
df_hv = pd.read_csv(r"D:\Documents\mato\OneDrive - BRGM\Bureau\Aleasismique_Mayotte\Data_processed\HV\HV_processed.csv", sep=";")
df_aem = pd.read_csv(r"D:\Documents\mato\OneDrive - BRGM\Bureau\Aleasismique_Mayotte\Data_processed\AEM\aem_processed_v1.csv", sep=";")
'''

conn = psycopg2.connect(host="dpg-cecsgdsgqg459grbbo70-a.frankfurt-postgres.render.com", port = 5432, database="dbmayotte", user="ciapy", password="fkid4gbdDC58V5wwjZGA4rqqcLnRVyZ0")
statment= f""" select * FROM public.dataset """
df= pd.read_sql_query(statment ,con=conn)
df

#Fonction

def general_analysis(df):
    global shp_df
    shp_df = df.shape
    global dtype_df
    dtype_df =df.dtypes
    dtype_df= dtype_df.to_frame()
    global info_df
    info_df = df.info()
    global isna_df
    isna_df = df.isna().sum()
    return shp_df,dtype_df,info_df,isna_df

def univarirate_analysis_A(df):
    descr_df= df.describe(include='all')
    for i, column in enumerate(df.columns, 1):
        plt.subplots(i)
        plot_hist = sns.histplot(df[column])
    return descr_df, plot_hist
    
def correlation_matrix(df):
    corr_df= df.corr()
    heatmapcorr_df = sns.heatmap(df.corr(), annot=True, linewidths=0.5)
    return corr_df, heatmapcorr_df

general_analysis(df)
univarirate_analysis_A(df)
correlation_matrix(df)


tablepointes = dash_table.DataTable(
    id='tablepointes',
    columns=[{"name": i, "id": i} for i in df.columns],
    data=df.to_dict('records'),
    page_action="native",
    page_size=5,
)



layout =  dcc.Tabs([
        
   dcc.Tab( label='H/V', children=[ 
       
        html.Div([
            html.H2("General"),
            html.H4("Description générale des données"),
        
'''
         html.Div([   dbc.Card(
               [
                  html.H2(shp_df, className="card-title"),
                   html.P("Shape ", className="card-text"),
               ],
               body=True,
               color="primary",
               inverse=True, style={'height': '13vh', 'width': '20vh', 'textAlign': 'center'},  className="h-100"
           ),]),
        

 # html.Div([  dash_table.DataTable(
 #    data=dtype_df.to_dict('records'),
 #    columns=[{'id': c, 'name': c} for c in dtype_df.columns]),]),
html.Div([
            html.H2("Analyse univariée"),
            html.H2("Matrice de corrélation"),
            html.H2("Analyse bivariée"),
            ]),]),
                                       
        dcc.Tab( label='AEM', children=[ 
            
            html.Div([
                html.H2("General") ,           
                html.H4("Description générale des données"),

                                        html.H2("Analyse univariée"),
                                      html.H2("Matrice de corrélation"),
                                              html.H2("Analyse bivariée"),
                                       ]) ]),
        dcc.Tab( label='Pointés', children=[ 
            html.Div([
                html.H2("General"),          
                html.H4("Description générale des données"),

                 html.H2("Analyse univariée"),
                  html.H2("Matrice de corrélation"),
                   html.H2("Analyse bivariée"),
                                                            ]) '''
            ])  ]) ])


 

