# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 09:32:38 2022

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
import dash_gif_component as gif



layout=              html.Div(
                    [
#                        dbc.Row([
               
#                       dbc.Col([  html.Div([
#     gif.GifPlayer(
#         gif='assets/Yl4w.gif', still = 'assets/earthcom-facebook.jpg'
#     )
# ]),]),
                      
                      dbc.Col([ html.Div(
                                  [
                                      html.H4(
                                          "Artificial Intelligence for Earth Science",
                                          style={"font-weight": "bold",'textAlign': 'center'},
                                      ),
                                      html.P([
                                          "Analyse exploratoire statistiques et géostatistiques des données " ,html.Br()," AEM, BSS et Profondeur Interfaces " 
                                     ] ,style={"font-weight": "bold",'textAlign': 'center'}),
                                  ]
                              ),]),
#                        dbc.Col([   html.Div([
#     gif.GifPlayer(
#         gif='assets/fin-brain2-1.gif', still= 'assets/image.png'
#     )
# ])])
#                     ],
#                     className="three column",
#                     id="title",
#                 )
                      

html.H1('', 
        style={'background-image': 'url(https://www.atterrir.com/wp-content/uploads/2016/08/mayotte.jpg)', 
        'background-size': '100%',
          'position': 'fixed',
          'width': '100%',
          'height': '100%'})    

], style={ 'marginRight' : '5px',
            'marginLeft' : '5px'})
