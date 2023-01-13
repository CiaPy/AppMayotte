# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 15:04:33 2023

@author: mato
"""


import logging
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn import tree
import plotly.figure_factory as ff
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, recall_score, precision_score,accuracy_score

from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
import dash_labs as dl



layout = dbc.Container([
html.Div([
    html.Br(),
    dcc.Markdown('# Application de prédiction de présence d interface'),
    html.Br(),
    dcc.Tabs(id='tabs', value='tab-intro', children=[
        dcc.Tab(label='Objectif', value='tab-intro'),
        dcc.Tab(label='Prediction', value='tab-predict'),
        dcc.Tab(label='Evaluation', value='tab-evaluate'),
        dcc.Tab(label='Interprétation', value='tab-explain'),

    ]),
    html.Div(id='tabs-content'),
])
])