# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 09:33:13 2022

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




layout = dbc.Row(
    [
        dbc.Col([
            html.Label("Emissions of MERCURE"),
            dcc.RadioItems(['Low','High'], value='High', id='emissions', labelStyle={'display': 'block'}),
        ], width=2),

        dbc.Col([
            dcc.Graph(id='my-graph', animate=True,
                      animation_options={'transition':{'duration': 750, 'ease': 'cubic-in-out'}}),
        ], width=10)

    ]
)