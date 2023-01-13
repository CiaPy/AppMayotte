# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 16:41:31 2023

@author: mato
"""

import dash
from dash import html, dcc, callback, Input, Output
from dash.dependencies import State
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
import dash_labs as dl
import dash_gif_component as gif
from dash import dash_table
from tabs import intro, prediction, explain, evaluation

import glob
import os

layout = [dcc.Markdown("""
### Prediction
""")]