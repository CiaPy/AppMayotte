# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 11:52:55 2023

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
import glob
import os

# local path
imagePath = r'D:\Documents\mato\OneDrive - BRGM\Bureau\Aleasismique_Mayotte\Code\Application_EDA\dash_app\assets\image_projet\\'

# string list of file names... 1-image.jpg, 2-image.jpg, etc
list_of_images = [os.path.basename(x) for x in glob.glob('{}*.jpg'.format(imagePath))]

# encoded list of bytes of images... used this before in Dash, will explain more 
encodedList = [base64.b64encode(open(imagePath + i, 'rb').read()) for i in list_of_images]

titleimg = ['Levé electromagnétique aéroporté', 'Carte de Mayotte', ' Effet de site lithologique']

layout =     dbc.Carousel(
        id="carousel",
        items = [
           {'src':'data:image/jpg;base64,{}'.format(x.decode())} for x in encodedList
            
         
            
            ],
        controls=True,
        indicators=True,
        interval=None,
    ),