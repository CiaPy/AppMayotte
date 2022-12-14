# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 15:38:38 2022

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
from dash import dash_table



from classes_py.data import get_datapointés,group_by_source,get_datahv,count_hv
from classes_py.model import Model
from classes_py.result import Result


from pages.accueil import layout as lay_home
from pages.statistiques import layout as lay_stat
from pages.geostatistiques import layout as lay_geostat



#Read image
img_greenrock =r'D:\Documents\mato\OneDrive - BRGM\Bureau\Aleasismique_Mayotte\Code\Application_EDA\dash_app\assets\GREEN-ROCK.jpg' # replace with your own image
encoded_imggr = base64.b64encode(open(img_greenrock, 'rb').read())

#Composants de l'application
#Navbar
navbar =dbc.NavbarSimple([
    dbc.NavItem(html.Img(src='data:image/jpg;base64,{}'.format(encoded_imggr.decode()))),
    dbc.NavItem(html.H2("Exploratory Data Analysis")),
    dbc.NavItem(dbc.NavLink("Accueil", href="/", active="exact")),
   dbc.NavItem(dbc.NavLink("Machine Learning", href="/page-1", active="exact")),
   # dbc.NavItem(dbc.NavLink("Geostastiques", href="/page-2", active="exact")), 
   
     ])

#Content
content = html.Div(id="page-content", style={ 'marginLeft' : '-18px',
            'marginBottom' : '5px'
         })

#Footer
fdivs =  [html.H2("Footer")]
    
footer = html.Div(fdivs, style={
    "position": "fixed",
    "bottom": 0,
    "left": 0,
    "right": 0,
    "height": "2rem 3rem",
    "padding": "1rem 1rem",
    "background-color": "white",
}                                          )




#My app
app = dash.Dash(__name__,external_stylesheets=[dbc.themes.LUX],use_pages=True,suppress_callback_exceptions=True)
server = app.server

#Structure de l'app
app.layout = dbc.Container([
                          html.Div([ dcc.Location(id="url",refresh=False), 
                          navbar, 
                          content,footer 
                          ]),
              dash.page_container  ],style={ 'marginBottom' : '5px'
                         },fluid=True)



# Update the pages
@callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])

def display_page(pathname):
    if pathname == '/':
        return lay_home
    elif pathname == '/page-1':
        return  lay_stat
    # elif pathname == '/page-2':
    #     return lay_geostat
    else :
        return '404' 


          

if __name__ == "__main__":

    app.run_server(debug=False, host="0.0.0.0", port=8080)