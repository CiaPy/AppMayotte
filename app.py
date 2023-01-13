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
from tabs import intro, prediction, explain, evaluation

import glob
import os

#from classes_py.data import get_datapointés,group_by_source,get_datahv,count_hv
from classes_py.RF import layout as lay_rf
# from classes_py.result import Result


from pages.accueil import layout as lay_home
#from pages.statistiques import layout as lay_stat
#from pages.geostatistiques import layout as lay_geostat
from pages.projet import layout as lay_projet


#Read image
img_greenrock =r'/assets/IA.png' # replace with your own image
encoded_imggr = base64.b64encode(open(img_greenrock, 'rb').read())


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

app.title = 'IA Mayotte'


navbar = dbc.Navbar(
    [
     dbc.Container(
        [
       
            dbc.Row(
                [
           dbc.Col(html.Img(src=app.get_asset_url("logo.png"), height="150px")),
           dbc.Col( dbc.NavItem(dbc.NavLink("Accueil", href="/", active="exact"))),
           dbc.Col( dbc.NavItem(dbc.NavLink("Le projet", href="/page-1", active="exact"))),
           dbc.Col( dbc.NavItem(dbc.NavLink("Machine Learning", href="/page-2", active="exact")),)
                ],className="flex-grow-1",
                align="center",
               
            ),
          
     ]),
        #dbc.NavbarToggler(id="navbar-toggler")
       
    ],
    color="white"
  
    
)

#Structure de l'app
app.layout = dbc.Container([
                          html.Div([ dcc.Location(id="url",refresh=False), 
                          navbar, 
                          content,
                          footer 
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
          return lay_projet
    elif pathname == '/page-2':
        return  lay_rf
    
    else :
        return '404' 


@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-intro': return intro.layout
    elif tab == 'tab-predict': return prediction.layout
    elif tab == 'tab-explain': return explain.layout
    elif tab == 'tab-evaluate': return evaluation.layout


if __name__ == "__main__":
    app.run_server(debug=True,  port=8080)
