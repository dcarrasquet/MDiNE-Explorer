import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import os
import re
import base64

from maindash import app

def layout_home():
    return html.Div([
            html.H3('Welcome to the Home Page'),
            html.Div([
                html.P('The aim is to provide an easy-to-use graphical interface for estimating differential co-occurrence networks using the MDiNE model [ref].'),
                html.P('''MDiNE is a Bayesian hierarchical model that estimates species interactions from a table of counts and a table of covariates. 
It is possible to modify aprioris on various model parameters from a predefined list. 
A tutorial is available at: https://example.com'''),
                html.Img(src='/assets/co_occurence_networks.png', style={'width': '50%','display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'})
                
    
    ])
        ])